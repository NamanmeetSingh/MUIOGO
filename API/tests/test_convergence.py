import pandas as pd
import numpy as np
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 1. SCHEMAS (From schemas.py)
class OgCoreInputSchema(BaseModel):
    years: List[int]
    energy_cost_index: List[float]
    carbon_tax_revenue: List[float]

# 2. ETL PIPELINE (From transformer.py)
class DimensionalityBridge:
    @staticmethod
    def forward_pass_aggregation(clews_df: pd.DataFrame, carbon_tax_rate: float = 50.0) -> OgCoreInputSchema:
        emissions_df = clews_df[clews_df['Variable'] == 'AnnualEmissions']
        investment_df = clews_df[clews_df['Variable'] == 'CapitalInvestment']

        total_emissions = emissions_df.groupby('Year')['Value'].sum()
        total_investment = investment_df.groupby('Year')['Value'].sum()

        tax_revenue = total_emissions * carbon_tax_rate

        return OgCoreInputSchema(
            years=total_emissions.index.tolist(),
            energy_cost_index=(total_investment / 1000.0).tolist(),
            carbon_tax_revenue=tax_revenue.tolist()
        )

    @staticmethod
    def calculate_raw_scale(og_core_gdp_vector: pd.Series, demand_elasticity: float = 0.8) -> pd.Series:
        gdp_pct_change = og_core_gdp_vector.pct_change().fillna(0)
        return 1.0 + (gdp_pct_change * demand_elasticity)

    @staticmethod
    def apply_scale_vectorized(base_demand: pd.DataFrame, final_scaling_factors: pd.Series) -> pd.DataFrame:
        updated_demand = base_demand.copy()
        mapped_scales = updated_demand['Year'].map(final_scaling_factors).fillna(1.0)
        updated_demand['Value'] = updated_demand['Value'] * mapped_scales
        return updated_demand

# 3. ORCHESTRATOR (From converger.py)
class ConvergingOrchestrator:
    def __init__(self, alpha: float = 0.3, tolerance: float = 1e-4, max_iterations: int = 20):
        self.alpha = alpha  # Dampening factor
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.previous_scale = None

    def _apply_dampening(self, calculated_scale: pd.Series) -> pd.Series:
        if self.previous_scale is None:
            self.previous_scale = pd.Series(1.0, index=calculated_scale.index)
        
        calc, prev = calculated_scale.align(self.previous_scale, fill_value=1.0)
        return prev + self.alpha * (calc - prev)

    def check_convergence(self, current_scale: pd.Series) -> bool:
        if self.previous_scale is None:
            return False
        calc, prev = current_scale.align(self.previous_scale, fill_value=1.0)
        max_diff = np.max(np.abs(calc - prev))
        logger.info(f"Convergence Delta: {max_diff:.6f} (Threshold: {self.tolerance})")
        return bool(max_diff < self.tolerance)

    def run_iteration(self, n: int, og_gdp_output: pd.Series, base_demand: pd.DataFrame):
        logger.info(f"--- Iteration {n} ---")
        raw_scale = DimensionalityBridge.calculate_raw_scale(og_gdp_output)
        final_scale = self._apply_dampening(raw_scale)
        
        if self.check_convergence(final_scale):
            logger.info(f"SUCCESS: Models reached dynamic equilibrium at iteration {n}.")
            return None, True
            
        self.previous_scale = final_scale
        updated_demand = DimensionalityBridge.apply_scale_vectorized(base_demand, final_scale)
        return updated_demand, False

# 4. REAL DATA EXTRACTION & TESTING
def extract_real_clews_data(csv_folder: Path) -> pd.DataFrame:
    """Reads official OSeMOSYS CSVs and normalizes them for the ETL Bridge."""
    
    # 1. Ingest Emissions
    emissions_file = csv_folder / "AnnualTechnologyEmission.csv"
    if not emissions_file.exists():
        raise FileNotFoundError(f"Missing demo data: {emissions_file}")
        
    df_emissions = pd.read_csv(emissions_file)
    df_emissions = df_emissions.rename(columns={'y': 'Year', 'AnnualTechnologyEmission': 'Value'})
    df_emissions['Variable'] = 'AnnualEmissions'

    # 2. Ingest Capital Investment
    investment_file = csv_folder / "CapitalInvestment.csv"
    if not investment_file.exists():
        raise FileNotFoundError(f"Missing demo data: {investment_file}")
        
    df_investment = pd.read_csv(investment_file)
    df_investment = df_investment.rename(columns={'y': 'Year', 'CapitalInvestment': 'Value'})
    df_investment['Variable'] = 'CapitalInvestment'

    # 3. Combine into a single unified Dataframe
    return pd.concat([df_emissions, df_investment], ignore_index=True)

def run_mock_ogcore(macro_inputs: OgCoreInputSchema) -> pd.Series:
    """Simulates OG-Core outputting a GDP time series based on energy costs."""
    years = macro_inputs.years
    energy_costs = np.array(macro_inputs.energy_cost_index)
    
    base_gdp = 1000.0
    gdp_series = []
    
    for cost in energy_costs:
        # High energy investments cause a slight drag on baseline 3% growth
        growth_rate = 0.03 - (cost * 0.0005) 
        base_gdp = base_gdp * (1 + growth_rate)
        gdp_series.append(base_gdp)
        
    return pd.Series(gdp_series, index=years)

def test_real_pipeline():
    logger.info("Starting Full Pipeline Test using official CLEWs.Demo data...")
    
    # Path to the newly extracted mentor demo data
    demo_csv_path = Path("WebAPP/DataStorage/CLEWs Demo/res/REF/csv")
    
    # Baseline Demand Profile (Mocked for reverse pass)
    base_demand = pd.DataFrame({
        'Year': list(range(2020, 2071)),
        'Value': [100.0] * 51
    })

    orchestrator = ConvergingOrchestrator(alpha=0.3, tolerance=1e-4)

    for i in range(1, 15):
        # 1. Extract Real Data
        try:
            clews_output_df = extract_real_clews_data(demo_csv_path)
        except Exception as e:
            logger.error(f"Data Extraction Failed: {e}")
            return
            
        # 2. ETL Forward Pass (CLEWS -> OG-Core)
        og_inputs = DimensionalityBridge.forward_pass_aggregation(clews_output_df)
        
        if i == 1:
            logger.info(f"REAL DATA PARSED. Time Horizon: {og_inputs.years[0]} to {og_inputs.years[-1]}")
            logger.info(f"Carbon Tax Revenue Vector (First 3): {np.round(og_inputs.carbon_tax_revenue[:3], 2)}")
        
        # 3. Run OG-Core (Mock)
        og_gdp_vector = run_mock_ogcore(og_inputs)
        
        # 4. Orchestrate & Apply Dampening (OG-Core -> CLEWS)
        updated_demand_df, has_converged = orchestrator.run_iteration(i, og_gdp_vector, base_demand)
        
        if has_converged:
            break

if __name__ == "__main__":
    test_real_pipeline()