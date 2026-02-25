import pandas as pd
import numpy as np
import logging
from pydantic import BaseModel, Field
from typing import List

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# SCHEMAS (From schemas.py)
class OgCoreInputSchema(BaseModel):
    years: List[int]
    energy_cost_index: List[float]
    carbon_tax_revenue: List[float]

# ETL PIPELINE (From transformer.py)
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

# ORCHESTRATOR (From converger.py)
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

# MOCK EXECUTION & TESTING
def generate_mock_clews_data(demand_modifier: float = 1.0) -> pd.DataFrame:
    """Generates dummy long-format OSeMOSYS outputs for years 2025-2035."""
    years = list(range(2025, 2036))
    data = []
    for y in years:
        # Assumed Relation: higher demand = higher emissions and investments
        data.append(['AnnualEmissions', 'REG1', 'CO2', y, 500 * demand_modifier * (1 + (y-2025)*0.02)])
        data.append(['CapitalInvestment', 'REG1', 'SOLAR', y, 2000 * demand_modifier * (1 + (y-2025)*0.05)])
    return pd.DataFrame(data, columns=['Variable', 'Region', 'Dimension', 'Year', 'Value'])

def run_mock_ogcore(macro_inputs: OgCoreInputSchema) -> pd.Series:
    """Simulates OG-Core outputting a GDP time series based on energy costs."""
    years = macro_inputs.years
    energy_costs = np.array(macro_inputs.energy_cost_index)
    
    # Mock relation: High energy costs drag down the baseline 3% GDP growth
    base_gdp = 1000.0
    gdp_series = []
    
    for i, cost in enumerate(energy_costs):
        growth_rate = 0.03 - (cost * 0.005) # Cost drag
        base_gdp = base_gdp * (1 + growth_rate)
        gdp_series.append(base_gdp)
        
    return pd.Series(gdp_series, index=years)

def test_full_pipeline():
    logger.info("Starting Full Pipeline Test...")
    
    # Baseline Demand Profile (CLEWS Input)
    base_demand = pd.DataFrame({
        'Variable': ['SpecifiedDemandProfile'] * 11,
        'Region': ['REG1'] * 11,
        'Dimension': ['ELC'] * 11,
        'Year': list(range(2025, 2036)),
        'Value': [100.0] * 11
    })

    orchestrator = ConvergingOrchestrator(alpha=0.25, tolerance=1e-4)
    current_demand_modifier = 1.0

    for i in range(1, 15):
        # Run CLEWS (Mock)
        clews_output_df = generate_mock_clews_data(demand_modifier=current_demand_modifier)
        
        # ETL Forward Pass (CLEWS -> OG-Core)
        og_inputs = DimensionalityBridge.forward_pass_aggregation(clews_output_df)
        
        # Run OG-Core (Mock)
        og_gdp_vector = run_mock_ogcore(og_inputs)
        
        # Orchestrate & Apply Dampening (OG-Core -> CLEWS)
        updated_demand_df, has_converged = orchestrator.run_iteration(i, og_gdp_vector, base_demand)
        
        if has_converged:
            break
            
        # (For this mock, we just take the mean scale to feed the generator)
        current_demand_modifier = updated_demand_df['Value'].mean() / 100.0

if __name__ == "__main__":
    test_full_pipeline()