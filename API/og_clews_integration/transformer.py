import pandas as pd
import logging
from .schemas import OgCoreInputSchema

logger = logging.getLogger(__name__)

class DimensionalityBridge:
    """Bi-directional ETL pipeline for the OG-CLEWS converging module."""

    @staticmethod
    def forward_pass_aggregation(clews_df: pd.DataFrame, carbon_tax_rate: float = 50.0) -> OgCoreInputSchema:
        """Collapses multi-dimensional arrays into 1D macroeconomic time-series vectors."""
        emissions_df = clews_df[clews_df['Variable'] == 'AnnualEmissions']
        investment_df = clews_df[clews_df['Variable'] == 'CapitalInvestment']

        # Collapse dimensions to 1D Series indexed by Year
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
        """Calculates the raw, undampened scaling factor based on GDP shifts."""
        gdp_pct_change = og_core_gdp_vector.pct_change().fillna(0)
        return 1.0 + (gdp_pct_change * demand_elasticity)

    @staticmethod
    def apply_scale_vectorized(base_demand: pd.DataFrame, final_scaling_factors: pd.Series) -> pd.DataFrame:
        """Applies scaling factors using C-optimized pandas mapping (O(1) lookup)."""
        updated_demand = base_demand.copy()
        
        # Vectorized mapping eliminates the slow row-by-row iteration
        mapped_scales = updated_demand['Year'].map(final_scaling_factors).fillna(1.0)
        updated_demand['Value'] = updated_demand['Value'] * mapped_scales
        
        return updated_demand