from .schemas import ClewsOutputSchema, OgCoreInputSchema, OgCoreOutputSchema, ClewsInputSchema
import logging

logger = logging.getLogger(__name__)

class ModelTransformer:
    """Handles the Extract-Transform-Load (ETL) mapping between CLEWS and OG-Core."""
    
    @staticmethod
    def clews_to_ogcore(clews_data: ClewsOutputSchema, carbon_tax_rate: float = 0.05) -> OgCoreInputSchema:
        """Transforms energy system outputs into macroeconomic inputs."""
        logger.debug(f"Transforming CLEWS iteration {clews_data.iteration} for OG-Core.")
        
        # Simulated Pandas transformation logic
        energy_index = clews_data.avg_energy_price / 50.0  # Normalize against a baseline
        tax_revenue = clews_data.total_emissions * carbon_tax_rate
        
        return OgCoreInputSchema(
            energy_cost_index=energy_index,
            carbon_tax_revenue=tax_revenue
        )

    @staticmethod
    def ogcore_to_clews(og_data: OgCoreOutputSchema) -> ClewsInputSchema:
        """Transforms macroeconomic outputs into energy system inputs."""
        logger.debug(f"Transforming OG-Core iteration {og_data.iteration} for CLEWS.")
        
        # Simulated baseline GDP mapping
        gdp_multiplier = 1.0 + (og_data.gdp_growth_rate / 100.0)
        
        return ClewsInputSchema(
            macro_gdp_multiplier=gdp_multiplier
        )