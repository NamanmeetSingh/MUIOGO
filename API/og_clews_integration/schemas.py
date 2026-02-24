from pydantic import BaseModel, Field

class ClewsOutputSchema(BaseModel):
    """Data structure expected as output from a CLEWS/OSeMOSYS run."""
    iteration: int = Field(ge=1, description="Current simulation loop iteration")
    avg_energy_price: float = Field(ge=0.0, description="Average energy price (e.g., USD/GJ)")
    total_emissions: float = Field(ge=0.0, description="Total CO2 emissions in kilotons")

class OgCoreInputSchema(BaseModel):
    """Data structure expected as input for an OG-Core macro run."""
    energy_cost_index: float = Field(ge=0.0, description="Normalized energy cost index")
    carbon_tax_revenue: float = Field(ge=0.0, description="Derived carbon tax revenue")

class OgCoreOutputSchema(BaseModel):
    """Data structure expected as output from an OG-Core run."""
    iteration: int = Field(ge=1, description="Current simulation loop iteration")
    gdp_growth_rate: float = Field(description="Percentage GDP growth rate")
    industrial_investment: float = Field(ge=0.0, description="Total capital investment")

class ClewsInputSchema(BaseModel):
    """Data structure expected as input for a CLEWS run."""
    macro_gdp_multiplier: float = Field(ge=0.1, description="Multiplier for energy demand projections")