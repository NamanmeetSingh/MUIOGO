from pydantic import BaseModel, Field

class ClewsOutputSchema(BaseModel):
    """Data structure expected as output from a CLEWS/OSeMOSYS run."""
    iteration: int = Field(ge=1, description="Current simulation loop iteration")
    # Mapped directly to OSeMOSYS VARIABLES_C dictionary
    AnnualEmissions: float = Field(ge=0.0, description="Total annual emissions (r, e, y)")
    TotalDiscountedCostByTechnology: float = Field(ge=0.0, description="Total system cost (r, t, y)")

class OgCoreInputSchema(BaseModel):
    """Data structure expected as input for an OG-Core macro run."""
    energy_cost_index: float = Field(ge=0.0, description="Normalized energy cost index")
    carbon_tax_revenue: float = Field(ge=0.0, description="Derived carbon tax revenue")

class OgCoreOutputSchema(BaseModel):
    """Data structure expected as output from an OG-Core run."""
    iteration: int = Field(ge=1, description="Current simulation loop iteration")
    gdp_growth_rate: float = Field(description="Percentage GDP growth rate")

class ClewsInputSchema(BaseModel):
    """Data structure expected as input for a CLEWS run."""
    # Mapped directly to OSeMOSYS PARAMETERS_C dictionary
    SpecifiedDemandProfile: float = Field(ge=0.1, description="Demand profile modifier (r, f, y, l)")
    DiscountRate: float = Field(ge=0.0, description="Macro-adjusted discount rate (r)")