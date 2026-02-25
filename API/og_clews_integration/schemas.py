from pydantic import BaseModel, Field
from typing import List


class ClewsOutputSchema(BaseModel):
    """Data structure expected as output from a CLEWS/OSeMOSYS run."""
    iteration: int = Field(ge=1, description="Current simulation loop iteration")
    
    # Mapped directly to OSeMOSYS VARIABLES_C dictionary
    AnnualEmissions: float = Field(ge=0.0, description="Total annual emissions (r, e, y)")
    TotalDiscountedCostByTechnology: float = Field(ge=0.0, description="Total system cost (r, t, y)")

class OgCoreInputSchema(BaseModel):
    """Data structure expected as input for an OG-Core macro run over time path T."""
    
    years: List[int] = Field(description="Array of overlapping years mapping to OG-Core time periods")
    energy_cost_index: List[float] = Field(description="1D vector of normalized energy cost indices over T")
    carbon_tax_revenue: List[float] = Field(description="1D vector of derived carbon tax revenues over T")

class OgCoreOutputSchema(BaseModel):
    """Data structure expected as output from an OG-Core run."""
    
    iteration: int = Field(ge=1, description="Current simulation loop iteration")
    gdp_growth_rate: float = Field(description="Percentage GDP growth rate")

class ClewsInputSchema(BaseModel):
    """Data structure expected as input for a CLEWS run."""
    
    # Mapped directly to OSeMOSYS PARAMETERS_C dictionary
    SpecifiedDemandProfile: float = Field(ge=0.1, description="Demand profile modifier (r, f, y, l)")
    DiscountRate: float = Field(ge=0.0, description="Macro-adjusted discount rate (r)")