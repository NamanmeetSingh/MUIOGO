from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any

class MacroBaselineSchema(BaseModel):
    """
    Strict data contract for incoming OG-Core macroeconomic baseline data.
    Note: This schema defines the core required subset for the Track 1 coupling.
    Additional parameters are captured dynamically to support model expansion.
    """
    
    # Allow extra fields to be passed without throwing a validation error
    model_config = ConfigDict(extra='allow')
    
    # --- Demographics ---
    population_growth_rate: float = Field(
        ge=-0.1, le=0.1, 
        description="Annual population growth rate (e.g., 0.02 for 2%)"
    )
    
    # --- Preferences & Labor ---
    frisch: float = Field(
        gt=0.0, default=0.4,
        description="Frisch elasticity of labor supply"
    )
    
    # --- Technology & Production ---
    g_y_annual: float = Field(
        ge=-0.1, le=0.2, default=0.03,
        description="Annual labor productivity growth rate"
    )
    Z: float = Field(
        gt=0.0, default=1.0,
        description="Total factor productivity scaling parameter"
    )
    
    # --- Fiscal Policy (Taxes) ---
    labor_income_tax_rate: float = Field(
        ge=0.0, le=1.0, 
        description="Baseline effective tax rate on labor income"
    )
    capital_income_tax_rate: float = Field(
        ge=0.0, le=1.0, 
        description="Baseline effective tax rate on capital income"
    )
    corporate_tax_rate: float = Field(
        ge=0.0, le=1.0, 
        description="Baseline corporate tax rate"
    )
    
    # --- Economy ---
    starting_gdp: float = Field(
        gt=0.0, 
        description="Baseline Gross Domestic Product in billions (local currency or USD)"
    )
    
    # --- Scalability ---
    additional_parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Dynamic dictionary for extending OG-Core parameters before the final UI schema is locked."
    )