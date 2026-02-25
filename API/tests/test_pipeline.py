import sys
import os
# Force Python to look in the parent directory (API/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
from pydantic import ValidationError
from og_clews_integration.schemas import ClewsOutputSchema
from og_clews_integration.transformer import DimensionalityBridge

def test_clews_schema_validation():
    """Test that the schema catches invalid negative prices using real OSeMOSYS variables."""
    with pytest.raises(ValidationError):
        # AnnualEmissions and Cost cannot be negative
        ClewsOutputSchema(iteration=1, AnnualEmissions=-10.0, TotalDiscountedCostByTechnology=500.0)

def test_forward_pass_aggregation():
    """Test the pandas ETL transformer collapses dimensions correctly."""
    # 1. Create a mock OSeMOSYS long-format dataframe
    mock_clews_df = pd.DataFrame({
        'Variable': ['AnnualEmissions', 'AnnualEmissions', 'CapitalInvestment', 'CapitalInvestment'],
        'Region': ['REG1', 'REG1', 'REG1', 'REG1'],
        'Dimension': ['CO2', 'CH4', 'SOLAR', 'WIND'],
        'Year': [2026, 2026, 2026, 2026],
        'Value': [100.0, 50.0, 1000.0, 2000.0]
    })
    
    # 2. Run it through our Dimensionality Bridge
    og_input = DimensionalityBridge.forward_pass_aggregation(mock_clews_df, carbon_tax_rate=0.1)
    
    # 3. Assert it collapsed the dimensions into 1D arrays properly
    assert og_input.years == [2026]
    # Total investment = 3000. Normalized (/1000) = 3.0
    assert og_input.energy_cost_index == [3.0]  
    # Total emissions = 150. Tax revenue (150 * 0.1) = 15.0
    assert og_input.carbon_tax_revenue == [15.0]