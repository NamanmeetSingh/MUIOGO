import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from pydantic import ValidationError

from og_clews_integration.schemas import ClewsOutputSchema
from og_clews_integration.transformer import ModelTransformer

def test_clews_schema_validation():
    """Test that the schema catches invalid negative prices."""
    with pytest.raises(ValidationError):
        ClewsOutputSchema(iteration=1, avg_energy_price=-10.0, total_emissions=500.0)

def test_transformer_logic():
    """Test the math in the ETL transformer."""
    mock_clews = ClewsOutputSchema(iteration=1, avg_energy_price=100.0, total_emissions=1000.0)
    og_input = ModelTransformer.clews_to_ogcore(mock_clews, carbon_tax_rate=0.1)
    
    assert og_input.energy_cost_index == 2.0  # 100.0 / 50.0
    assert og_input.carbon_tax_revenue == 100.0 # 1000.0 * 0.1

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))