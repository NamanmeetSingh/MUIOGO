import pandas as pd
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DimensionalityBridge:
    """
    ETL Pipeline to map high-dimensional OSeMOSYS outputs 
    into macroeconomic scalar vectors for OG-Core ingestion.
    """
    def __init__(self, osemosys_output_dir: str):
        self.output_dir = Path(osemosys_output_dir)

    def extract_macro_shock(self, target_csv: str = "CapitalInvestment.csv") -> dict:
        """
        Reads a standard OSeMOSYS CSV, aggregates across all regions and technologies, 
        and returns a 1D vector of total investment per year.
        """
        file_path = self.output_dir / target_csv
        
        if not file_path.exists():
            logger.error(f"ETL Failure: {target_csv} not found at {file_path}")
            raise FileNotFoundError(f"Missing OSeMOSYS output: {target_csv}")

        try:
            df = pd.read_csv(file_path)
            
            # Verify expected schema exists
            expected_columns = {'REGION', 'TECHNOLOGY', 'YEAR', 'VALUE'}
            if not expected_columns.issubset(set(df.columns)):
                raise ValueError(f"Schema mismatch in {target_csv}. Expected columns: {expected_columns}")

            # The Core ETL Logic: Squash dimensions down to a single Year -> Value vector
            yearly_totals = df.groupby('YEAR')['VALUE'].sum()
            shock_vector = yearly_totals.to_dict()

            return shock_vector

        except Exception as e:
            logger.error(f"Failed to process OSeMOSYS data: {str(e)}")
            raise

    def generate_ogcore_payload(self) -> str:
        """
        Packages the aggregated vectors into the JSON schema required by the API.
        """
        logger.info("Initializing Dimensionality Bridge extraction...")
        
        capital_investment = self.extract_macro_shock("CapitalInvestment.csv")
        # We can easily expand this to extract other files like TotalDiscountedCost.csv later
        
        payload = {
            "metadata": {
                "source": "OSeMOSYS_CLEWS",
                "integration_mode": "Two-Pass Feasibility",
                "status": "ready_for_macro"
            },
            "macro_shocks": {
                "total_capital_investment": capital_investment
            }
        }
        
        return json.dumps(payload, indent=4)

# Local Testing Block
if __name__ == "__main__":
    # If you ever want to test this locally without the full app:
    # 1. Create a dummy folder called 'test_outputs'
    # 2. Put a fake 'CapitalInvestment.csv' inside it
    # 3. Run this script!
    
    # bridge = DimensionalityBridge("./test_outputs")
    # print(bridge.generate_ogcore_payload())
    pass