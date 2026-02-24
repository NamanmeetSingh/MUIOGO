import logging
from .schemas import ClewsOutputSchema, OgCoreOutputSchema
from .transformer import ModelTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def mock_clews_run(inputs: dict, iteration: int) -> ClewsOutputSchema:
    """Mock execution of the OSeMOSYS solver."""
    multiplier = inputs.get('macro_gdp_multiplier', 1.0)
    return ClewsOutputSchema(
        iteration=iteration,
        avg_energy_price=50.0 * multiplier,
        total_emissions=1000.0 * multiplier
    )

def mock_ogcore_run(inputs: dict, iteration: int) -> OgCoreOutputSchema:
    """Mock execution of the OG-Core overlapping generations model."""
    cost_index = inputs.get('energy_cost_index', 1.0)
    simulated_growth = 3.5 - ((cost_index - 1.0) * 2.0)
    return OgCoreOutputSchema(
        iteration=iteration,
        gdp_growth_rate=simulated_growth,
        industrial_investment=500.0
    )

def run_converging_simulation(max_iterations: int = 15, tolerance: float = 0.005, alpha: float = 0.5):
    """
    Executes the two-way coupled run until macroeconomic outputs stabilize.
    Uses dampening factor (alpha) to prevent oscillation.
    """
    logger.info("Initializing OG-CLEWS Converging Simulation...")
    
    current_clews_input = {'macro_gdp_multiplier': 1.0}
    previous_gdp_growth = 3.5 # Assumed baseline
    
    for i in range(1, max_iterations + 1):
        logger.info(f"--- Iteration {i} ---")
        
        # 1. Run CLEWS and Transform
        clews_out = mock_clews_run(current_clews_input, i)
        og_input = ModelTransformer.clews_to_ogcore(clews_out)
        
        # 2. Run OG-Core
        og_out = mock_ogcore_run(og_input.model_dump(), i)
        
        # 3. Apply Dampening Factor to the Target Variable
        calculated_gdp = og_out.gdp_growth_rate
        dampened_gdp = (alpha * calculated_gdp) + ((1 - alpha) * previous_gdp_growth)
        
        # 4. Check Convergence
        delta = abs((dampened_gdp - previous_gdp_growth) / previous_gdp_growth)
        logger.info(f"GDP Growth: {dampened_gdp:.4f}% | Delta: {delta:.4f}")
        
        if delta <= tolerance:
            logger.info(f"SUCCESS: Convergence achieved after {i} iterations.")
            break
            
        if i == max_iterations:
            logger.warning("WARNING: Maximum iterations reached without convergence.")
            
        # Update state for next loop
        previous_gdp_growth = dampened_gdp
        next_clews_input = ModelTransformer.ogcore_to_clews(og_out)
        current_clews_input = next_clews_input.model_dump()

if __name__ == "__main__":
    run_converging_simulation()