import logging
import numpy as np
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_nd_convergence(old_vector: np.ndarray, new_vector: np.ndarray, tolerance: float) -> Tuple[bool, float]:
    """
    Calculates the maximum absolute percentage change across all monitored variables.
    Returns a tuple of (has_converged, max_delta).
    """
    # Add a tiny epsilon to the denominator to prevent division by zero
    epsilon = 1e-9
    deltas = np.abs((new_vector - old_vector) / (old_vector + epsilon))
    max_delta = np.max(deltas)
    return max_delta <= tolerance, float(max_delta)

def run_converging_simulation(max_iterations: int = 15, tolerance: float = 0.005, alpha: float = 0.5):
    """
    Executes the two-way coupled run evaluating N-dimensional convergence.
    """
    logger.info("Initializing N-Dimensional OG-CLEWS Convergence...")
    
    # We now track an array of macro variables: e.g., [GDP Growth, Interest Rate, Carbon Tax Rev]
    # In production, these map to the outputs from the pandas ETL layer
    previous_macro_state = np.array([3.5, 2.0, 100.0]) 
    
    for i in range(1, max_iterations + 1):
        logger.info(f"--- Iteration {i} ---")
        
        # ... (Mock execution & ETL steps remain structurally similar, but pass arrays) ...
        # Simulate an OG-Core calculation returning a new vector
        calculated_state = previous_macro_state + np.random.uniform(-0.5, 0.5, 3) / i 
        
        # Apply Dampening Factor (alpha) across the entire vector
        dampened_state = (alpha * calculated_state) + ((1 - alpha) * previous_macro_state)
        
        # N-Dimensional Convergence Check
        converged, max_delta = check_nd_convergence(previous_macro_state, dampened_state, tolerance)
        
        logger.info(f"State Vector: {np.round(dampened_state, 3)} | Max Delta: {max_delta:.4f}")
        
        if converged:
            logger.info(f"SUCCESS: System achieved multi-dimensional convergence after {i} iterations.")
            break
            
        if i == max_iterations:
            logger.warning("WARNING: Maximum iterations reached without convergence.")
            
        # Update state vector for the next loop
        previous_macro_state = dampened_state

if __name__ == "__main__":
    run_converging_simulation()