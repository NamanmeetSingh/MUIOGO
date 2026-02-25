import numpy as np
import pandas as pd
import logging
from .transformer import DimensionalityBridge

logger = logging.getLogger(__name__)

class ConvergingOrchestrator:
    """
    Manages the stateful iterative loop between CLEWS and OG-Core.
    Applies mathematical dampening to prevent infinite oscillation.
    """
    def __init__(self, alpha: float = 0.2, tolerance: float = 1e-4, max_iterations: int = 20):
        self.alpha = alpha  # Dampening factor
        self.tolerance = tolerance # Epsilon threshold
        self.max_iterations = max_iterations
        self.previous_scale = None # Tracks Scale_{n-1}

    def _apply_dampening(self, calculated_scale: pd.Series) -> pd.Series:
        """
        Applies the dampening factor to smoothly glide models toward equilibrium.
        Formula: Scale_n = Scale_{n-1} + alpha * (Scale_calculated - Scale_{n-1})
        """
        if self.previous_scale is None:
            self.previous_scale = pd.Series(1.0, index=calculated_scale.index)

        # Align indices to ensure safe vector math
        calc, prev = calculated_scale.align(self.previous_scale, fill_value=1.0)
        
        # Calculate dampened scale
        dampened_scale = prev + self.alpha * (calc - prev)
        return dampened_scale

    def check_convergence(self, current_scale: pd.Series) -> bool:
        """Evaluates the maximum absolute difference between iterations."""
        if self.previous_scale is None:
            return False

        calc, prev = current_scale.align(self.previous_scale, fill_value=1.0)
        max_diff = np.max(np.abs(calc - prev))
        
        logger.info(f"Convergence Delta: {max_diff:.6f} (Threshold: {self.tolerance})")
        return bool(max_diff < self.tolerance)

    def run_iteration(self, n: int, og_gdp_output: pd.Series, base_demand: pd.DataFrame) -> pd.DataFrame:
        """Executes a single step of the integration loop."""
        logger.info(f"--- Iteration {n} ---")
        
        raw_scale = DimensionalityBridge.calculate_raw_scale(og_gdp_output)
        final_scale = self._apply_dampening(raw_scale)
        
        if self.check_convergence(final_scale):
            logger.info(f"SUCCESS: Models reached dynamic equilibrium at iteration {n}.")
            return None # Signals the main runner to halt

        # Update state and return the newly scaled physical parameters
        self.previous_scale = final_scale
        return DimensionalityBridge.apply_scale_vectorized(base_demand, final_scale)