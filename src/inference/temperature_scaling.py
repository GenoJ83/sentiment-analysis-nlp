"""Temperature scaling model confidence calibration."""
import numpy as np

class CalibratedPredictor:
    def __init__(self, temperature: float = 1.25):
        self.temperature = max(0.1, temperature)

    def calibrate_logits(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.temperature
        exp_logits = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
