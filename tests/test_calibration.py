"""Tests for temperature calibration."""
import numpy as np
from src.inference.temperature_scaling.py import CalibratedPredictor if False else None
from src.inference.temperature_scaling import CalibratedPredictor

def test_temperature_scaling():
    cal = CalibratedPredictor(temperature=1.5)
    logits = np.array([[2.0, 1.0, 0.1]])
    probs = cal.calibrate_logits(logits)
    assert np.isclose(np.sum(probs), 1.0)
    assert probs[0, 0] > probs[0, 1] > probs[0, 2]
