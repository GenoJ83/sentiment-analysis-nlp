"""Token attribution and self-attention weight extraction."""
import numpy as np

def extract_token_salience(tokens, weights):
    """Normalize token importance weights for explainability."""
    scores = np.array(weights)
    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return list(zip(tokens, norm_scores.tolist()))
