"""Corpus loader and synthetic consumer review generator."""
import pandas as pd
import numpy as np

SENTIMENT_CLASSES = ["Negative", "Neutral", "Positive"]

def generate_review_corpus(n_samples: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    pos_phrases = ["exceptional build quality", "fast shipping and great support", "works like a charm", "highly recommend to everyone", "stellar performance"]
    neg_phrases = ["stopped working after two days", "terrible customer support", "defective item and cheap plastic", "complete waste of money", "battery drains instantly"]
    neu_phrases = ["average item as described", "acceptable for the price", "mediocre packaging but functions", "neither good nor bad", "standard delivery timeline"]

    texts, labels = [], []
    for _ in range(n_samples):
        label_idx = rng.integers(0, 3)
        if label_idx == 0:
            text = f"Review: {rng.choice(neg_phrases)}! In addition, {rng.choice(neg_phrases)}."
        elif label_idx == 1:
            text = f"Review: {rng.choice(neu_phrases)}. Overall {rng.choice(neu_phrases)}."
        else:
            text = f"Review: {rng.choice(pos_phrases)}! Truly {rng.choice(pos_phrases)}."
        texts.append(text)
        labels.append(label_idx)

    return pd.DataFrame({"text": texts, "label": labels, "sentiment": [SENTIMENT_CLASSES[i] for i in labels]})

if __name__ == "__main__":
    df = generate_review_corpus(500)
    print(f"Generated {len(df)} customer reviews across 3 sentiment classes.")
