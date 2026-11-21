import pytest
pytest.importorskip("torch")
"""Unit tests for text preprocessing, transformer model, and API."""
from src.preprocessing.text_cleaner import TextCleaner
from src.models.train_transformer import TransformerSentimentClassifier
from src.inference.onnx_pipeline import SentimentInferenceEngine
import torch

def test_text_cleaner():
    cleaner = TextCleaner()
    res = cleaner.clean("Amazing phone! Check https://example.com <p>now</p>.")
    assert "https" not in res
    assert "<p>" not in res
    assert "amazing phone" in res

def test_transformer_forward():
    model = TransformerSentimentClassifier()
    x = torch.randint(0, 5000, (2, 16))
    out = model(x)
    assert out.shape == (2, 3)

def test_inference_engine():
    model = TransformerSentimentClassifier()
    engine = SentimentInferenceEngine(model)
    res = engine.predict("Great battery life!")
    assert res["sentiment"] in ["Negative", "Neutral", "Positive"]
