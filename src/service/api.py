"""FastAPI application for real-time text sentiment scoring."""
from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.models.train_transformer import TransformerSentimentClassifier
from src.inference.onnx_pipeline import SentimentInferenceEngine

app = FastAPI(title="NLP Sentiment Intelligence API", version="1.0.0")

class TextPayload(BaseModel):
    text: str = Field(..., min_length=2, example="The product exceeded all my expectations!")

model = TransformerSentimentClassifier()
engine = SentimentInferenceEngine(model)

@app.get("/health")
def health():
    return {"status": "ready", "model": "DistilBERT-Sentiment"}

@app.post("/analyze")
def analyze_sentiment(payload: TextPayload):
    return engine.predict(payload.text)
