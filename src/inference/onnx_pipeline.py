"""Fast token inference engine for production deployment."""
import torch
from src.data.ingest import SENTIMENT_CLASSES
from src.preprocessing.text_cleaner import TextCleaner

class SentimentInferenceEngine:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.cleaner = TextCleaner()

    def predict(self, text: str):
        cleaned = self.cleaner.clean(text)
        # Hash mapping for tokenizer simulation
        tokens = [hash(word) % 5000 for word in cleaned.split()][:32] or [0]
        input_ids = torch.tensor([tokens], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(input_ids)
            probs = torch.softmax(logits, dim=1)[0].numpy()
            pred_idx = int(probs.argmax())

        return {
            "input_text": text,
            "cleaned_text": cleaned,
            "sentiment": SENTIMENT_CLASSES[pred_idx],
            "confidence": float(round(probs[pred_idx], 4)),
            "probabilities": {SENTIMENT_CLASSES[i]: float(round(probs[i], 4)) for i in range(3)}
        }
