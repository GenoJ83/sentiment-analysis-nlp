"""Regex text normalization and tokenizer encoding wrapper."""
import re
from typing import List

class TextCleaner:
    def __init__(self):
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.html_pattern = re.compile(r"<.*?>")
        self.punct_pattern = re.compile(r"[^\w\s]")

    def clean(self, text: str) -> str:
        text = text.lower()
        text = self.url_pattern.sub("", text)
        text = self.html_pattern.sub("", text)
        text = self.punct_pattern.sub(" ", text)
        return " ".join(text.split())

    def clean_batch(self, texts: List[str]) -> List[str]:
        return [self.clean(t) for t in texts]
