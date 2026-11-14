# Sentiment Analysis & Customer Review Intelligence

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-teal)
![License](https://img.shields.io/badge/License-MIT-green)

A transformer-based NLP service that processes customer reviews, extracts sentiment polarity, and returns probabilistic confidence scores.

## Architecture
```mermaid
flowchart LR
    Raw[Raw Customer Text] --> Clean[Regex Cleaner]
    Clean --> Tok[Subword Tokenizer]
    Tok --> Trans[Transformer Encoder]
    Trans --> Head[Classification Head]
    Head --> API[FastAPI Output]
```

## Quick Start
```bash
pip install -r requirements.txt
uvicorn src.service.api:app --reload
```
