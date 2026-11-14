"""Transformer architecture fine-tuning pipeline simulation."""
import torch
import torch.nn as nn

class TransformerSentimentClassifier(nn.Module):
    def __init__(self, vocab_size: int = 5000, embed_dim: int = 128, num_classes: int = 3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=256, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pooler = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        embeds = self.embedding(input_ids)
        hidden = self.transformer(embeds) # (batch, seq, embed_dim)
        pooled = self.pooler(hidden.transpose(1, 2)).squeeze(2)
        logits = self.classifier(pooled)
        return logits
