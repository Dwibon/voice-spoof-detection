# src/model.py

import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


class Wav2Vec2Classifier(nn.Module):
    """
    Frozen Wav2Vec2 backbone + small trainable classifier head.
    Outputs a single logit (positive class = spoof), used with BCEWithLogitsLoss.
    """

    def __init__(self, model_name="facebook/wav2vec2-base", freeze_backbone=True):
        super().__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.backbone = Wav2Vec2Model.from_pretrained(model_name)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        hidden_dim = self.backbone.config.hidden_size  # 768 for base

        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),  # single logit output
        )

        self.freeze_backbone = freeze_backbone

    def forward(self, input_values):
        # input_values: (batch, num_samples) — already preprocessed/normalized
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(input_values)
        else:
            outputs = self.backbone(input_values)

        pooled = outputs.last_hidden_state.mean(dim=1)  # (batch, hidden_dim)
        logits = self.classifier_head(pooled)  # (batch, 1)
        return logits.squeeze(-1)  # (batch,)