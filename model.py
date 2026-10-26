"""Frequency fingerprint extractor and lightweight classification heads."""

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def load_denoiser(checkpoint_path):
    """Load the original 17-layer Corvi DnCNN without changing weight keys."""
    layers = []
    for index in range(17):
        use_bn = 0 < index < 16
        layers.append(
            nn.Conv2d(
                3 if index == 0 else 64,
                3 if index == 16 else 64,
                kernel_size=3,
                padding=0,
                bias=not use_bn,
            )
        )
        if use_bn:
            layers.append(nn.BatchNorm2d(64))
        if index < 16:
            layers.append(nn.ReLU(inplace=True))
    network = nn.Sequential(*layers)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    network.load_state_dict(checkpoint["network"])
    return network.requires_grad_(False).eval()


class FingerprintExtractor(nn.Module):
    """Extract embeddings from log-magnitude spectra of frozen noise residuals."""

    def __init__(self, denoiser_checkpoint, feature_dim=128, pretrained=True):
        super().__init__()
        self.denoiser = load_denoiser(denoiser_checkpoint)
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.base = resnet50(weights=weights)
        self.base.fc = nn.Linear(self.base.fc.in_features, feature_dim)

    def train(self, mode=True):
        super().train(mode)
        # Frozen BatchNorm statistics must remain fixed during extractor training.
        self.denoiser.eval()
        return self

    def forward(self, images):
        with torch.no_grad():
            images = (images.clamp(-1, 1) + 1) / 2
            residuals = self.denoiser(images) / 256.0
            spectrum = torch.log1p(torch.fft.fft2(residuals).abs())
        return self.base(spectrum)


class Classifier(nn.Module):
    """Binary head used for feature pairs or optional real/fake supervision."""

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.fc(features).squeeze(1)
