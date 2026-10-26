"""Supervised contrastive learning and balanced feature-pair supervision."""

import torch
from torch.nn import functional as F


def supervised_contrastive_loss(features, labels, temperature, negative_weight=0.0):
    """Combine supervised contrastive loss with a clamped negative similarity term."""
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    features = F.normalize(features, dim=1)
    similarity = features @ features.T / temperature
    off_diagonal = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = labels[:, None].eq(labels[None, :]) & off_diagonal
    negative = ~labels[:, None].eq(labels[None, :]) & off_diagonal
    if not positive.any(dim=1).all():
        raise ValueError("Each anchor needs at least one same-class positive sample.")

    # Exclude self-similarity before logsumexp to avoid exponential overflow.
    log_normalizer = similarity.masked_fill(~off_diagonal, -torch.inf).logsumexp(
        dim=1, keepdim=True
    )
    log_probability = similarity - log_normalizer
    positive_term = (log_probability * positive).sum(1) / positive.sum(1)
    negative_term = (similarity * negative).sum(1) / negative.sum(1).clamp_min(1)
    return (-positive_term + negative_weight * negative_term.clamp_min(0)).mean()


def balanced_pair_loss(classifier, features, labels):
    """Use every unordered positive pair and an equal number of negative pairs."""
    indices = torch.triu_indices(
        len(labels), len(labels), offset=1, device=labels.device
    )
    same = labels[indices[0]].eq(labels[indices[1]])
    positives = indices[:, same]
    negatives = indices[:, ~same]
    count = min(positives.size(1), negatives.size(1))
    if not positives.size(1) or not count:
        raise ValueError(
            "Pair supervision requires multiple classes and samples per class."
        )
    selection = torch.randperm(negatives.size(1), device=labels.device)[:count]
    pairs = torch.cat((positives, negatives[:, selection]), dim=1)
    targets = torch.cat(
        (features.new_ones(positives.size(1)), features.new_zeros(count))
    )
    logits = classifier(torch.cat((features[pairs[0]], features[pairs[1]]), dim=1))
    return F.binary_cross_entropy_with_logits(logits, targets)
