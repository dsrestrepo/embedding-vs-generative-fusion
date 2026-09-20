"""Lightweight, reproducible fusion experiments on frozen embeddings.

This module is intentionally independent of the legacy training helpers so that
all rebuttal baselines use the same split, early-stopping, metrics, and
prediction-export path.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data_utils import split_data


METHODS = ("image_only", "text_only", "linear", "mlp_early", "mlp_late", "gated_attention", "mcr_late", "i2moe")


class EarlyHead(nn.Module):
    def __init__(self, text_dim, image_dim, classes, hidden=128, dropout=0.2):
        super().__init__()
        in_dim = text_dim + image_dim
        self.features = nn.Identity() if hidden == 0 else nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.BatchNorm1d(hidden)
        )
        self.classifier = nn.Linear(in_dim if hidden == 0 else hidden, classes)

    def forward(self, text, image):
        return self.classifier(self.features(torch.cat((text, image), dim=1)))


class UnimodalHead(nn.Module):
    def __init__(self, input_dim, classes, modality):
        super().__init__()
        self.classifier = nn.Linear(input_dim, classes)
        self.modality = modality

    def forward(self, text, image):
        return self.classifier(image if self.modality == "image" else text)


class LateHead(nn.Module):
    def __init__(self, text_dim, image_dim, classes, hidden=128, dropout=0.2):
        super().__init__()
        self.text_fc = nn.Sequential(nn.Linear(text_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.BatchNorm1d(hidden))
        self.image_fc = nn.Sequential(nn.Linear(image_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.BatchNorm1d(hidden))
        self.classifier = nn.Linear(2 * hidden, classes)

    def forward(self, text, image):
        return self.classifier(torch.cat((self.text_fc(text), self.image_fc(image)), dim=1))


class GatedAttentionHead(nn.Module):
    """Feature-level modality gate: softmax attention over two projected inputs."""
    def __init__(self, text_dim, image_dim, classes, hidden=128, dropout=0.2):
        super().__init__()
        self.text_fc = nn.Sequential(nn.Linear(text_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.BatchNorm1d(hidden))
        self.image_fc = nn.Sequential(nn.Linear(image_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.BatchNorm1d(hidden))
        self.gate = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        self.classifier = nn.Linear(hidden, classes)

    def forward(self, text, image, return_gates=False):
        text_h, image_h = self.text_fc(text), self.image_fc(image)
        gates = torch.softmax(self.gate(torch.cat((text_h, image_h), dim=1)), dim=1)
        logits = self.classifier(gates[:, :1] * text_h + gates[:, 1:] * image_h)
        return (logits, gates) if return_gates else logits


class MCRLateHead(LateHead):
    """Late fusion with auxiliary unimodal heads used by the fixed MCR loss."""
    def __init__(self, text_dim, image_dim, classes, hidden=128, dropout=0.2):
        super().__init__(text_dim, image_dim, classes, hidden, dropout)
        self.text_classifier = nn.Linear(hidden, classes)
        self.image_classifier = nn.Linear(hidden, classes)

    def forward(self, text, image, return_aux=False):
        text_h, image_h = self.text_fc(text), self.image_fc(image)
        fused = self.classifier(torch.cat((text_h, image_h), dim=1))
        if return_aux:
            return fused, self.text_classifier(text_h), self.image_classifier(image_h)
        return fused


class I2MoEHead(nn.Module):
    """Two-modality I²MoE (ICML 2025) using late-fusion interaction experts.

    The four experts model text uniqueness, image uniqueness, synergy, and
    redundancy.  A reweighting MLP produces a sample-specific convex mixture.
    """
    def __init__(self, text_dim, image_dim, classes, hidden=128, dropout=0.2):
        super().__init__()
        self.experts = nn.ModuleList([LateHead(text_dim, image_dim, classes, hidden, dropout) for _ in range(4)])
        self.reweight = nn.Sequential(nn.Linear(text_dim + image_dim, hidden), nn.ReLU(), nn.Linear(hidden, 4))

    def forward(self, text, image, return_aux=False):
        weights = torch.softmax(self.reweight(torch.cat((text, image), dim=1)), dim=1)
        full = torch.stack([expert(text, image) for expert in self.experts], dim=1)
        logits = (full * weights.unsqueeze(2)).sum(dim=1)
        if not return_aux:
            return logits
        no_text, no_image = torch.zeros_like(text), torch.zeros_like(image)
        absent_text = torch.stack([expert(no_text, image) for expert in self.experts], dim=1)
        absent_image = torch.stack([expert(text, no_image) for expert in self.experts], dim=1)
        return logits, full, absent_text, absent_image, weights


def build_model(method, text_dim, image_dim, classes):
    if method == "image_only":
        return UnimodalHead(image_dim, classes, "image")
    if method == "text_only":
        return UnimodalHead(text_dim, classes, "text")
    if method == "linear":
        return EarlyHead(text_dim, image_dim, classes, hidden=0)
    if method == "mlp_early":
        return EarlyHead(text_dim, image_dim, classes)
    if method == "mlp_late":
        return LateHead(text_dim, image_dim, classes)
    if method == "gated_attention":
        return GatedAttentionHead(text_dim, image_dim, classes)
    if method == "mcr_late":
        return MCRLateHead(text_dim, image_dim, classes)
    if method == "i2moe":
        return I2MoEHead(text_dim, image_dim, classes)
    raise ValueError(f"Unsupported method: {method}")


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _mcr_loss(fused, text, image, labels, ce):
    """Fixed MCR-style regularizer (lambda=0.01, contrastive coefficient=1).

    It keeps each unimodal head predictive and penalizes the fused predictor
    when it underperforms either unimodal prediction on the same example.
    """
    fused_loss = ce(fused, labels)
    text_loss, image_loss = ce(text, labels), ce(image, labels)
    per_fused = nn.functional.cross_entropy(fused, labels, reduction="none")
    per_best_uni = torch.minimum(
        nn.functional.cross_entropy(text, labels, reduction="none"),
        nn.functional.cross_entropy(image, labels, reduction="none"),
    )
    contrastive = torch.relu(per_fused - per_best_uni).mean()
    return fused_loss + 0.01 * (text_loss + image_loss) + contrastive


def _i2moe_loss(logits, full, absent_text, absent_image, labels, ce):
    """Official I²MoE interaction objectives with a fixed 0.01 weight."""
    # expert 0: text uniqueness; expert 1: image uniqueness
    triplet = nn.TripletMarginLoss(margin=1.0, p=2)
    text_unique = triplet(full[:, 0], absent_image[:, 0], absent_text[:, 0])
    image_unique = triplet(full[:, 1], absent_text[:, 1], absent_image[:, 1])
    # expert 2: synergy should change when either input is ablated.
    synergy = (nn.functional.cosine_similarity(full[:, 2], absent_text[:, 2]).mean() +
               nn.functional.cosine_similarity(full[:, 2], absent_image[:, 2]).mean()) / 2
    # expert 3: redundancy should remain stable when one input is ablated.
    redundancy = ((1 - nn.functional.cosine_similarity(full[:, 3], absent_text[:, 3]).mean()) +
                  (1 - nn.functional.cosine_similarity(full[:, 3], absent_image[:, 3]).mean())) / 2
    return ce(logits, labels) + 0.01 * (text_unique + image_unique + synergy + redundancy)


def _metrics(y_true, probs):
    pred = probs.argmax(axis=1)
    result = {"accuracy": accuracy_score(y_true, pred), "f1_score": f1_score(y_true, pred, average="macro")}
    try:
        result["auc"] = (roc_auc_score(y_true, probs[:, 1]) if probs.shape[1] == 2
                         else roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        result["auc"] = float("nan")
    return result


def train_and_predict(df, text_cols, image_cols, label_col, method, seed=42, epochs=100,
                      batch_size=4096, lr=3e-4, patience=5):
    """Train one fixed method and return aggregate metrics plus test predictions."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    _set_seed(seed)
    splits = split_data(df, val_size=0.1, random_state=seed)
    train_df, val_df, test_df = splits if len(splits) == 3 else (*splits, None)
    if val_df is None:
        raise ValueError("A validation split is required for the rebuttal protocol.")
    encoder = LabelEncoder().fit(train_df[label_col].astype(str))
    for name, frame in (("validation", val_df), ("test", test_df)):
        unseen = set(frame[label_col].astype(str)) - set(encoder.classes_)
        if unseen:
            raise ValueError(f"{name} has labels unseen in train: {sorted(unseen)}")

    def tensors(frame):
        return (torch.tensor(frame[text_cols].to_numpy(np.float32)),
                torch.tensor(frame[image_cols].to_numpy(np.float32)),
                torch.tensor(encoder.transform(frame[label_col].astype(str)), dtype=torch.long))
    train_t, val_t, test_t = tensors(train_df), tensors(val_df), tensors(test_df)
    train_loader = DataLoader(TensorDataset(*train_t), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(TensorDataset(*val_t), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(TensorDataset(*test_t), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(method, len(text_cols), len(image_cols), len(encoder.classes_)).to(device)
    counts = np.bincount(train_t[2].numpy(), minlength=len(encoder.classes_))
    weight = torch.tensor(len(train_t[2]) / (len(counts) * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_state, best_val, stale, best_epoch = None, -np.inf, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for text, image, labels in train_loader:
            text, image, labels = text.to(device, non_blocking=True), image.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if method == "mcr_late":
                fused, text_logits, image_logits = model(text, image, return_aux=True)
                loss = _mcr_loss(fused, text_logits, image_logits, labels, criterion)
            elif method == "i2moe":
                logits, full, absent_text, absent_image, _ = model(text, image, return_aux=True)
                loss = _i2moe_loss(logits, full, absent_text, absent_image, labels, criterion)
            else:
                loss = criterion(model(text, image), labels)
            loss.backward(); optimizer.step()
        model.eval(); val_probs, val_true, val_loss_total = [], [], 0.0
        with torch.no_grad():
            for text, image, labels in val_loader:
                text, image, labels_device = text.to(device), image.to(device), labels.to(device)
                logits = model(text, image)
                val_probs.append(torch.softmax(logits, 1).cpu().numpy()); val_true.append(labels.numpy())
                val_loss_total += criterion(logits, labels_device).item() * labels.size(0)
        # Match the original paper implementation: early fusion checkpoints use
        # validation loss. Other new fusion baselines retain Macro-F1 selection.
        validation_score = -val_loss_total / len(val_loader.dataset) if method == "mlp_early" else _metrics(np.concatenate(val_true), np.concatenate(val_probs))["f1_score"]
        if validation_score > best_val:
            best_state, best_val, stale, best_epoch = copy.deepcopy(model.state_dict()), validation_score, 0, epoch
        else:
            stale += 1
            if stale >= patience: break
    model.load_state_dict(best_state)
    model.eval(); probs, true, gates = [], [], []
    with torch.no_grad():
        for text, image, labels in test_loader:
            text, image = text.to(device), image.to(device)
            if method == "gated_attention":
                logits, gate = model(text, image, return_gates=True); gates.append(gate.cpu().numpy())
            else:
                logits = model(text, image)
            probs.append(torch.softmax(logits, 1).cpu().numpy()); true.append(labels.numpy())
    probs, true = np.concatenate(probs), np.concatenate(true)
    metrics = _metrics(true, probs)
    metrics.update({"epochs_run": best_epoch, "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df), "n_classes": len(encoder.classes_)})
    return metrics, test_df, true, probs, (np.concatenate(gates) if gates else None), model.cpu()
