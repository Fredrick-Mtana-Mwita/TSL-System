"""
Training pipeline for the TSL Transformer.

Usage:
    cd backend
    python train.py

Generates synthetic data automatically if none exists, then trains for up to
50 epochs with early stopping. Saves the best checkpoint to:
    checkpoints/tsl_transformer.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models.transformer_model import TSLTransformer, NUM_CLASSES
from services.model_service import CHECKPOINT_PATH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils.dataset import DATA_DIR, TSLDataset, generate_synthetic_dataset

EPOCHS     = 50
BATCH_SIZE = 32
LR         = 1e-3
PATIENCE   = 10


def train() -> None:
    if not (DATA_DIR / "train.npy").exists():
        print("No dataset found — generating synthetic data…")
        generate_synthetic_dataset(n_per_class=300)

    train_ds = TSLDataset("train")
    val_ds   = TSLDataset("val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training on {DEVICE}  |  {len(train_ds)} train  /  {len(val_ds)} val samples")

    model     = TSLTransformer(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    no_improve   = 0

    for epoch in range(1, EPOCHS + 1):
        # --- train ---
        model.train()
        t_loss = t_correct = t_total = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item() * len(y)
            t_correct += (logits.argmax(1) == y).sum().item()
            t_total   += len(y)

        # --- validate ---
        model.eval()
        v_correct = v_total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                v_correct += (model(X).argmax(1) == y).sum().item()
                v_total   += len(y)

        val_acc = v_correct / v_total
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{EPOCHS}  "
            f"loss {t_loss/t_total:.4f}  "
            f"train {t_correct/t_total:.3f}  "
            f"val {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            no_improve = 0
            print(f"  ✓ checkpoint saved (val acc {best_val_acc:.3f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    print(f"\nTraining complete — best val acc: {best_val_acc:.3f}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()
