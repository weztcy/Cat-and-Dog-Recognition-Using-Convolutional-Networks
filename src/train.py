"""
Model Training Script for Cat and Dog Classification.
Trains Custom Baseline CNN, MobileNetV2 Feature Extractor, and MobileNetV2 Fine-Tuned.
Tracks loss, accuracy, saves best checkpoints, and plots training histories.
"""

import os
import sys
import time
import json
import random
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

from src.dataset import get_dataloaders
from src.models import CustomBaselineCNN, build_mobilenet_v2, unfreeze_mobilenet_top_layers

# Reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
VIS_DIR = Path("visualizations")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels, _ in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def run_training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler,
                      num_epochs, model_name, save_path, patience=5, device=torch.device("cpu")):
    print(f"\n--- Training {model_name} for up to {num_epochs} epochs ---")
    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_no_improve = 0
    start_time = time.time()

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["lr"].append(current_lr)

        duration = time.time() - epoch_start
        print(f"Epoch {epoch:02d}/{num_epochs:02d} [{duration:.1f}s] - "
              f"Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}% (lr: {current_lr:.1e})")

        # Save best based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "model_name": model_name
            }, save_path)
            print(f"  --> Saved new best checkpoint to {save_path} (Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs).")
                break

    total_time = time.time() - start_time
    print(f"Finished {model_name} in {total_time:.1f}s. Best Val Acc: {best_val_acc*100:.2f}%, Best Val Loss: {best_val_loss:.4f}")
    return history, total_time, best_val_acc, best_val_loss


def plot_histories(histories: dict):
    print("\nGenerating Training History Visualizations...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # 1. Training & Validation Loss
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300, sharey=False)
    models_keys = [
        ("baseline", "Custom CNN Baseline"),
        ("mobilenet_frozen", "MobileNetV2 (Feature Extractor)"),
        ("mobilenet_finetuned", "MobileNetV2 (Fine-Tuned)")
    ]

    for i, (key, title) in enumerate(models_keys):
        hist = histories[key]["history"]
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[i].plot(epochs, hist["train_loss"], 'o-', color='#e74c3c', label='Train Loss', linewidth=2)
        axes[i].plot(epochs, hist["val_loss"], 's--', color='#2980b9', label='Val Loss', linewidth=2)
        axes[i].set_title(title, fontsize=12, fontweight='bold')
        axes[i].set_xlabel("Epoch", fontsize=10)
        axes[i].set_ylabel("Binary Cross Entropy Loss", fontsize=10)
        axes[i].legend(frameon=True)
        axes[i].grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Kurva Perbandingan Training & Validation Loss Selama Pelatihan", fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "training_loss.png", bbox_inches='tight')
    plt.close()
    print(" - Saved visualizations/training_loss.png")

    # 2. Training & Validation Accuracy
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300, sharey=True)
    for i, (key, title) in enumerate(models_keys):
        hist = histories[key]["history"]
        epochs = range(1, len(hist["train_acc"]) + 1)
        axes[i].plot(epochs, [a * 100 for a in hist["train_acc"]], 'o-', color='#27ae60', label='Train Accuracy', linewidth=2)
        axes[i].plot(epochs, [a * 100 for a in hist["val_acc"]], 's--', color='#8e44ad', label='Val Accuracy', linewidth=2)
        axes[i].set_title(title, fontsize=12, fontweight='bold')
        axes[i].set_xlabel("Epoch", fontsize=10)
        if i == 0:
            axes[i].set_ylabel("Akurasi (%)", fontsize=10)
        axes[i].legend(loc="lower right", frameon=True)
        axes[i].grid(True, linestyle="--", alpha=0.6)
        axes[i].set_ylim(45, 102)

    plt.suptitle("Kurva Perbandingan Training & Validation Accuracy Selama Pelatihan", fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "training_accuracy.png", bbox_inches='tight')
    plt.close()
    print(" - Saved visualizations/training_accuracy.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"TRAINING PIPELINE INITIALIZED (Device: {device})")
    print("=" * 60)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32, num_workers=0)
    criterion = nn.BCEWithLogitsLoss()

    all_histories = {}

    # -------------------------------------------------------------
    # 1. Custom Baseline CNN
    # -------------------------------------------------------------
    baseline_model = CustomBaselineCNN(dropout_rate=0.4).to(device)
    baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=1e-3, weight_decay=1e-4)
    baseline_scheduler = ReduceLROnPlateau(baseline_optimizer, mode='min', factor=0.5, patience=3)

    baseline_save_path = MODELS_DIR / "baseline_custom_cnn.pt"
    base_hist, base_time, base_acc, base_loss = run_training_loop(
        model=baseline_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=baseline_optimizer,
        scheduler=baseline_scheduler,
        num_epochs=15,
        model_name="CustomBaselineCNN",
        save_path=baseline_save_path,
        patience=5,
        device=device
    )
    all_histories["baseline"] = {
        "model_name": "CustomBaselineCNN",
        "best_val_acc": round(base_acc, 4),
        "best_val_loss": round(base_loss, 4),
        "duration_seconds": round(base_time, 2),
        "history": base_hist
    }

    # -------------------------------------------------------------
    # 2. MobileNetV2 - Feature Extractor (Frozen Backbone)
    # -------------------------------------------------------------
    mobilenet_feat = build_mobilenet_v2(pretrained=True, freeze_features=True, dropout_rate=0.3).to(device)
    # Only classifier params require grad
    feat_optimizer = optim.Adam(filter(lambda p: p.requires_grad, mobilenet_feat.parameters()), lr=1e-3, weight_decay=1e-4)
    feat_scheduler = ReduceLROnPlateau(feat_optimizer, mode='min', factor=0.5, patience=2)

    feat_save_path = MODELS_DIR / "mobilenetv2_feature_extractor.pt"
    feat_hist, feat_time, feat_acc, feat_loss = run_training_loop(
        model=mobilenet_feat,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=feat_optimizer,
        scheduler=feat_scheduler,
        num_epochs=8,
        model_name="MobileNetV2_FeatureExtractor",
        save_path=feat_save_path,
        patience=4,
        device=device
    )
    all_histories["mobilenet_frozen"] = {
        "model_name": "MobileNetV2_FeatureExtractor",
        "best_val_acc": round(feat_acc, 4),
        "best_val_loss": round(feat_loss, 4),
        "duration_seconds": round(feat_time, 2),
        "history": feat_hist
    }

    # -------------------------------------------------------------
    # 3. MobileNetV2 - Fine-Tuning (Unfreeze Top Layers)
    # -------------------------------------------------------------
    # Load best weights from feature extractor phase
    checkpoint = torch.load(feat_save_path, map_location=device, weights_only=False)
    mobilenet_feat.load_state_dict(checkpoint["model_state_dict"])
    
    # Unfreeze top layers (from layer 14 onwards)
    unfreeze_mobilenet_top_layers(mobilenet_feat, start_layer_index=14)

    # Use smaller learning rate for fine tuning
    ft_optimizer = optim.Adam([
        {"params": [p for i, blk in enumerate(mobilenet_feat.features) if i >= 14 for p in blk.parameters()], "lr": 1e-4},
        {"params": mobilenet_feat.classifier.parameters(), "lr": 5e-4}
    ], weight_decay=1e-4)
    ft_scheduler = ReduceLROnPlateau(ft_optimizer, mode='min', factor=0.5, patience=2)

    ft_save_path = MODELS_DIR / "mobilenetv2_finetuned.pt"
    ft_hist, ft_time, ft_acc, ft_loss = run_training_loop(
        model=mobilenet_feat,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=ft_optimizer,
        scheduler=ft_scheduler,
        num_epochs=8,
        model_name="MobileNetV2_FineTuned",
        save_path=ft_save_path,
        patience=4,
        device=device
    )
    all_histories["mobilenet_finetuned"] = {
        "model_name": "MobileNetV2_FineTuned",
        "best_val_acc": round(ft_acc, 4),
        "best_val_loss": round(ft_loss, 4),
        "duration_seconds": round(ft_time, 2),
        "history": ft_hist
    }

    # Copy / Save best model overall as best_model.pt
    best_overall_key = max(all_histories, key=lambda k: all_histories[k]["best_val_acc"])
    print(f"\nBest Model on Internal Validation: {best_overall_key} with Val Acc: {all_histories[best_overall_key]['best_val_acc']*100:.2f}%")

    best_source_path = {
        "baseline": baseline_save_path,
        "mobilenet_frozen": feat_save_path,
        "mobilenet_finetuned": ft_save_path
    }[best_overall_key]

    best_ckpt = torch.load(best_source_path, map_location=device, weights_only=False)
    torch.save(best_ckpt, MODELS_DIR / "best_model.pt")
    print(f"Saved best model checkpoint to {MODELS_DIR / 'best_model.pt'}")

    # Save model config metadata
    model_config = {
        "best_model_architecture": all_histories[best_overall_key]["model_name"],
        "best_model_stage": best_overall_key,
        "best_val_accuracy": all_histories[best_overall_key]["best_val_acc"],
        "best_val_loss": all_histories[best_overall_key]["best_val_loss"],
        "input_resolution": [224, 224],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "classes": {
            "0": "cats",
            "1": "dogs"
        },
        "framework": f"PyTorch {torch.__version__}",
        "device": str(device),
        "seed": RANDOM_SEED
    }
    with open(MODELS_DIR / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    # Save training histories
    with open(RESULTS_DIR / "training_histories.json", "w", encoding="utf-8") as f:
        json.dump(all_histories, f, indent=2)

    # Plot graphs
    plot_histories(all_histories)
    print("=" * 60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
