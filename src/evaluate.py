"""
Model Evaluation and Analysis Script.
Evaluates models on internal validation and hold-out test set.
Computes Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix,
and generates all evaluation visualizations and Grad-CAM examples.
"""

import os
import sys
import json
from pathlib import Path
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)

import torch
import torch.nn as nn
from torchvision import transforms

from src.dataset import CatDogDataset, get_transforms
from src.models import CustomBaselineCNN, build_mobilenet_v2
from src.predict import GradCAM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
VIS_DIR = Path("visualizations")

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'


def get_model_predictions(model, dataset, device):
    model.eval()
    y_true = []
    y_probs = []
    paths = []

    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    with torch.no_grad():
        for inputs, labels, batch_paths in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
            y_probs.extend(probs.tolist())
            y_true.extend(labels.numpy().tolist())
            paths.extend(batch_paths)

    y_true = np.array(y_true, dtype=int)
    y_probs = np.array(y_probs, dtype=float)
    y_preds = (y_probs >= 0.5).astype(int)

    return y_true, y_probs, y_preds, paths


def compute_metrics(y_true, y_probs, y_preds):
    acc = accuracy_score(y_true, y_preds)
    prec = precision_score(y_true, y_preds, zero_division=0)
    rec = recall_score(y_true, y_preds, zero_division=0)
    f1 = f1_score(y_true, y_preds, zero_division=0)
    auc = roc_auc_score(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    cm = confusion_matrix(y_true, y_preds)

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "pr_auc": round(float(ap), 4),
        "confusion_matrix": cm.tolist()
    }


def load_model_from_checkpoint(model_type: str, checkpoint_path: Path):
    if model_type == "baseline":
        model = CustomBaselineCNN()
    else:
        model = build_mobilenet_v2(pretrained=False, freeze_features=False)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(DEVICE)
    model.eval()
    return model


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    labels = ["Cat (0)", "Dog (1)"]
    annot = [
        [f"{cm[0][0]}\n(TN: {cm[0][0]/(cm[0][0]+cm[0][1])*100:.1f}%)", f"{cm[0][1]}\n(FP: {cm[0][1]/(cm[0][0]+cm[0][1])*100:.1f}%)"],
        [f"{cm[1][0]}\n(FN: {cm[1][0]/(cm[1][0]+cm[1][1])*100:.1f}%)", f"{cm[1][1]}\n(TP: {cm[1][1]/(cm[1][0]+cm[1][1])*100:.1f}%)"]
    ]
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=True,
                xticklabels=labels, yticklabels=labels, ax=ax,
                annot_kws={"size": 12, "weight": "bold"}, linewidths=1.5, linecolor="white")

    ax.set_title("Confusion Matrix — Hold-Out Test Set (1.000 Gambar)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Prediksi Model", fontsize=11, fontweight="bold")
    ax.set_ylabel("Label Sebenarnya (Ground Truth)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f" - Saved {save_path}")


def plot_roc_curve(y_true, y_probs, roc_auc, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.plot(fpr, tpr, color="#2980b9", lw=2.5, label=f'MobileNetV2 Fine-Tuned (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='#7f8c8d', lw=1.5, linestyle='--', label='Random Guess (AUC = 0.5000)')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight="bold")
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11, fontweight="bold")
    ax.set_title('Receiver Operating Characteristic (ROC) Curve — Hold-Out Test', fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f" - Saved {save_path}")


def plot_precision_recall_curve(y_true, y_probs, pr_auc, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.plot(recall, precision, color="#27ae60", lw=2.5, label=f'MobileNetV2 Fine-Tuned (PR-AUC = {pr_auc:.4f})')
    baseline = np.sum(y_true) / len(y_true)
    ax.plot([0, 1], [baseline, baseline], color='#e67e22', lw=1.5, linestyle='--', label=f'Baseline Seimbang (AP = {baseline:.2f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (True Positive Rate)', fontsize=11, fontweight="bold")
    ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=11, fontweight="bold")
    ax.set_title('Precision-Recall Curve — Hold-Out Test', fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc="lower left", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f" - Saved {save_path}")


def plot_model_comparison(comparison_df: pd.DataFrame, save_path):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    
    x = np.arange(len(metrics))
    width = 0.25

    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    rects_list = []

    for i, (_, row) in enumerate(comparison_df.iterrows()):
        offset = (i - 1) * width
        vals = [row[m] * 100 for m in metrics]
        rects = ax.bar(x + offset, vals, width, label=row["model_label"], color=colors[i], edgecolor="black", linewidth=0.8)
        rects_list.append(rects)

    ax.set_ylabel('Persentase Skor (%)', fontsize=11, fontweight="bold")
    ax.set_title('Perbandingan Metrik Model pada Internal Validation Set', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10, fontweight="bold")
    ax.set_ylim(50, 105)
    ax.legend(loc='lower right', frameon=True)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    for rects in rects_list:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f" - Saved {save_path}")


def plot_prediction_examples(df_preds: pd.DataFrame, save_path):
    # Select 2 TP, 2 TN, 2 FP, 2 FN
    tp_samples = df_preds[(df_preds["actual_label"] == 1) & (df_preds["predicted_label"] == 1)].sort_values(by="confidence", ascending=False).head(2)
    tn_samples = df_preds[(df_preds["actual_label"] == 0) & (df_preds["predicted_label"] == 0)].sort_values(by="confidence", ascending=False).head(2)
    fp_samples = df_preds[(df_preds["actual_label"] == 0) & (df_preds["predicted_label"] == 1)].head(2)
    fn_samples = df_preds[(df_preds["actual_label"] == 1) & (df_preds["predicted_label"] == 0)].head(2)

    selected = pd.concat([tn_samples, tp_samples, fp_samples, fn_samples]).reset_index(drop=True)
    titles = [
        "True Negative (Cat -> Cat)",
        "True Negative (Cat -> Cat)",
        "True Positive (Dog -> Dog)",
        "True Positive (Dog -> Dog)",
        "False Positive (Cat -> Dog)",
        "False Positive (Cat -> Dog)",
        "False Negative (Dog -> Cat)",
        "False Negative (Dog -> Cat)"
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=300)
    axes = axes.flatten()

    for idx, row in selected.iterrows():
        ax = axes[idx]
        img = Image.open(row["path"])
        ax.imshow(img)
        is_correct = row["correct"]
        color = "#27ae60" if is_correct else "#c0392b"

        status_text = "BENAR" if is_correct else "SALAH"
        ax.set_title(f"[{status_text}] {titles[idx]}\nPred: {row['predicted_class']} ({row['confidence']:.1f}%)\nAktual: {row['actual_class']}",
                     fontsize=9, color=color, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Analisis Contoh Prediksi Benar dan Salah pada Hold-Out Test Set", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f" - Saved {save_path}")


def generate_gradcam_examples(model, target_layer, df_preds: pd.DataFrame, save_path):
    print("\nGenerating Grad-CAM visualizations...")
    gradcam = GradCAM(model, target_layer)
    _, eval_tf = get_transforms()

    # Select representative samples: 2 cats and 2 dogs (1 correct, 1 misclassified/low confidence each)
    cat_correct = df_preds[(df_preds["actual_label"] == 0) & (df_preds["correct"] == True)].iloc[0]
    dog_correct = df_preds[(df_preds["actual_label"] == 1) & (df_preds["correct"] == True)].iloc[0]
    
    cat_wrong_df = df_preds[(df_preds["actual_label"] == 0) & (~df_preds["correct"])]
    cat_hard = cat_wrong_df.iloc[0] if len(cat_wrong_df) > 0 else df_preds[(df_preds["actual_label"] == 0)].iloc[1]

    dog_wrong_df = df_preds[(df_preds["actual_label"] == 1) & (~df_preds["correct"])]
    dog_hard = dog_wrong_df.iloc[0] if len(dog_wrong_df) > 0 else df_preds[(df_preds["actual_label"] == 1)].iloc[1]

    samples = [cat_correct, dog_correct, cat_hard, dog_hard]
    subtitles = ["Cat (Correct)", "Dog (Correct)", f"Cat ({'Misclassified' if not cat_hard['correct'] else 'Low Conf'})", f"Dog ({'Misclassified' if not dog_hard['correct'] else 'Low Conf'})"]

    fig, axes = plt.subplots(4, 3, figsize=(12, 14), dpi=300)

    for i, (sample, sub) in enumerate(zip(samples, subtitles)):
        orig_img = Image.open(sample["path"]).convert("RGB")
        tensor_img = eval_tf(orig_img).unsqueeze(0).to(DEVICE)
        tensor_img.requires_grad = True

        pred_lbl = int(sample["predicted_label"])
        cam_mask = gradcam.generate(tensor_img, target_class=pred_lbl)

        # Resize CAM to original image size
        cam_img = Image.fromarray((cam_mask * 255).astype(np.uint8)).resize(orig_img.size, resample=Image.Resampling.BILINEAR)
        cam_array = np.array(cam_img) / 255.0

        # Create heatmap overlay
        heatmap = plt.cm.jet(cam_array)[:, :, :3]
        overlay = (0.55 * np.array(orig_img) / 255.0) + (0.45 * heatmap)
        overlay = np.clip(overlay, 0, 1)

        # Col 1: Original
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f"Original: {sub}\nAktual: {sample['actual_class']}", fontsize=10, fontweight="bold")
        axes[i, 0].axis("off")

        # Col 2: Grad-CAM Heatmap
        axes[i, 1].imshow(cam_mask, cmap="jet")
        axes[i, 1].set_title(f"Grad-CAM Heatmap\nTarget: {sample['predicted_class']}", fontsize=10, fontweight="bold")
        axes[i, 1].axis("off")

        # Col 3: Overlay
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"Overlay Citra + Atensi\nPred: {sample['predicted_class']} ({sample['confidence']:.1f}%)", fontsize=10, fontweight="bold")
        axes[i, 2].axis("off")

    plt.suptitle("Visualisasi Interpretabilitas Grad-CAM (MobileNetV2 Layer Terakhir)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f" - Saved {save_path}")


def main():
    print("=" * 60)
    print("STARTING COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)

    _, eval_tf = get_transforms()
    val_ds = CatDogDataset(split_name="val_internal", transform=eval_tf)
    test_ds = CatDogDataset(split_name="holdout_test", transform=eval_tf)

    # 1. Evaluate baseline, mobilenet frozen, mobilenet finetuned on VAL_INTERNAL
    models_to_eval = [
        ("baseline", "Custom CNN Baseline", MODELS_DIR / "baseline_custom_cnn.pt"),
        ("mobilenet_frozen", "MobileNetV2 Feature Extractor", MODELS_DIR / "mobilenetv2_feature_extractor.pt"),
        ("mobilenet_finetuned", "MobileNetV2 Fine-Tuned", MODELS_DIR / "mobilenetv2_finetuned.pt")
    ]

    val_metrics_list = []

    for key, label, path in models_to_eval:
        if not path.exists():
            print(f"Checkpoint {path} not found! Skipping {key}.")
            continue
        print(f"\nEvaluating {label} on Internal Validation Set...")
        model = load_model_from_checkpoint(key, path)
        y_true, y_probs, y_preds, _ = get_model_predictions(model, val_ds, DEVICE)
        metrics = compute_metrics(y_true, y_probs, y_preds)
        metrics["model_key"] = key
        metrics["model_label"] = label
        val_metrics_list.append(metrics)
        print(f" -> Accuracy: {metrics['accuracy']*100:.2f}%, F1: {metrics['f1_score']*100:.2f}%, ROC-AUC: {metrics['roc_auc']:.4f}")

    df_comparison = pd.DataFrame(val_metrics_list)
    print("\nInternal Validation Comparison:")
    print(df_comparison[["model_label", "accuracy", "precision", "recall", "f1_score", "roc_auc"]])

    plot_model_comparison(df_comparison, VIS_DIR / "model_comparison.png")

    # Select best model based on F1 score on validation internal
    best_row = df_comparison.sort_values(by="f1_score", ascending=False).iloc[0]
    best_model_key = best_row["model_key"]
    best_model_label = best_row["model_label"]
    print(f"\n>>> Selected Best Model based on Internal Validation: {best_model_label} (F1: {best_row['f1_score']:.4f})")

    # 2. Evaluate Best Model ON HOLDOUT TEST SET
    print(f"\nEvaluating Best Model ({best_model_label}) on Final Hold-Out Test Set (1.000 gambar)...")
    best_model = load_model_from_checkpoint(best_model_key, MODELS_DIR / f"{best_model_key if best_model_key == 'baseline' else 'mobilenetv2_finetuned'}.pt")
    test_y_true, test_y_probs, test_y_preds, test_paths = get_model_predictions(best_model, test_ds, DEVICE)
    test_metrics = compute_metrics(test_y_true, test_y_probs, test_y_preds)

    print("\nHold-Out Test Results:")
    print(f"Accuracy:  {test_metrics['accuracy']*100:.2f}%")
    print(f"Precision: {test_metrics['precision']*100:.2f}%")
    print(f"Recall:    {test_metrics['recall']*100:.2f}%")
    print(f"F1-Score:  {test_metrics['f1_score']*100:.2f}%")
    print(f"ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {test_metrics['pr_auc']:.4f}")

    # Build predictions dataframe
    df_preds = pd.DataFrame({
        "path": test_paths,
        "actual_label": test_y_true,
        "actual_class": ["dogs" if y == 1 else "cats" for y in test_y_true],
        "predicted_label": test_y_preds,
        "predicted_class": ["dogs" if y == 1 else "cats" for y in test_y_preds],
        "probability_dog": np.round(test_y_probs, 4),
        "confidence": np.round(np.where(test_y_preds == 1, test_y_probs, 1.0 - test_y_probs) * 100, 2),
        "correct": test_y_true == test_y_preds
    })
    df_preds.to_csv(RESULTS_DIR / "predictions.csv", index=False)
    print(f" - Saved {RESULTS_DIR / 'predictions.csv'}")

    # Classification report
    cr_text = classification_report(test_y_true, test_y_preds, target_names=["cats (0)", "dogs (1)"], digits=4)
    print("\nClassification Report:\n", cr_text)
    with open(RESULTS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(cr_text)

    # Save comprehensive metrics JSON
    final_results = {
        "best_model_selected": best_model_label,
        "selection_criterion": "highest internal validation f1-score / accuracy",
        "internal_validation_comparison": val_metrics_list,
        "final_holdout_test_metrics": test_metrics
    }
    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    # Generate test visualizations
    cm = np.array(test_metrics["confusion_matrix"])
    plot_confusion_matrix(cm, VIS_DIR / "confusion_matrix.png")
    plot_roc_curve(test_y_true, test_y_probs, test_metrics["roc_auc"], VIS_DIR / "roc_curve.png")
    plot_precision_recall_curve(test_y_true, test_y_probs, test_metrics["pr_auc"], VIS_DIR / "precision_recall_curve.png")
    plot_prediction_examples(df_preds, VIS_DIR / "prediction_examples.png")

    # Grad-CAM
    target_layer = best_model.features[-1] if hasattr(best_model, "features") else best_model.features[-1]
    generate_gradcam_examples(best_model, target_layer, df_preds, VIS_DIR / "gradcam_examples.png")

    print("=" * 60)
    print("ALL EVALUATIONS AND VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
