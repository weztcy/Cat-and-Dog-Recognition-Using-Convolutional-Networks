"""
Cat and Dog Recognition - Data Audit & Exploration
Performs dataset verification, integrity checks, duplicate detection,
stratified train/validation splitting, and generates exploratory visualizations.
"""

import os
import hashlib
import json
import random
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATASET_DIR = Path("dataset")
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
VIS_DIR = Path("visualizations")
RESULTS_DIR = Path("results")

VIS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_hash(file_path: Path) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def audit_images():
    print("=" * 60)
    print("STARTING DATASET AUDIT AND VERIFICATION")
    print("=" * 60)

    splits = {
        "train": TRAIN_DIR,
        "validation (hold-out test)": VAL_DIR
    }

    records = []
    hash_map = {}
    corrupted_files = []

    for split_name, split_path in splits.items():
        for class_name in ["cats", "dogs"]:
            class_folder = split_path / class_name
            if not class_folder.exists():
                print(f"Warning: {class_folder} does not exist!")
                continue

            files = [f for f in class_folder.iterdir() if f.is_file() and not f.name.endswith(".py")]
            print(f"Scanning {split_name} -> {class_name}: {len(files)} files found.")

            for f in files:
                rel_path = str(f.relative_to(Path(".")))
                # Check integrity
                is_valid = True
                img_format = None
                img_mode = None
                width, height = None, None

                try:
                    with Image.open(f) as img:
                        img.verify()
                    # Re-open to read attributes (verify closes/invalidates img)
                    with Image.open(f) as img:
                        img_format = img.format
                        img_mode = img.mode
                        width, height = img.size
                except Exception as e:
                    is_valid = False
                    corrupted_files.append((rel_path, str(e)))

                file_hash = calculate_hash(f) if is_valid else None
                if file_hash:
                    if file_hash in hash_map:
                        hash_map[file_hash].append(rel_path)
                    else:
                        hash_map[file_hash] = [rel_path]

                records.append({
                    "path": rel_path,
                    "filename": f.name,
                    "split_source": split_name,
                    "category": class_name,
                    "label": 0 if class_name == "cats" else 1,
                    "valid": is_valid,
                    "format": img_format,
                    "mode": img_mode,
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(width / height, 4) if width and height else None,
                    "hash": file_hash
                })

    df = pd.DataFrame(records)
    print("\nAudit Summary:")
    print(f"Total scanned files: {len(df)}")
    print(f"Corrupted files: {len(corrupted_files)}")
    if corrupted_files:
        print(f"Corrupted details: {corrupted_files}")

    # Duplicate check
    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
    print(f"Exact MD5 Duplicate clusters found: {len(duplicates)}")
    if duplicates:
        for h, paths in duplicates.items():
            print(f" - Hash {h}: {paths}")

    # Check color modes and formats
    print("\nFile formats detected:")
    print(df["format"].value_counts().to_dict())
    print("\nColor modes detected:")
    print(df["mode"].value_counts().to_dict())

    # Create stratified 80:20 split on 'train' folder
    train_mask = df["split_source"] == "train"
    train_df = df[train_mask].copy()

    # Stratified sample
    train_cats = train_df[train_df["category"] == "cats"].sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    train_dogs = train_df[train_df["category"] == "dogs"].sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    split_ratio = 0.8
    n_train_cats = int(len(train_cats) * split_ratio)
    n_train_dogs = int(len(train_dogs) * split_ratio)

    internal_train_cats = train_cats.iloc[:n_train_cats]
    internal_val_cats = train_cats.iloc[n_train_cats:]

    internal_train_dogs = train_dogs.iloc[:n_train_dogs]
    internal_val_dogs = train_dogs.iloc[n_train_dogs:]

    internal_train = pd.concat([internal_train_cats, internal_train_dogs]).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    internal_val = pd.concat([internal_val_cats, internal_val_dogs]).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    holdout_test = df[df["split_source"] != "train"].copy().reset_index(drop=True)

    internal_train["final_split"] = "train_internal"
    internal_val["final_split"] = "val_internal"
    holdout_test["final_split"] = "holdout_test"

    combined_df = pd.concat([internal_train, internal_val, holdout_test]).reset_index(drop=True)

    split_counts = pd.crosstab(combined_df["final_split"], combined_df["category"], margins=True)
    print("\nFinal Dataset Split Verification:")
    print(split_counts)

    # Save splits
    split_info = {
        "random_seed": RANDOM_SEED,
        "train_internal_count": len(internal_train),
        "val_internal_count": len(internal_val),
        "holdout_test_count": len(holdout_test),
        "total_images": len(combined_df),
        "splits": {
            row["path"]: {
                "label": int(row["label"]),
                "category": row["category"],
                "split": row["final_split"],
                "width": int(row["width"]),
                "height": int(row["height"])
            }
            for _, row in combined_df.iterrows()
        }
    }

    with open(RESULTS_DIR / "dataset_split.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    # Save audit summary
    audit_summary = {
        "total_scanned": len(df),
        "corrupted_count": len(corrupted_files),
        "duplicate_hash_clusters": len(duplicates),
        "formats": df["format"].value_counts().to_dict(),
        "color_modes": df["mode"].value_counts().to_dict(),
        "dimension_stats": {
            "min_width": int(df["width"].min()),
            "max_width": int(df["width"].max()),
            "mean_width": float(df["width"].mean()),
            "min_height": int(df["height"].min()),
            "max_height": int(df["height"].max()),
            "mean_height": float(df["height"].mean()),
            "mean_aspect_ratio": float(df["aspect_ratio"].mean())
        },
        "split_counts": {
            "train_internal": {
                "cats": int((internal_train["category"] == "cats").sum()),
                "dogs": int((internal_train["category"] == "dogs").sum()),
                "total": len(internal_train)
            },
            "val_internal": {
                "cats": int((internal_val["category"] == "cats").sum()),
                "dogs": int((internal_val["category"] == "dogs").sum()),
                "total": len(internal_val)
            },
            "holdout_test": {
                "cats": int((holdout_test["category"] == "cats").sum()),
                "dogs": int((holdout_test["category"] == "dogs").sum()),
                "total": len(holdout_test)
            }
        }
    }
    with open(RESULTS_DIR / "dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    # Generate Visualizations
    generate_visualizations(combined_df)
    print("=" * 60)
    print("DATA AUDIT COMPLETED SUCCESSFULLY!")
    print("=" * 60)


def generate_visualizations(df: pd.DataFrame):
    print("\nGenerating EDA Visualizations...")

    # 1. Class Distribution
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    split_order = ["train_internal", "val_internal", "holdout_test"]
    split_labels = ["Train Internal (80%)", "Validation Internal (20%)", "Hold-Out Test (Archive Val)"]
    
    cat_counts = [len(df[(df["final_split"] == s) & (df["category"] == "cats")]) for s in split_order]
    dog_counts = [len(df[(df["final_split"] == s) & (df["category"] == "dogs")]) for s in split_order]

    x = np.arange(len(split_order))
    width = 0.35

    rects1 = ax.bar(x - width/2, cat_counts, width, label='Cats (Label 0)', color='#3498db', edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x + width/2, dog_counts, width, label='Dogs (Label 1)', color='#e67e22', edgecolor='black', linewidth=0.8)

    ax.set_title('Distribusi Kelas Berdasarkan Pembagian Subset Dataset', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Jumlah Sampel Citra', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(split_labels, fontsize=10)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax.set_ylim(0, max(cat_counts + dog_counts) * 1.18)

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h}',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "class_distribution.png")
    plt.close()
    print(" - Saved visualizations/class_distribution.png")

    # 2. Sample Image Grid
    fig, axes = plt.subplots(2, 6, figsize=(15, 6), dpi=300)
    cat_samples = df[df["category"] == "cats"].sample(6, random_state=RANDOM_SEED)
    dog_samples = df[df["category"] == "dogs"].sample(6, random_state=RANDOM_SEED)

    for i, (_, row) in enumerate(cat_samples.iterrows()):
        ax = axes[0, i]
        img = Image.open(row["path"])
        ax.imshow(img)
        ax.set_title(f"Cat ({row['width']}x{row['height']})", fontsize=9, color="#2980b9", fontweight="bold")
        ax.axis('off')

    for i, (_, row) in enumerate(dog_samples.iterrows()):
        ax = axes[1, i]
        img = Image.open(row["path"])
        ax.imshow(img)
        ax.set_title(f"Dog ({row['width']}x{row['height']})", fontsize=9, color="#d35400", fontweight="bold")
        ax.axis('off')

    plt.suptitle("Contoh Sampel Citra Kucing (Cats) dan Anjing (Dogs)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "sample_images.png", bbox_inches='tight')
    plt.close()
    print(" - Saved visualizations/sample_images.png")

    # 3. Image Dimensions Distribution
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)

    # Width distribution
    sns.histplot(df["width"], bins=30, kde=True, ax=axes[0], color="#2ecc71", edgecolor="black")
    axes[0].set_title("Distribusi Lebar Citra (Width)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Lebar (Piksel)")
    axes[0].set_ylabel("Frekuensi")
    axes[0].axvline(df["width"].median(), color="red", linestyle="--", label=f"Median: {int(df['width'].median())}px")
    axes[0].legend()

    # Height distribution
    sns.histplot(df["height"], bins=30, kde=True, ax=axes[1], color="#9b59b6", edgecolor="black")
    axes[1].set_title("Distribusi Tinggi Citra (Height)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Tinggi (Piksel)")
    axes[1].set_ylabel("Frekuensi")
    axes[1].axvline(df["height"].median(), color="red", linestyle="--", label=f"Median: {int(df['height'].median())}px")
    axes[1].legend()

    # Aspect ratio scatter vs size
    sns.scatterplot(data=df, x="width", y="height", hue="category", alpha=0.5, palette={"cats": "#3498db", "dogs": "#e67e22"}, ax=axes[2])
    axes[2].set_title("Dimensi Citra (Width vs Height)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Lebar (Piksel)")
    axes[2].set_ylabel("Tinggi (Piksel)")
    axes[2].plot([0, 500], [0, 500], 'r--', alpha=0.7, label='Rasio 1:1')
    axes[2].legend(frameon=True)

    plt.suptitle("Analisis Dimensi dan Variasi Resolusi Dataset", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "image_dimensions.png", bbox_inches='tight')
    plt.close()
    print(" - Saved visualizations/image_dimensions.png")


if __name__ == "__main__":
    audit_images()
