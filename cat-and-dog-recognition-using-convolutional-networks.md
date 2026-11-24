# Portfolio Project Data — Cat and Dog Recognition Using Convolutional Networks

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/cat_and_dog_recognition/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image10.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "cat-and-dog-recognition-using-convolutional-networks",
  "code": "PRO-36 // ML-21",
  "title": "Cat and Dog Recognition Using Convolutional Networks",
  "category": "machine-learning",
  "categoryLabel": "Deep Learning",
  "year": "2024",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/Cat-and-Dog-Recognition-Using-Convolutional-Networks",
  "license": "MIT License",
  "banner": "/images/projects/cat_and_dog_recognition/banner.webp",
  "summary": "End-to-end binary cat vs dog image classifier comparing Custom CNN and fine-tuned MobileNetV2 with Grad-CAM and Streamlit inference.",
  "projectDescription": "Binary computer vision pipeline classifying cat and dog JPEG images using PyTorch. Evaluates a 4-block Custom CNN against two-stage MobileNetV2 transfer learning under a strict hold-out split, achieving 97.70% test accuracy with Grad-CAM explainability and a Streamlit app.",
  "architectureSummary": "MD5 Image Audit & Leakage-Free Split -> 224x224 Augmentation -> Custom CNN vs Two-Stage MobileNetV2 -> Grad-CAM & Streamlit Inference",
  "techStack": [
    "Python",
    "PyTorch",
    "Torchvision",
    "NumPy",
    "Pandas",
    "Scikit-Learn",
    "Pillow",
    "Matplotlib",
    "Seaborn",
    "Streamlit"
  ],
  "telemetryMatrix": {
    "projectType": "Image Classification Pipeline",
    "targetPlatform": "Python CLI & Streamlit Web App",
    "primaryDomain": "Computer Vision & Deep Learning",
    "architecturePipeline": "Custom CNN vs Fine-Tuned MobileNetV2",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2024"
  },
  "mlProjectInfo": {
    "projectType": "Binary Image Classification System",
    "learningParadigm": "Deep Learning",
    "modelDomain": "Computer Vision Animal Recognition",
    "projectStatus": "Completed",
    "developmentYear": "2024",
    "capabilityBadges": [
      "MobileNetV2 Fine-Tuning",
      "Grad-CAM Explainability",
      "Streamlit Inference App"
    ]
  },
  "mlProblemContext": "Distinguishing cats from dogs across varied breeds, poses, lighting, and domestic or outdoor backgrounds challenges scratch-trained CNNs on small datasets. This project benchmarks a custom baseline against MobileNetV2 transfer learning while enforcing strict hold-out isolation to eliminate data leakage.",
  "mlDatasetOverview": {
    "profileBadge": "3,000 RGB JPEGs",
    "source": "Local Cat & Dog Image Archive (dataset/original/train and validation)",
    "modality": "Computer Vision",
    "scale": "3,000 Images (1,600 Train / 400 Val / 1,000 Hold-Out Test)",
    "inputShape": "3 x 224 x 224 RGB Tensors",
    "dataFormat": "JPEG RGB Images",
    "description": "Perfectly balanced binary image collection comprising 1,500 cat and 1,500 dog photographs. The 2,000-image training archive is stratified 80:20 into internal train (1,600) and validation (400) splits, while the 1,000-image validation archive remains strictly isolated as a final hold-out test set.",
    "provenanceNote": "Verified via automated MD5 hash deduplication and PIL integrity checks (0 corrupted files, 0 duplicate clusters)."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "Spatial RGB Tensor Space",
    "spaceNote": "Variable raw resolutions standardized to ImageNet-normalized 224x224 tensors.",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "Binary Species Label (cats = 0, dogs = 1)",
      "type": "Binary Class",
      "distribution": "Exact 50:50 class balance across internal train (800/800), validation (200/200), and hold-out test (500/500)."
    },
    "inputSignals": [
      {
        "name": "RGB Pixel Intensity Channels",
        "type": "Spatial Tensor",
        "importanceOrRole": "Primary 3-Channel Input",
        "description": "Normalized red, green, and blue pixel matrices capturing fur texture, facial contours, and color contrast."
      },
      {
        "name": "Raw Image Width & Height",
        "type": "Spatial Metadata",
        "importanceOrRole": "Preprocessing Normalization",
        "description": "Original dimensions span 59-1,050 px wide and 41-768 px tall (mean 405.66 x 360.20 px, aspect ratio 1.1617)."
      },
      {
        "name": "Grad-CAM Last-Layer Feature Maps",
        "type": "Convolutional Activation",
        "importanceOrRole": "Spatial Attribution",
        "description": "High-level 7x7 feature activations in MobileNetV2 block 18 highlighting eyes, ears, and muzzle regions."
      }
    ],
    "dataCharacteristicsSummary": "All 3,000 JPEG RGB images passed integrity and MD5 deduplication checks with zero corrupted or duplicate files. Image dimensions span 59-1,050 px wide and 41-768 px tall (mean aspect ratio 1.1617), requiring standardized 224x224 cropping and ImageNet channel normalization across balanced 50:50 class splits."
  },
  "mlLearningApproach": {
    "paradigm": "Supervised Deep Learning via Binary Logit Optimization & Two-Stage Transfer Learning",
    "methodologyDescription": "A two-stage transfer learning strategy compares a 4-block Custom CNN trained from scratch against an ImageNet-pretrained MobileNetV2. Stage 1 freezes the convolutional backbone to train a Dropout-Linear head, while Stage 2 unfreezes blocks 14-18 with differential learning rates for domain adaptation.",
    "justification": "On 1,600 training images, scratch-trained CNNs suffer representation underfitting (70.50% validation accuracy), whereas reusing MobileNetV2 ImageNet filters followed by low-learning-rate top-layer fine-tuning preserves edge/texture priors and raises validation accuracy to 98.25%."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "Image Integrity & MD5 Deduplication Audit",
      "description": "Scans all 3,000 JPEG files with PIL verification, confirms 100% RGB color mode, and validates zero duplicate MD5 hash clusters across splits.",
      "badge": "Data Audit"
    },
    {
      "step": "02",
      "title": "Stratified Split & Hold-Out Isolation",
      "description": "Partitions train/ (2,000 images) 80:20 with random seed 42 into internal train and validation splits while locking validation/ (1,000) as a one-time hold-out test set.",
      "badge": "Anti-Leakage"
    },
    {
      "step": "03",
      "title": "Stochastic Training Augmentation",
      "description": "Applies RandomResizedCrop(224), RandomHorizontalFlip(0.5), RandomRotation(15), and ColorJitter exclusively to internal training batches.",
      "badge": "Augmentation"
    },
    {
      "step": "04",
      "title": "Deterministic Evaluation Scaling & Normalization",
      "description": "Standardizes validation and hold-out images via Resize(256), CenterCrop(224), and ImageNet mean/std normalization.",
      "badge": "Normalization"
    }
  ],
  "mlDataPreparationNote": "Random geometric and photometric augmentations are restricted solely to internal training; internal validation and hold-out test sets use deterministic resizing and ImageNet normalization.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "MobileNetV2 Fine-Tuned vs CustomBaselineCNN",
    "architectureSummary": "ImageNet-pretrained MobileNetV2 inverted residual backbone with a custom Dropout(0.3) + Linear(1280, 1) classifier head, fine-tuned across blocks 14-18 and benchmarked against a 4-block Conv2d-BatchNorm-ReLU-MaxPool CustomBaselineCNN with Global Average Pooling.",
    "configurationBadge": "Two-Stage Transfer Learning",
    "hyperparameters": [
      {
        "name": "Input Resolution",
        "value": "224 x 224 RGB",
        "description": "Standardized spatial input tensor dimension for all three models."
      },
      {
        "name": "Batch Size",
        "value": "32",
        "description": "Mini-batch size used across train, validation, and hold-out loaders."
      },
      {
        "name": "Learning Rate (Fine-Tuning)",
        "value": "1e-4 / 5e-4",
        "description": "Differential Adam learning rates for unfrozen backbone blocks 14-18 and classifier head."
      },
      {
        "name": "Weight Decay",
        "value": "1e-4",
        "description": "L2 regularization coefficient applied across all Adam optimizers."
      },
      {
        "name": "LR Scheduler",
        "value": "ReduceLROnPlateau",
        "description": "Halves learning rate (factor 0.5) when validation loss plateaus (patience 2-3)."
      },
      {
        "name": "Max Epochs",
        "value": "15 / 8 / 8",
        "description": "15 epochs for Custom CNN baseline; 8 epochs each for frozen and fine-tuned MobileNetV2."
      }
    ],
    "objectiveCriterion": "Binary Cross-Entropy with Logits (BCEWithLogitsLoss)",
    "optimizationSolver": "Adam Optimizer with ReduceLROnPlateau & Early Stopping",
    "trainingProcess": "Models train on 1,600 augmented images (batch size 32) using Adam, BCEWithLogitsLoss, and ReduceLROnPlateau (factor 0.5) monitored on 400 validation images. Best checkpoints save on lowest validation loss; MobileNetV2 Fine-Tuned peaks at epoch 6 with 98.25% validation accuracy and 0.0530 validation loss."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "Stratified Internal Validation Selection + One-Time Hold-Out Test",
    "description": "All three architectures are compared on the 400-image internal validation set to choose the best model objectively. The winning MobileNetV2 Fine-Tuned checkpoint is then evaluated once on the locked 1,000-image hold-out test set.",
    "metrics": [
      {
        "label": "TEST ACCURACY",
        "value": "97.70%",
        "benchmark": "Baseline: 70.50%"
      },
      {
        "label": "TEST F1-SCORE",
        "value": "97.71%",
        "benchmark": "Baseline: 69.43%"
      },
      {
        "label": "TEST PRECISION",
        "value": "97.23%",
        "benchmark": "Target > 95.00%"
      },
      {
        "label": "TEST RECALL",
        "value": "98.20%",
        "benchmark": "Target > 95.00%"
      },
      {
        "label": "TEST ROC-AUC",
        "value": "0.9987",
        "benchmark": "Baseline: 0.7682"
      },
      {
        "label": "TEST PR-AUC",
        "value": "0.9988",
        "benchmark": "Baseline: 0.7781"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Streamlit Inference & Confidence Breakdown",
      "badge": "Web App UI",
      "imageUrl": "/images/projects/cat_and_dog_recognition/image01.webp",
      "telemetryNote": "Displays active MobileNetV2 Fine-Tuned telemetry alongside 100.00% confidence Cat classification."
    },
    {
      "title": "Real-Time Grad-CAM Attention Overlay",
      "badge": "Explainability UI",
      "imageUrl": "/images/projects/cat_and_dog_recognition/image02.webp",
      "telemetryNote": "Highlights heatmap attention focused on facial features and eyes within the Streamlit inference panel."
    },
    {
      "title": "Hold-Out Metrics, Confusion Matrix & ROC Curve",
      "badge": "Test Evaluation",
      "imageUrl": "/images/projects/cat_and_dog_recognition/image03.webp",
      "telemetryNote": "Summarizes 97.70% accuracy, 486 TN / 14 FP / 9 FN / 491 TP matrix, and 0.9987 ROC-AUC."
    },
    {
      "title": "Model Benchmark & Anti-Leakage Overview",
      "badge": "Comparison UI",
      "imageUrl": "/images/projects/cat_and_dog_recognition/image04.webp",
      "telemetryNote": "Contrasts Custom CNN vs MobileNetV2 stages alongside prediction grids and split rules."
    },
    {
      "title": "Dataset Samples, Error Cases & Grad-CAM Collage",
      "badge": "Visual Diagnostics",
      "imageUrl": "/images/projects/cat_and_dog_recognition/image05.webp",
      "telemetryNote": "Pairs sample images with TP/TN/FP/FN predictions and Grad-CAM overlays across correct and misclassified cases."
    },
    {
      "title": "Image Width, Height & Aspect Ratio Distribution",
      "badge": "EDA Histogram",
      "imageUrl": "/images/projects/cat_and_dog_recognition/image06.webp",
      "telemetryNote": "Documents median 447 px width and 374 px height across 3,000 images before 224x224 scaling."
    },
    {
      "title": "Training & Validation Loss Convergence Curves",
      "badge": "Loss Curve",
      "imageUrl": "/images/projects/cat_and_dog_recognition/image07.webp",
      "telemetryNote": "Shows smooth validation loss reduction down to 0.0530 at epoch 6 for MobileNetV2 Fine-Tuned."
    },
    {
      "title": "Training & Validation Accuracy Trajectories",
      "badge": "Accuracy Curve",
      "imageUrl": "/images/projects/cat_and_dog_recognition/image08.webp",
      "telemetryNote": "Illustrates progression from 70.50% baseline to 95.75% frozen and 98.25% fine-tuned validation accuracy."
    },
    {
      "title": "Hold-Out ROC & Precision-Recall Curves",
      "badge": "ROC / PR Curves",
      "imageUrl": "/images/projects/cat_and_dog_recognition/image09.webp",
      "telemetryNote": "Confirms near-ideal separability on 1,000 hold-out images with 0.9987 ROC-AUC and 0.9988 PR-AUC."
    },
    {
      "title": "Composite Confusion Matrix, Comparison & Splits",
      "badge": "Summary Grid",
      "imageUrl": "/images/projects/cat_and_dog_recognition/image10.webp",
      "telemetryNote": "Consolidates hold-out confusion counts, internal validation bars, and 50:50 split balance."
    }
  ],
  "mlVisualizationTelemetryNote": "MobileNetV2 Fine-Tuned generalizes cleanly from 98.25% internal validation accuracy (0.0530 loss) to 97.70% hold-out test accuracy with only 23 errors across 1,000 unseen images.",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "Grad-CAM visualizations on the final MobileNetV2 convolutional layer confirm that predictions rely on anatomical cues—cat eyes, whiskers, and pointed ears versus dog muzzles, noses, and jaw contours—rather than background shortcuts, while misclassifications stem from toy breeds, long-haired cats, or occluded poses.",
    "keyDrivers": [
      {
        "factor": "Pretrained ImageNet Feature Hierarchy",
        "impact": "Raises internal validation accuracy from 70.50% (scratch CNN) to 95.75% with frozen weights alone."
      },
      {
        "factor": "Selective Top-Layer Fine-Tuning (Blocks 14-18)",
        "impact": "Adds +2.50% internal validation accuracy (to 98.25%) and reduces validation loss from 0.1481 to 0.0530."
      },
      {
        "factor": "Anatomical Head & Facial Attention",
        "impact": "Grad-CAM maps localize on eyes, whiskers, triangular ears, muzzles, and noses while ignoring room backgrounds."
      }
    ],
    "limitations": [
      {
        "title": "Modest Training Set Scale",
        "description": "Effective training relies on 1,600 images, limiting coverage of rare or hairless breeds."
      },
      {
        "title": "Single-Subject Binary Scope",
        "description": "Classifier assumes one primary animal per frame and cannot localize multiple pets or reject non-pet species."
      },
      {
        "title": "224x224 Input Downscaling",
        "description": "Resizing discards fine whisker and pupil detail in low-light, distant, or curled-up edge cases."
      }
    ],
    "futureImprovements": [
      {
        "title": "Vision Transformer Backbone Benchmarking",
        "target": "Evaluate MobileViT or Swin Transformer backbones for global spatial context."
      },
      {
        "title": "Multi-Object Detection Extension",
        "target": "Extend single-label classification into bounding-box detection for multi-pet scenes."
      },
      {
        "title": "Out-of-Distribution Rejection Class",
        "target": "Add negative wildlife/background classes and confidence thresholding to handle non-pet uploads."
      }
    ],
    "discoveryNote": "Hold-out errors total 14 cats (2.8% error rate) and 9 dogs (1.8% error rate), primarily on toy-breed puppies and long-haired or occluded cats."
  },
  "mlModelArtifacts": [
    {
      "name": "best_model.pt",
      "format": "PT"
    },
    {
      "name": "mobilenetv2_finetuned.pt",
      "format": "PT"
    },
    {
      "name": "mobilenetv2_feature_extractor.pt",
      "format": "PT"
    },
    {
      "name": "baseline_custom_cnn.pt",
      "format": "PT"
    },
    {
      "name": "model_config.json",
      "format": "JSON"
    },
    {
      "name": "metrics.json",
      "format": "JSON"
    },
    {
      "name": "training_histories.json",
      "format": "JSON"
    },
    {
      "name": "dataset_audit.json",
      "format": "JSON"
    },
    {
      "name": "dataset_split.json",
      "format": "JSON"
    },
    {
      "name": "classification_report.txt",
      "format": "TXT"
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Deep Learning",
    "modelArchitecture": "Fine-Tuned MobileNetV2 CNN",
    "coreFramework": "PyTorch",
    "validationMethod": "Stratified Split Hold-Out"
  }
}
```
