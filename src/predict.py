"""
Inference module for Cat and Dog Classification.
Loads trained best model, performs single-image or batch prediction,
and supports Grad-CAM visualization generation.
"""

import os
import sys
import json
from pathlib import Path
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from src.models import CustomBaselineCNN, build_mobilenet_v2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = Path("models")
CONFIG_PATH = MODELS_DIR / "model_config.json"
MODEL_PATH = MODELS_DIR / "best_model.pt"

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class GradCAM:
    """
    Grad-CAM implementation for convolutional neural networks.
    Hooks into the specified target layer to capture gradients and feature maps.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, target_class: int = 1) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        # Binary logit
        if target_class == 1:
            score = output[0, 0]
        else:
            score = -output[0, 0]

        score.backward(retain_graph=True)

        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)  # [C, 1, 1]
        cam = torch.sum(weights * activations, dim=0)  # [H, W]
        cam = torch.relu(cam).cpu().numpy()

        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        return cam


class CatDogPredictor:
    def __init__(self, model_path: str = str(MODEL_PATH), config_path: str = str(CONFIG_PATH)):
        self.device = DEVICE
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        arch = self.config.get("best_model_architecture", "MobileNetV2_FineTuned")
        if "MobileNet" in arch:
            self.model = build_mobilenet_v2(pretrained=False, freeze_features=False)
            self.target_layer = self.model.features[-1]
        else:
            self.model = CustomBaselineCNN()
            self.target_layer = self.model.features[-1]

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        self.gradcam = GradCAM(self.model, self.target_layer)

    def predict(self, image_input, generate_cam: bool = False):
        """
        image_input: filepath (str/Path) or PIL Image.
        Returns dict with prediction details.
        """
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise ValueError("Unsupported image input type. Use file path or PIL Image.")

        tensor_img = EVAL_TRANSFORM(pil_img).unsqueeze(0).to(self.device)

        with torch.set_grad_enabled(generate_cam):
            tensor_img.requires_grad = True if generate_cam else False
            output = self.model(tensor_img)
            logit = output.item()
            prob_dog = float(torch.sigmoid(output).item())
            prob_cat = 1.0 - prob_dog

        pred_label = 1 if prob_dog >= 0.5 else 0
        pred_class = "Dog" if pred_label == 1 else "Cat"
        confidence = prob_dog if pred_label == 1 else prob_cat

        cam_mask = None
        if generate_cam:
            cam_mask = self.gradcam.generate(tensor_img, target_class=pred_label)

        return {
            "predicted_class": pred_class,
            "predicted_label": pred_label,
            "confidence": round(confidence * 100, 2),
            "dog_probability": round(prob_dog * 100, 2),
            "cat_probability": round(prob_cat * 100, 2),
            "raw_logit": round(logit, 4),
            "pil_image": pil_img,
            "cam_mask": cam_mask
        }


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <path_to_image>")
        # Example test with an existing image
        sample_img = "dataset/validation/cats/cat.2000.jpg"
        if os.path.exists(sample_img):
            print(f"Running sample prediction on {sample_img}...")
            predictor = CatDogPredictor()
            res = predictor.predict(sample_img)
            print("Result:", res)
        return

    img_path = sys.argv[1]
    predictor = CatDogPredictor()
    res = predictor.predict(img_path)
    print(f"\nImage: {img_path}")
    print(f"Prediction: {res['predicted_class']} (Confidence: {res['confidence']}%)")
    print(f"Probabilities -> Cat: {res['cat_probability']}% | Dog: {res['dog_probability']}%")


if __name__ == "__main__":
    main()
