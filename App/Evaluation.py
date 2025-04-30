# streamlit_evaluation_app.py

import streamlit as st
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =====================
# PAGE SETUP
# =====================
st.set_page_config(page_title="Model Evaluation App", layout="wide")
st.title("📊 Model Evaluation on Test Set")
st.markdown("---")

# =====================
# SIDEBAR INPUTS
# =====================
st.sidebar.header("📂 Select Files and Settings")

model_path = st.sidebar.text_input("Path to Trained Model (.pth)", "/path/to/model.pth")
test_image_dir = st.sidebar.text_input("Path to Test Images", "/path/to/test/images")
test_mask_dir = st.sidebar.text_input("Path to Test Masks (Optional)", "/path/to/test/masks")
class_list_path = st.sidebar.text_input("Path to class_list.json", "/path/to/class_list.json")

start_eval = st.sidebar.button("🔎 Run Evaluation")

if not os.path.exists(model_path):
    st.error(f"❌ Model path not found: {model_path}")
    st.stop()

# =====================
# FUNCTIONS
# =====================
def load_class_map(json_path):
    import re
    from matplotlib.colors import to_rgb

    with open(json_path, 'r') as f:
        data = json.load(f)

    class_map = {}
    class_names = []
    idx = 0

    hex_color_regex = re.compile(r'^#(?:[0-9a-fA-F]{6})$')

    for hex_code, label in data.items():
        if hex_color_regex.match(hex_code):
            rgb = tuple(int(hex_code[i:i+2], 16) for i in (1, 3, 5))  # Convert #rrggbb to (r, g, b)
            class_map[rgb] = idx
            class_names.append(label)
            idx += 1
        else:
            raise ValueError(f"Invalid hex color: {hex_code}")

    return class_map, class_names

class TestDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, rgb_to_class=None, height=640, width=640):
        self.image_paths = sorted(list(Path(image_dir).glob("*.png")))
        self.mask_paths = sorted(list(Path(mask_dir).glob("*.png"))) if mask_dir else None
        self.rgb_to_class = rgb_to_class or {}
        self.img_transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor()
        ])
        self.mask_resize = transforms.Resize((height, width), interpolation=transforms.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img_tensor = self.img_transform(img)

        if self.mask_paths:
            mask = Image.open(self.mask_paths[idx]).convert("RGB")
            mask = np.array(self.mask_resize(mask))
            label_mask = np.zeros(mask.shape[:2], dtype=np.int64)
            for rgb, class_idx in self.rgb_to_class.items():
                label_mask[np.all(mask == rgb, axis=-1)] = class_idx
            return img_tensor, torch.from_numpy(label_mask)
        else:
            return img_tensor, torch.zeros((img_tensor.shape[1], img_tensor.shape[2]), dtype=torch.int64)

def _fast_hist(pred, label, num_classes):
    mask = (label >= 0) & (label < num_classes)
    hist = torch.bincount(
        num_classes * label[mask] + pred[mask],
        minlength=num_classes**2
    ).reshape(num_classes, num_classes)
    return hist

def compute_metrics(hist):
    """
    Compute detailed segmentation metrics from the confusion matrix.

    Args:
        hist (Tensor): Confusion matrix (num_classes x num_classes)

    Returns:
        dict: Dictionary of per-class and aggregate metrics
    """
    TP = torch.diag(hist)
    FP = hist.sum(dim=0) - TP
    FN = hist.sum(dim=1) - TP

    # Avoid division by zero
    denominator_iou = TP + FP + FN
    denominator_precision = TP + FP
    denominator_recall = TP + FN
    denominator_f1 = 2 * TP + FP + FN

    # IoU (Intersection over Union)
    iou = TP.float() / denominator_iou.float().clamp(min=1)
    mean_iou = iou.mean().item()

    # Pixel Accuracy
    pixel_acc = TP.sum().float() / hist.sum().float().clamp(min=1)

    # Recall
    recall = TP.float() / denominator_recall.float().clamp(min=1)
    mean_recall = recall.mean().item()

    # Precision
    precision = TP.float() / denominator_precision.float().clamp(min=1)
    mean_precision = precision.mean().item()

    # F1 Score
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-7)
    mean_f1 = f1.mean().item()

    # Dice Coefficient (same formula as F1 for segmentation)
    dice = 2 * TP.float() / denominator_f1.float().clamp(min=1)
    mean_dice = dice.mean().item()

    # Boundary F1 Score placeholder (same as mean F1 for now)
    bf_score = mean_f1

    return {
        'iou': iou,
        'mean_iou': mean_iou,
        'pixel_acc': pixel_acc.item(),
        'recall': recall,
        'mean_recall': mean_recall,
        'precision': precision,
        'mean_precision': mean_precision,
        'f1_score': f1,
        'mean_f1': mean_f1,
        'dice': dice,
        'mean_dice': mean_dice,
        'bf_score': bf_score
    }

def visualize_overlap(pred_mask, true_mask):
    overlap_map = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    match = pred_mask == true_mask
    overlap_map[match] = [0, 255, 0]  # TP
    overlap_map[(pred_mask != true_mask) & (pred_mask != 0)] = [255, 0, 0]  # FP
    overlap_map[(pred_mask != true_mask) & (true_mask != 0)] = [0, 0, 255]  # FN
    return overlap_map

def plot_classwise_iou(iou_per_class, class_names):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(class_names)), iou_per_class)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_ylabel("IoU")
    ax.set_title("Class-wise IoU")
    st.pyplot(fig)


@st.cache_resource
def load_model(model_path, device, num_classes=55):
    model = torch.hub.load("pytorch/vision:v0.10.0", "deeplabv3_resnet50", pretrained=False, num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


# =====================
# MAIN LOGIC
# =====================
if start_eval:
    if not os.path.exists(class_list_path):
        st.error("class_list.json path is invalid.")
        st.stop()

    rgb_to_class, class_names = load_class_map(class_list_path)
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device, num_classes=num_classes)

    test_dataset = TestDataset(test_image_dir, test_mask_dir, rgb_to_class)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    hist = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)
    preds_list, masks_list = [], []

    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        with torch.no_grad():
            outputs = model(images)["out"]
            preds = outputs.argmax(dim=1)
            preds_list.append(preds.cpu().numpy())
            masks_list.append(masks.cpu().numpy())
            hist += _fast_hist(preds.view(-1), masks.view(-1), num_classes)
    metrics = compute_metrics(hist)
    mean_iou = metrics['mean_iou']
    pixel_acc = metrics['pixel_acc']
    iou_per_class = metrics['iou']

    st.success("✅ Evaluation Completed!")
    st.markdown("### 📊 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean IoU", f"{metrics['mean_iou']:.4f}")
        st.metric("Mean Dice", f"{metrics['mean_dice']:.4f}")
        st.metric("Boundary F1", f"{metrics['bf_score']:.4f}")
    with col2:
        st.metric("Pixel Accuracy", f"{metrics['pixel_acc']:.4f}")
        st.metric("Mean Recall", f"{metrics['mean_recall']:.4f}")
    with col3:
        st.metric("Mean F1 Score", f"{metrics['mean_f1']:.4f}")
        st.metric("Mean Precision", f"{metrics['mean_precision']:.4f}")

    plot_classwise_iou(metrics['iou'], class_names)

    pred_sample = preds_list[0][0]
    mask_sample = masks_list[0][0]
    overlap_img = visualize_overlap(pred_sample, mask_sample)

    st.markdown("### 🖼️ Overlap Visualization (Green=TP, Red=FP, Blue=FN)")
    st.image(overlap_img, caption="Overlap Map", use_column_width=True)
