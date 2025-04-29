# streamlit_evaluation_app.py

import streamlit as st
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
import numpy as np
import os
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

start_eval = st.sidebar.button("🔎 Run Evaluation")

if os.path.isdir(model_path):
    model_path = os.path.join(model_path, "best_model.pth")

if not os.path.exists(model_path):
    st.error(f"❌ Model path not found: {model_path}")
    st.stop()

# =====================
# RGB TO CLASS MAPPING
# =====================
rgb_to_class = {
    (255, 0, 0): 0, (200, 0, 0): 1, (150, 0, 0): 2, (128, 0, 0): 3,
    (182, 89, 6): 4, (150, 50, 4): 5, (90, 30, 1): 6, (90, 30, 30): 7,
    (204, 153, 255): 8, (189, 73, 155): 9, (239, 89, 191): 10,
    (255, 128, 0): 11, (200, 128, 0): 12, (150, 128, 0): 13,
    (0, 255, 0): 14, (0, 200, 0): 15, (0, 150, 0): 16,
    (0, 128, 255): 17, (30, 28, 158): 18, (60, 28, 100): 19,
    (0, 255, 255): 20, (30, 220, 220): 21, (60, 157, 199): 22,
    (255, 255, 0): 23, (255, 255, 200): 24, (233, 100, 0): 25,
    (110, 110, 0): 26, (128, 128, 0): 27, (255, 193, 37): 28,
    (64, 0, 64): 29, (185, 122, 87): 30, (0, 0, 100): 31,
    (139, 99, 108): 32, (210, 50, 115): 33, (255, 0, 128): 34,
    (255, 246, 143): 35, (150, 0, 150): 36, (204, 255, 153): 37,
    (238, 162, 173): 38, (33, 44, 177): 39, (180, 50, 180): 40,
    (255, 70, 185): 41, (238, 233, 191): 42, (147, 253, 194): 43,
    (150, 150, 200): 44, (180, 150, 200): 45, (72, 209, 204): 46,
    (200, 125, 210): 47, (159, 121, 238): 48, (128, 0, 255): 49,
    (255, 0, 255): 50, (135, 206, 255): 51, (241, 230, 255): 52,
    (96, 69, 143): 53, (53, 46, 82): 54
}
class_names = list(rgb_to_class.values())

# =====================
# FUNCTIONS
# =====================
class TestDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, height=640, width=640):
        self.image_paths = sorted(list(Path(image_dir).glob("*.png")))
        self.mask_paths = sorted(list(Path(mask_dir).glob("*.png"))) if mask_dir else None
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
            for rgb, idx_class in rgb_to_class.items():
                label_mask[np.all(mask == rgb, axis=-1)] = idx_class
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
    intersection = torch.diag(hist)
    union = hist.sum(dim=1) + hist.sum(dim=0) - intersection
    iou = intersection.float() / (union.float().clamp(min=1))
    pixel_acc = intersection.sum().float() / hist.sum().float().clamp(min=1)
    return iou.mean().item(), pixel_acc.item(), iou.cpu().numpy()

def visualize_overlap(pred_mask, true_mask):
    overlap_map = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    match = pred_mask == true_mask
    overlap_map[match] = [0, 255, 0]  # TP: green
    overlap_map[(pred_mask != true_mask) & (pred_mask != 0)] = [255, 0, 0]  # FP: red
    overlap_map[(pred_mask != true_mask) & (true_mask != 0)] = [0, 0, 255]  # FN: blue
    return overlap_map

def plot_classwise_iou(iou_per_class, class_names):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(class_names)), iou_per_class)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_ylabel("IoU")
    ax.set_title("Class-wise IoU")
    st.pyplot(fig)

def plot_confusion_matrix(hist, class_names):
    cm = hist.cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names, cmap='viridis', norm='log')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (log scale)")
    st.pyplot(fig)

@st.cache_resource
def load_model(model_path, device, num_classes=55):
    model = torch.hub.load("pytorch/vision:v0.10.0", "deeplabv3_resnet50", pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# =====================
# MAIN LOGIC
# =====================
if start_eval:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    test_dataset = TestDataset(test_image_dir, test_mask_dir)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    hist = torch.zeros(55, 55, dtype=torch.int64, device=device)
    preds_list, masks_list = [], []

    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        with torch.no_grad():
            outputs = model(images)["out"]
            preds = outputs.argmax(dim=1)
            preds_list.append(preds.cpu().numpy())
            masks_list.append(masks.cpu().numpy())
            hist += _fast_hist(preds.view(-1), masks.view(-1), 55)

    mean_iou, pixel_acc, iou_per_class = compute_metrics(hist)

    st.success("✅ Evaluation Completed!")
    st.metric("Mean IoU", f"{mean_iou:.4f}")
    st.metric("Pixel Accuracy", f"{pixel_acc:.4f}")

    # Plot classwise IoU
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(range(len(class_names)), iou_per_class)
    ax1.set_xticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=90)
    ax1.set_ylabel("IoU")
    ax1.set_title("Class-wise IoU")
    st.pyplot(fig1)

    # Visualize one sample
    pred_sample = preds_list[0][0]
    mask_sample = masks_list[0][0]
    overlap_img = visualize_overlap(pred_sample, mask_sample)

    st.markdown("### 🖼️ Overlap Visualization (Green=TP, Red=FP, Blue=FN)")
    st.image(overlap_img, caption="Overlap Map", use_column_width=True)