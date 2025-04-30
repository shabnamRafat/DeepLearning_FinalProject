# streamlit_segmentation_direct.py

import streamlit as st
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# RGB TO CLASS MAPPING
# ==========================
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

# ==========================
# PAGE SETUP
# ==========================
st.set_page_config(page_title="Model Training App", layout="wide")
st.title("📊 Model Training ")
st.markdown("---")

# ==========================
# USER INPUTS
# ==========================
st.sidebar.header("🛠️ Configure Settings")

model_type = st.sidebar.selectbox("Select Model Architecture", ["DeepLabV3+ (ResNet-50)", "Custom CNN"])
epochs = st.sidebar.slider("Number of Training Epochs", min_value=1, max_value=50, value=5)
learning_rate = st.sidebar.select_slider("Learning Rate", options=[1e-5, 1e-4, 1e-3], value=1e-4)

st.subheader("📂 Input Folders Configuration")

image_folder = st.text_input("Path to Input Image Folder (RGB)", "C:/path/to/images/")
mask_folder = st.text_input("Path to Ground Truth Mask Folder", "C:/path/to/masks/")
save_checkpoint_path = st.text_input("Path to Save Trained Model", "C:/path/to/save/demo_model.pth")

st.markdown("---")

# ==========================
# TRAINING FUNCTION
# ==========================
def start_training(model_type, image_folder, mask_folder, epochs, learning_rate, save_checkpoint_path):
    st.info(f"🚀 Starting training with model: **{model_type}**")

    # Collect all image and mask paths
    image_paths = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    mask_paths = sorted([
        os.path.join(mask_folder, f)
        for f in os.listdir(mask_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if len(image_paths) != len(mask_paths) or len(image_paths) < 2:
        st.error("❌ You need at least 2 matching image-mask pairs.")
        st.stop()

    st.success(f"📷 Found {len(image_paths)} images and {len(mask_paths)} masks.")
    st.write(f"📁 Model will be saved at: `{save_checkpoint_path}`")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"Using device: **{device}**")

    # Load one image to get original size
    input_image = Image.open(image_paths[0]).convert("RGB")
    orig_w, orig_h = input_image.size
    st.write(f"🖼️ Original image size: {orig_w} x {orig_h}")

    target_size = (640, 640) if model_type == "DeepLabV3+ (ResNet-50)" else (256, 256)
    st.write(f"🔄 Resizing all images to: {target_size}")

    # Define transforms
    transform_img = T.Compose([
        T.Resize(target_size),
        T.ToTensor()
    ])
    transform_mask = T.Compose([
        T.Resize(target_size, interpolation=T.InterpolationMode.NEAREST)
    ])

    # Load and transform all images and masks
    input_images = []
    mask_images = []
    for img_path, mask_path in zip(image_paths, mask_paths):
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        img_tensor = transform_img(img)

        mask_np = np.array(transform_mask(mask))  # (H, W, 3)

        label_mask = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.int64)
        for rgb, class_idx in rgb_to_class.items():
            matches = np.all(mask_np == rgb, axis=-1)
            label_mask[matches] = class_idx

        mask_tensor = torch.from_numpy(label_mask).long()  # (H, W)

        input_images.append(img_tensor)
        mask_images.append(mask_tensor)

    input_tensor = torch.stack(input_images).to(device)  # (B, 3, H, W)
    mask_tensor = torch.stack(mask_images).to(device)    # (B, H, W)

    st.write(f"📐 Final input tensor shape: {tuple(input_tensor.shape)}")
    st.write(f"📐 Final mask tensor shape: {tuple(mask_tensor.shape)}")

    # ==========================
    # Model Build
    # ==========================
    if model_type == "DeepLabV3+ (ResNet-50)":
        model = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "deeplabv3_resnet50",
            pretrained=False,
            num_classes=55   # <-- changed here
        )
    else:
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 55, kernel_size=1)  # <-- changed here
        )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ==========================
    # Training loop
    # ==========================
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(input_tensor)
        if isinstance(outputs, dict) and "out" in outputs:
            outputs = outputs["out"]

        loss = criterion(outputs, mask_tensor)
        loss.backward()
        optimizer.step()
        st.write(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss.item():.4f}")

    if os.path.isdir(save_checkpoint_path):
        save_checkpoint_path = os.path.join(save_checkpoint_path, "demo_model.pth")

    torch.save(model.state_dict(), save_checkpoint_path)
    st.success(f"✅ Training Completed and Model Saved at `{save_checkpoint_path}`")

# ==========================
# ACTION ON BUTTON CLICK
# ==========================
train_button = st.button("🚀 Start Training!")

if train_button:
    start_training(
        model_type=model_type,
        image_folder=image_folder,
        mask_folder=mask_folder,
        epochs=epochs,
        learning_rate=learning_rate,
        save_checkpoint_path=save_checkpoint_path
    )
