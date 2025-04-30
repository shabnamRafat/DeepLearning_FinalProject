import streamlit as st
import numpy as np
import json
from PIL import Image
from pathlib import Path

# =====================
# PAGE SETUP
# =====================
st.set_page_config(page_title="Mask Analysis App", layout="wide")
st.title("🖼️ Image Segmentation: Masking and Analysis")
st.markdown("---")

# =====================
# FUNCTIONS
# =====================
def analyze_label(label_array):
    total_pixels = label_array.size
    unique, counts = np.unique(label_array, return_counts=True)
    class_distribution = dict(zip(unique, counts))

    assigned_pixels = sum(counts)
    dominant_class = max(class_distribution, key=class_distribution.get)
    dominant_percentage = (class_distribution[dominant_class] / total_pixels) * 100

    return total_pixels, assigned_pixels, class_distribution, dominant_class, dominant_percentage


def convert_hex_json_to_indexed(json_path):
    try:
        with open(json_path, 'r') as f:
            hex_class_mapping = json.load(f)

        indexed_mapping = {}
        rgb_to_index = {}

        for i, (hex_color, class_name) in enumerate(hex_class_mapping.items()):
            indexed_mapping[str(i)] = class_name

            # Convert hex to RGB
            hex_clean = hex_color.lstrip("#")
            rgb = tuple(int(hex_clean[j:j + 2], 16) for j in (0, 2, 4))
            rgb_to_index[rgb] = i

        return indexed_mapping, rgb_to_index
    except Exception as e:
        print(f"⚠️ Warning: Failed to load class list JSON. Reason: {e}")
        return {}, {}

# =====================
# SIDEBAR INPUTS
# =====================
st.sidebar.header("📂 Provide Folders")

image_folder_path = st.sidebar.text_input("Path to Image Folder (PNG)", value="/path/to/images")
mask_folder_path = st.sidebar.text_input("Path to Mask Folder (PNG)", value="/path/to/masks")
class_list_path = st.sidebar.text_input("Path to Class List JSON", value="/path/to/class_list.json")

start_analysis = st.sidebar.button("🔍 Start Analysis")

# =====================
# MAIN LOGIC
# =====================
if start_analysis:
    image_folder = Path(image_folder_path)
    mask_folder = Path(mask_folder_path)
    class_list_file = Path(class_list_path)

    if not image_folder.exists() or not mask_folder.exists() or not class_list_file.exists():
        st.error("❌ Provided folder paths are incorrect. Please check and try again.")
    else:
        image_files = sorted(list(image_folder.glob("*.png")))
        mask_files = sorted(list(mask_folder.glob("*.png")))

        st.success(f"✅ Found {len(image_files)} images and {len(mask_files)} masks.")

        # Load class list and RGB mapping
        class_mapping, rgb_to_index = convert_hex_json_to_indexed(class_list_file)

        if len(image_files) != len(mask_files):
            st.warning("⚠️ Number of images and masks do not match exactly!")

        for idx, (img_path, mask_path) in enumerate(zip(image_files, mask_files)):
            st.subheader(f"Sample {idx+1}: {img_path.name}")

            img = Image.open(img_path).convert("RGB")
            mask_rgb = np.array(Image.open(mask_path).convert("RGB"))

            # Convert RGB mask to class index mask
            label_array = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
            for rgb, index in rgb_to_index.items():
                matches = np.all(mask_rgb == rgb, axis=-1)
                label_array[matches] = index

            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Input Image")
            with col2:
                st.image(mask_rgb, caption="Label Mask")

            total_pixels, assigned_pixels, class_distribution, dominant_class, dominant_percentage = analyze_label(label_array)

            st.markdown("📈 **Mask Analysis:**")
            st.write(f"- Total Pixels: {total_pixels}")
            st.write(f"- Assigned Pixels: {assigned_pixels} ({(assigned_pixels/total_pixels)*100:.2f}%)")

            # Display Class Distribution
            st.write("- Class Distribution:")
            for class_id, count in sorted(class_distribution.items()):
                class_id_str = str(class_id)
                if class_id_str in class_mapping:
                    class_name = class_mapping[class_id_str]
                    percentage = (count / assigned_pixels) * 100
                    st.write(f"  - Class ID {class_id} ({class_name}): {count} pixels ({percentage:.2f}%)")
                else:
                    continue  # Skip unknown classes silently

            st.write(f"- Dominant Class: {dominant_class} ({dominant_percentage:.2f}% of image)")
            st.divider()
