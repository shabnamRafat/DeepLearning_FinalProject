import os
import csv
import json
from PIL import Image, ImageDraw
import numpy as np
import torch
from torchvision.datasets.vision import VisionDataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Resize

#
# 1) CSV GENERATOR: map camera images to label images
#
root_dir = '../Data/camera_lidar_semantic/'
output_file = 'a2d2_image_mask_pairs.csv'

pairs = []
for seq in sorted(os.listdir(root_dir)):
    seq_path = os.path.join(root_dir, seq)
    if not os.path.isdir(seq_path):
        continue

    # Path to camera images
    camera_path = os.path.join(seq_path, 'camera', 'cam_front_center')
    if not os.path.isdir(camera_path):
        continue

    # Path to label images
    label_path = os.path.join(seq_path, 'label', 'cam_front_center')
    if not os.path.isdir(label_path):
        continue

    # Find matching pairs
    for img_file in sorted(os.listdir(camera_path)):
        if not img_file.lower().endswith('.png'):
            continue

        img_path = os.path.join(camera_path, img_file)

        # Construct the expected label filename
        # Format: 20181204170238_camera_frontcenter_000130622.png -> 20181204170238_label_frontcenter_000130622.png
        label_file = img_file.replace('camera', 'label')
        label_file_path = os.path.join(label_path, label_file)

        if os.path.exists(label_file_path):
            pairs.append((img_path, label_file_path))
        else:
            # Extract timestamp and try to find a matching label file
            parts = img_file.split('_')
            if len(parts) >= 4:
                timestamp = parts[0]
                frame_id = parts[-1]

                # Try to find a matching label file using timestamp and frame ID
                for label_file in os.listdir(label_path):
                    if timestamp in label_file and frame_id in label_file and label_file.lower().endswith('.png'):
                        label_file_path = os.path.join(label_path, label_file)
                        pairs.append((img_path, label_file_path))
                        break
                else:
                    print(f"⚠️  no matching label for {img_path}")
            else:
                print(f"⚠️  unexpected filename format: {img_path}")

print(f"Found {len(pairs)} image-label pairs")

with open(output_file, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['image_path', 'mask_path'])
    w.writerows(pairs)


#
# 2) DATASET CLASS: read RGB-encoded label images
#
class A2D2_CSV_dataset(VisionDataset):
    def __init__(self, csv_file, class_list, cache,
                 height, width, transform, target_transform):
        super().__init__(None, transform=transform, target_transform=target_transform)

        # build color→id map
        with open(class_list) as f:
            cls = json.load(f)

        hex2rgb = {
            hex_code: tuple(
                int(hex_code.strip("#")[i: i + 2], 16)
                for i in (0, 2, 4)
            )
            for hex_code in cls.keys()
        }

        # Print color mapping for debugging
        print(f"Loaded {len(hex2rgb)} color mappings from class_list.json")
        if len(hex2rgb) < 5:
            print("WARNING: Very few color mappings found. Check class_list.json format.")

        # invert
        self.rgb2ids = {rgb: i for i, rgb in enumerate(hex2rgb.values())}

        self.height, self.width, self.cache = height, width, cache

        self.record_list = []
        with open(csv_file) as f:
            r = csv.DictReader(f)
            for row in r:
                self.record_list.append((row['image_path'], row['mask_path']))

        print(f"Loaded {len(self.record_list)} image-label pairs from CSV")

    def __len__(self):
        return len(self.record_list)

    def __getitem__(self, idx):
        img_path, label_path = self.record_list[idx]

        # 1) load & transform image
        image = read_image(img_path, mode=ImageReadMode.RGB)
        image = self.transform(image)

        # 2) build mask from RGB label image
        label = read_image(label_path, mode=ImageReadMode.RGB)
        label = self.target_transform(label)
        mask = torch.zeros(self.height, self.width, dtype=torch.int64)

        # Debug: Count pixels before assignment
        total_mask_pixels = self.height * self.width
        assigned_pixels = 0
        class_counts = {}

        for rgb, cid in self.rgb2ids.items():
            m = (label == torch.tensor(rgb).view(3, 1, 1)).all(dim=0)
            pixel_count = m.sum().item()
            assigned_pixels += pixel_count
            if pixel_count > 0:
                class_counts[cid] = pixel_count
            mask[m] = cid

        # Debug: Check if all pixels were assigned a class
        if idx < 5:  # Only print for first few samples
            percent_assigned = (assigned_pixels / total_mask_pixels) * 100
            print(f"\nSample {idx} (Label image):")
            print(f"  Total pixels: {total_mask_pixels}")
            print(f"  Assigned pixels: {assigned_pixels} ({percent_assigned:.2f}%)")
            print(f"  Class distribution: {class_counts}")
            if percent_assigned < 99:
                print(f"  WARNING: {100 - percent_assigned:.2f}% of pixels not assigned to any class!")

                # If few pixels are assigned, check for RGB values in the label
                if percent_assigned < 10:
                    unique_colors = set()
                    label_np = label.permute(1, 2, 0).cpu().numpy()
                    for h in range(min(100, label_np.shape[0])):
                        for w in range(min(100, label_np.shape[1])):
                            unique_colors.add(tuple(label_np[h, w]))

                    print(f"  First 10 unique RGB values in label: {list(unique_colors)[:10]}")
                    print(f"  Total unique RGB values: {len(unique_colors)}")

            # Check for dominant class
            if class_counts:
                dominant_class = max(class_counts.items(), key=lambda x: x[1])
                dominant_percent = (dominant_class[1] / total_mask_pixels) * 100
                print(f"  Dominant class: {dominant_class[0]} ({dominant_percent:.2f}% of image)")

                # Check for extreme imbalance
                if dominant_percent > 95:
                    print(f"  ALERT: Extreme class imbalance detected!")

        # normalize image & return
        return image.float().div(255), mask


#
# 3) EXAMPLE USAGE (make sure to pass these args in your training script)
#
if __name__ == "__main__":
    from torchvision.transforms import Compose

    # example paths — replace with your argparse args
    csv_file = 'a2d2_image_mask_pairs.csv'
    class_list = 'class_list.json'
    cache = '/tmp/cache'
    height, width = 1208, 1920

    # simple resize transforms
    image_transform = Resize((height, width), interpolation=InterpolationMode.BILINEAR)
    target_transform = Resize((height, width), interpolation=InterpolationMode.NEAREST)

    ds = A2D2_CSV_dataset(
        csv_file, class_list, cache,
        height, width, image_transform, target_transform
    )

    print("Found", len(ds), "samples")
    im, m = ds[0]
    print("Image:", im.shape, "Mask:", m.shape)

    # Visualize a few samples to verify
    try:
        import matplotlib.pyplot as plt

        # Create output directory for diagnostics
        os.makedirs("dataset_diagnostics", exist_ok=True)

        for i in range(min(5, len(ds))):
            img, mask = ds[i]

            plt.figure(figsize=(15, 5))

            # Display image
            plt.subplot(1, 2, 1)
            plt.imshow(img.permute(1, 2, 0).cpu())
            plt.title(f"Image {i}")
            plt.axis('off')

            # Display mask
            plt.subplot(1, 2, 2)
            plt.imshow(mask.cpu(), cmap='tab20')
            plt.title(f"Mask {i}")
            plt.axis('off')

            plt.savefig(f"dataset_diagnostics/sample_{i}.png")
            plt.close()

        print("Diagnostic visualizations saved to 'dataset_diagnostics' folder")
    except ImportError:
        print("Matplotlib not available for visualization")