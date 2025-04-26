# Save this as check_dataset.py
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader

# Import your dataset class
from Data_Preprocessing import A2D2_CSV_dataset

# Paths
csv_file = '../dataset-loading/a2d2_image_mask_pairs.csv'
class_list = '../dataset-loading/class_list.json'
cache = '../data/cache'
output_dir = '../diagnostics'

# Parameters
height, width = 805, 1280  # Using your reduced dimensions
num_classes = 55

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Create transforms
image_transform = Resize((height, width), interpolation=InterpolationMode.BILINEAR)
target_transform = Resize((height, width), interpolation=InterpolationMode.NEAREST)

# Create dataset
dataset = A2D2_CSV_dataset(
    csv_file=csv_file,
    class_list=class_list,
    cache=cache,
    height=height,
    width=width,
    transform=image_transform,
    target_transform=target_transform
)

print(f"Dataset contains {len(dataset)} samples")


# Analyze class distribution across a subset of samples
def analyze_class_distribution(dataset, num_samples=50):
    class_counts = torch.zeros(num_classes)
    samples_analyzed = min(num_samples, len(dataset))

    print(f"Analyzing class distribution across {samples_analyzed} samples...")

    for i in range(samples_analyzed):
        _, mask = dataset[i]
        for c in range(num_classes):
            class_counts[c] += (mask == c).sum().item()

    total_pixels = class_counts.sum().item()
    class_percentages = (class_counts / total_pixels) * 100

    # Print detailed class distribution
    print("\nClass Distribution:")
    print("-" * 50)
    print(f"{'Class ID':<10} {'Pixel Count':<15} {'Percentage':<10}")
    print("-" * 50)

    for c in range(num_classes):
        if class_counts[c] > 0:
            print(f"{c:<10} {int(class_counts[c]):<15} {class_percentages[c]:.2f}%")

    # Plot distribution
    non_zero_classes = []
    non_zero_percentages = []

    for c in range(num_classes):
        if class_percentages[c] > 0:
            non_zero_classes.append(c)
            non_zero_percentages.append(class_percentages[c].item())

    plt.figure(figsize=(12, 6))
    plt.bar(non_zero_classes, non_zero_percentages)
    plt.xlabel('Class ID')
    plt.ylabel('Percentage of Pixels')
    plt.title('Class Distribution in Dataset')
    plt.xticks(non_zero_classes)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close()

    # Check for imbalance
    non_zero_counts = class_counts[class_counts > 0]
    imbalance_ratio = non_zero_counts.max().item() / non_zero_counts.min().item()
    print(f"\nClass imbalance ratio (max/min): {imbalance_ratio:.2f}")

    # Find dominant class
    dominant_class = torch.argmax(class_counts).item()
    dominant_percentage = class_percentages[dominant_class].item()
    print(f"Dominant class: {dominant_class} ({dominant_percentage:.2f}% of all pixels)")

    return class_counts, dominant_class, dominant_percentage


# Visualize a few samples with their masks
def visualize_samples(dataset, num_samples=5):
    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)

    for i in range(num_samples):
        image, mask = dataset[i]

        # Convert image to numpy for visualization
        img_np = image.permute(1, 2, 0).numpy()

        # Create a colored mask for visualization
        mask_np = mask.numpy()
        unique_classes = np.unique(mask_np)

        plt.figure(figsize=(15, 7))

        # Plot original image
        plt.subplot(1, 3, 1)
        plt.imshow(img_np)
        plt.title(f"Sample {i} - Original Image")
        plt.axis('off')

        # Plot mask (colored by class)
        plt.subplot(1, 3, 2)
        plt.imshow(mask_np, cmap='tab20', vmin=0, vmax=num_classes - 1)
        plt.title(f"Segmentation Mask\nClasses: {unique_classes}")
        plt.axis('off')

        # Plot mask with dominant class highlighted
        plt.subplot(1, 3, 3)
        dominant_class = np.bincount(mask_np.flatten()).argmax()
        dominant_mask = (mask_np == dominant_class)

        # Create RGB image with dominant class highlighted
        highlight = np.zeros((height, width, 3), dtype=np.float32)
        highlight[:, :, 0] = dominant_mask * 1.0  # Red channel for dominant class

        plt.imshow(highlight)
        dominant_percent = (dominant_mask.sum() / (height * width)) * 100
        plt.title(f"Dominant Class: {dominant_class}\n({dominant_percent:.1f}% of image)")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'samples', f'sample_{i}.png'))
        plt.close()

        print(f"Analyzed sample {i} - Classes present: {unique_classes}")
        print(f"  Dominant class {dominant_class} covers {dominant_percent:.1f}% of image")


# Run analysis
class_counts, dominant_class, dominant_percentage = analyze_class_distribution(dataset)
visualize_samples(dataset)

print("\nAnalysis complete. Results saved to:", output_dir)