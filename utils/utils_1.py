import os
import torch
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import random
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _fast_hist(pred, label, num_classes):
    """
    Confusion-matrix histogram for one flattened batch.

    Args:
        pred: Predicted labels (flattened)
        label: Ground truth labels (flattened)
        num_classes: Number of classes

    Returns:
        Confusion matrix as a tensor of shape (num_classes, num_classes)
    """
    mask = (label >= 0) & (label < num_classes)
    hist = torch.bincount(
        num_classes * label[mask] + pred[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist


def compute_metrics(hist):
    """
    Compute per-class IoU, mean IoU, and additional metrics from confusion matrix.

    Args:
        hist: Confusion matrix tensor of shape (num_classes, num_classes)

    Returns:
        Dictionary containing multiple metrics
    """
    intersection = torch.diag(hist)
    union = hist.sum(dim=1) + hist.sum(dim=0) - intersection
    iou = intersection.float() / (union.float().clamp(min=1))
    pixel_acc = intersection.sum().float() / hist.sum().float().clamp(min=1)

    # Add recall and precision metrics
    recall = intersection.float() / hist.sum(dim=1).float().clamp(min=1)
    precision = intersection.float() / hist.sum(dim=0).float().clamp(min=1)

    # Add F1 score
    f1_score = 2 * precision * recall / (precision + recall).clamp(min=1e-7)

    # Add Dice coefficient
    dice = (2 * intersection.float()) / (hist.sum(dim=1) + hist.sum(dim=0)).float().clamp(min=1)

    # Calculate Boundary F1 Score (BF) - simplified version
    # This is a placeholder - real boundary detection requires more complex processing
    bf_score = f1_score  # In real implementation, this would focus on boundary pixels

    return {
        'iou': iou,
        'mean_iou': iou.mean().item(),
        'pixel_acc': pixel_acc.item(),
        'recall': recall,
        'mean_recall': recall.mean().item(),
        'precision': precision,
        'mean_precision': precision.mean().item(),
        'f1_score': f1_score,
        'mean_f1': f1_score.mean().item(),
        'dice': dice,
        'mean_dice': dice.mean().item(),
        'bf_score': bf_score.mean().item()
    }


def calculate_class_weights(train_loader, num_classes, device):
    """
    Calculate class weights for handling class imbalance

    Args:
        train_loader: DataLoader for training data
        num_classes: Number of classes
        device: Device to run calculations on

    Returns:
        Tensor containing weights for each class
    """
    print("Calculating class weights...")
    class_pixels = torch.zeros(num_classes, device=device)

    # Count pixels for each class in training set
    for i, (_, masks) in enumerate(train_loader):
        masks = masks.to(device)
        for c in range(num_classes):
            class_pixels[c] += (masks == c).sum().item()

        # Progress update
        if i % 10 == 0:
            print(f"Processed {i} batches for class weights calculation")

        # Limit to a subset for efficiency
        if i >= 100:  # Only scan part of the dataset for efficiency
            break

    # Handle empty classes
    if (class_pixels == 0).any():
        min_non_zero = class_pixels[class_pixels > 0].min()
        class_pixels[class_pixels == 0] = min_non_zero

    # Calculate weights: inverse frequency
    total_pixels = class_pixels.sum()
    class_weights = total_pixels / (class_pixels * num_classes)

    # Normalize weights to prevent extremely large values
    class_weights = torch.clip(class_weights, 0.1, 10.0)

    # Print class distribution
    print("Class pixels distribution:")
    for c in range(num_classes):
        percent = 100 * class_pixels[c] / total_pixels
        print(f"Class {c}: {class_pixels[c]:.0f} pixels ({percent:.2f}%), weight: {class_weights[c]:.4f}")

    return class_weights


def save_segmentation_results(model, data_loader, output_dir, device, color_map=None, save_metrics=True):
    """
    Run inference on a dataset and save segmentation output images and metrics

    Args:
        model: Trained segmentation model
        data_loader: DataLoader containing images to segment
        output_dir: Directory where to save output masks
        device: Device to run inference on
        color_map: Optional dictionary mapping class IDs to RGB colors
        save_metrics: Whether to save per-image metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    all_metrics = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Handle different output formats
            outputs = model(inputs)
            if isinstance(outputs, torch.Tensor):
                logits = outputs
            elif isinstance(outputs, dict) and "out" in outputs:
                logits = outputs["out"]
            else:
                logits = outputs

            preds = logits.argmax(dim=1)

            # Compute metrics for this batch
            batch_hist = torch.zeros(logits.size(1), logits.size(1), dtype=torch.int64, device=device)
            for j in range(preds.size(0)):
                pred_flat = preds[j].view(-1)
                target_flat = targets[j].view(-1)
                batch_hist += _fast_hist(pred_flat, target_flat, logits.size(1))

            batch_metrics = compute_metrics(batch_hist)

            # Convert tensors to CPU/numpy for saving results
            preds_cpu = preds.cpu().numpy()
            targets_cpu = targets.cpu().numpy()

            # Save each prediction in the batch
            for j, pred in enumerate(preds_cpu):
                img_metrics = {}

                # Calculate per-image metrics
                if save_metrics:
                    img_hist = _fast_hist(
                        torch.from_numpy(pred.flatten()).to(device),
                        torch.from_numpy(targets_cpu[j].flatten()).to(device),
                        logits.size(1)
                    )
                    img_metrics = compute_metrics(img_hist)
                    # Convert tensor values to Python types for JSON serialization
                    img_metrics = {k: v if isinstance(v, (int, float)) else v.tolist()
                                   for k, v in img_metrics.items()}

                # Convert class predictions to RGB if color map provided
                if color_map:
                    rgb_mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
                    for class_id, color in color_map.items():
                        if isinstance(class_id, str):
                            class_id = int(class_id)
                        rgb_mask[pred == class_id] = color
                    img = Image.fromarray(rgb_mask)
                else:
                    # Otherwise save as grayscale class ID image
                    img = Image.fromarray(pred.astype(np.uint8))

                # Save the image
                img_filename = f"prediction_{i}_{j}.png"
                img_path = os.path.join(output_dir, img_filename)
                img.save(img_path)

                # Save metrics for this image if requested
                if save_metrics:
                    metrics_filename = f"metrics_{i}_{j}.json"
                    metrics_path = os.path.join(output_dir, metrics_filename)
                    with open(metrics_path, 'w') as f:
                        json.dump(img_metrics, f, indent=2)

                # Add to all metrics with image filename
                if save_metrics:
                    all_metrics.append({
                        'filename': img_filename,
                        'metrics': img_metrics
                    })

            if i % 10 == 0:
                print(f"Processed {i} batches")

    # Save overall metrics summary
    if save_metrics:
        summary_path = os.path.join(output_dir, "metrics_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)

    print(f"Segmentation results saved to {output_dir}")


def generate_class_activation_maps(model, data_loader, output_dir, device, num_samples=5):
    """
    Generate class activation maps to visualize what the model is focusing on

    Args:
        model: Trained segmentation model
        data_loader: DataLoader containing images to analyze
        output_dir: Directory where to save visualizations
        device: Device to run inference on
        num_samples: Number of samples to visualize
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    # Create a custom colormap for the heatmap
    colors = [(0, 0, 0.7), (0, 0.7, 1), (0, 1, 0), (0.7, 1, 0), (1, 0.7, 0), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

    sample_count = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            if sample_count >= num_samples:
                break

            inputs = inputs.to(device)

            # Get model output
            outputs = model(inputs)
            # Handle different output formats
            if isinstance(outputs, torch.Tensor):
                logits = outputs
            elif isinstance(outputs, dict) and "out" in outputs:
                logits = outputs["out"]
            else:
                logits = outputs

            # Get predicted class labels
            preds = logits.argmax(dim=1)  # Shape: [B, H, W]

            # Get confidence scores (softmax probabilities)
            probs = torch.softmax(logits, dim=1)  # Shape: [B, C, H, W]

            # Get max probability for each pixel
            confidence, _ = probs.max(dim=1)  # Shape: [B, H, W]

            # Process each image in the batch
            for i in range(inputs.size(0)):
                if sample_count >= num_samples:
                    break

                # Convert tensors to numpy for visualization
                input_img = inputs[i].cpu().permute(1, 2, 0).numpy()
                pred_mask = preds[i].cpu().numpy()
                conf_map = confidence[i].cpu().numpy()

                # Normalize image for display
                input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())

                # Create figure with subplots
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))

                # Plot original image
                axs[0].imshow(input_img)
                axs[0].set_title('Original Image')
                axs[0].axis('off')

                # Plot segmentation mask
                axs[1].imshow(pred_mask, cmap='tab20', vmin=0, vmax=logits.size(1) - 1)
                axs[1].set_title('Segmentation Prediction')
                axs[1].axis('off')

                # Plot confidence heatmap
                im = axs[2].imshow(conf_map, cmap=cmap, vmin=0, vmax=1)
                axs[2].set_title('Confidence Map')
                axs[2].axis('off')

                # Add colorbar
                cbar = fig.colorbar(im, ax=axs[2], orientation='vertical', fraction=0.046, pad=0.04)
                cbar.set_label('Confidence Score')

                # Save figure
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'activation_map_{sample_count}.png'), dpi=200)
                plt.close(fig)

                sample_count += 1

    print(f"Generated {sample_count} class activation maps in {output_dir}")


def analyze_class_performance(model, data_loader, device, class_names=None):
    """
    Analyze and report per-class performance metrics

    Args:
        model: Trained segmentation model
        data_loader: DataLoader containing images to analyze
        device: Device to run inference on
        class_names: Optional dictionary mapping class IDs to names

    Returns:
        Dictionary of per-class metrics and problem classes
    """
    model.eval()

    # Initialize confusion matrix
    # Try to get number of classes from model
    try:
        # This might work for some models
        num_classes = next(iter(model.parameters())).size(0)
    except:
        # Fallback: run a sample through the model to determine output size
        sample_inputs, _ = next(iter(data_loader))
        sample_inputs = sample_inputs[:1].to(device)  # Take just one sample
        with torch.no_grad():
            outputs = model(sample_inputs)
            if isinstance(outputs, torch.Tensor):
                num_classes = outputs.size(1)
            elif isinstance(outputs, dict) and "out" in outputs:
                num_classes = outputs["out"].size(1)
            else:
                raise ValueError("Cannot determine number of classes from model output")

    hist = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            # Handle different output formats
            if isinstance(outputs, torch.Tensor):
                logits = outputs
            elif isinstance(outputs, dict) and "out" in outputs:
                logits = outputs["out"]
            else:
                logits = outputs

            preds = logits.argmax(dim=1)

            # Update confusion matrix
            for j in range(targets.size(0)):
                pred_flat = preds[j].view(-1)
                target_flat = targets[j].view(-1)
                hist += _fast_hist(pred_flat, target_flat, num_classes)

    # Compute per-class metrics
    metrics = compute_metrics(hist)

    # Find problematic classes (low IoU or F1)
    problem_classes = []
    class_metrics = {}

    for i in range(num_classes):
        # Skip classes not present in ground truth
        if hist.sum(dim=1)[i] == 0:
            continue

        class_name = class_names[i] if class_names and i in class_names else f"Class {i}"

        class_metrics[class_name] = {
            'iou': metrics['iou'][i].item(),
            'precision': metrics['precision'][i].item(),
            'recall': metrics['recall'][i].item(),
            'f1': metrics['f1_score'][i].item(),
            'dice': metrics['dice'][i].item(),
            'pixel_count': hist.sum(dim=1)[i].item(),
            'correct_pixels': hist[i, i].item()
        }

        # Identify problem classes (low IoU or high confusion)
        if metrics['iou'][i] < 0.5:
            # Find classes this class is most confused with
            confusion_with = []
            for j in range(num_classes):
                if i != j and hist[i, j] > 0:
                    confused_name = class_names[j] if class_names and j in class_names else f"Class {j}"
                    confusion_with.append({
                        'class': confused_name,
                        'count': hist[i, j].item(),
                        'percentage': (hist[i, j] / hist.sum(dim=1)[i]).item() * 100
                    })

            # Sort by confusion count (descending)
            confusion_with.sort(key=lambda x: x['count'], reverse=True)

            problem_classes.append({
                'class_name': class_name,
                'iou': metrics['iou'][i].item(),
                'confusion_with': confusion_with[:3]  # Top 3 confused classes
            })

    # Sort problem classes by IoU (ascending)
    problem_classes.sort(key=lambda x: x['iou'])

    # Return results
    return {
        'class_metrics': class_metrics,
        'problem_classes': problem_classes,
        'overall_metrics': {
            'mean_iou': metrics['mean_iou'],
            'pixel_acc': metrics['pixel_acc'],
            'mean_f1': metrics['mean_f1'],
            'mean_dice': metrics['mean_dice'],
            'boundary_f1': metrics['bf_score']
        }
    }


def generate_performance_report(model_info, metrics, class_performance, output_path):
    """
    Generate a comprehensive performance report for the model

    Args:
        model_info: Dictionary with model information
        metrics: Dictionary with overall metrics
        class_performance: Dictionary with class-wise performance
        output_path: Path to save the report
    """
    # Create the report
    with open(output_path, 'w') as f:
        # Header
        f.write("# Semantic Segmentation Model Performance Report\n\n")

        # Model information
        f.write("## Model Information\n\n")

        # Get architecture name, handling different naming conventions
        architecture = None
        if "network" in model_info:
            architecture = model_info["network"]
        elif "encoder" in model_info:
            architecture = f"DeepLabV3+ with {model_info['encoder']} encoder"

        f.write(f"- Architecture: {architecture or 'Unknown'}\n")
        f.write(f"- Input Resolution: {model_info.get('height', 1208)}x{model_info.get('width', 1920)}\n")
        f.write(f"- Number of Classes: {model_info.get('classes', 'Unknown')}\n")
        f.write(f"- Training Epochs: {model_info.get('epochs', 'Unknown')}\n")
        f.write(f"- Loss Function: {model_info.get('loss', 'CrossEntropy')}\n")

        # Optional settings that might not be in all model versions
        if "augment" in model_info:
            f.write(f"- Data Augmentation: {'Enabled' if model_info.get('augment') == 'True' else 'Disabled'}\n")
        if "handle_imbalance" in model_info:
            f.write(
                f"- Class Imbalance Handling: {'Enabled' if model_info.get('handle_imbalance') == 'True' else 'Disabled'}\n")

        f.write("\n")

        # Overall metrics
        f.write("## Overall Performance Metrics\n\n")
        f.write(f"- Mean IoU: {metrics.get('mean_iou', 0.0):.4f}\n")
        f.write(f"- Pixel Accuracy: {metrics.get('pixel_acc', 0.0):.4f}\n")
        f.write(f"- Mean F1 Score: {metrics.get('mean_f1', 0.0):.4f}\n")
        f.write(f"- Mean Dice Coefficient: {metrics.get('mean_dice', 0.0):.4f}\n")
        f.write(f"- Boundary F1 Score: {metrics.get('boundary_f1', 0.0):.4f}\n\n")

        # Class-wise performance
        f.write("## Class-wise Performance\n\n")
        f.write("| Class | IoU | Precision | Recall | F1 Score | Dice |\n")
        f.write("|-------|-----|-----------|--------|----------|------|\n")

        class_metrics = class_performance.get('class_metrics', {})
        for class_name, metrics in sorted(class_metrics.items(),
                                          key=lambda x: x[1]['iou'],
                                          reverse=True):
            f.write(f"| {class_name} | {metrics['iou']:.4f} | {metrics['precision']:.4f} | ")
            f.write(f"{metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['dice']:.4f} |\n")

        f.write("\n")

        # Problematic classes
        f.write("## Potentially Problematic Classes\n\n")
        problem_classes = class_performance.get('problem_classes', [])
        for i, problem in enumerate(problem_classes[:10]):  # Top 10 problems
            f.write(f"### {i + 1}. {problem['class_name']} (IoU: {problem['iou']:.4f})\n\n")
            f.write("Most confused with:\n")
            for confusion in problem['confusion_with']:
                f.write(f"- {confusion['class']}: {confusion['percentage']:.1f}%\n")
            f.write("\n")

        # Recommendations
        f.write("## Recommendations for Improvement\n\n")

        if problem_classes:
            f.write("Based on the analysis, consider the following improvements:\n\n")

            # General recommendations
            f.write("1. **Address Class Imbalance**: For classes with low IoU and low pixel count, consider:\n")
            f.write("   - Data augmentation focused on underrepresented classes\n")
            f.write("   - Class weighting in the loss function\n")
            f.write("   - Oversampling techniques\n\n")

            f.write("2. **Refine Boundary Detection**: If boundary F1 score is low:\n")
            f.write("   - Consider boundary-aware loss functions\n")
            f.write("   - Try higher resolution inputs for finer boundaries\n")
            f.write("   - Add boundary detection auxiliary task\n\n")

            f.write("3. **Targeted Augmentations**: Based on confusion patterns:\n")
            f.write("   - Increase contrast between commonly confused classes\n")
            f.write("   - Add more examples of confusing scenarios\n\n")

            # Specific recommendations for top problem classes
            worst_class = problem_classes[0]['class_name'] if problem_classes else "None"
            f.write(f"4. **Focus on '{worst_class}'**: This class has the lowest IoU. Consider:\n")
            f.write("   - Reviewing the annotation quality for this class\n")
            f.write("   - Adding more training examples\n")
            f.write("   - Special augmentations to highlight its distinctive features\n\n")
        else:
            f.write("The model is performing well across all classes. To further improve:\n\n")
            f.write(
                "1. **Fine-tune Hyperparameters**: Experiment with learning rate, batch size, and optimizer settings\n")
            f.write("2. **Try More Advanced Architectures**: Consider more recent segmentation models\n")
            f.write("3. **Ensemble Methods**: Combine predictions from multiple models\n\n")

        # Conclusion
        f.write("## Conclusion\n\n")
        f.write(f"The {architecture or 'segmentation'} model ")
        if metrics.get('mean_iou', 0) > 0.7:
            f.write("demonstrates strong performance across most classes. ")
        elif metrics.get('mean_iou', 0) > 0.5:
            f.write("shows reasonable performance, but has room for improvement. ")
        else:
            f.write("shows baseline functionality, but requires significant improvement. ")

        f.write("The analysis highlights specific classes that need attention, and the ")
        f.write("recommendations provide concrete steps to improve model performance in future iterations.\n")

    print(f"Performance report generated at {output_path}")


def load_color_map(class_list_path):
    """
    Load color map from class list JSON file, handling different formats

    Args:
        class_list_path: Path to class list JSON file

    Returns:
        Dictionary mapping class IDs to RGB colors
    """
    try:
        with open(class_list_path, 'r') as f:
            class_info = json.load(f)
    except Exception as e:
        print(f"Error loading class list: {e}")
        return None

    color_map = {}
    try:
        # Format 1: list of dictionaries with 'id' and 'color' fields
        if isinstance(class_info, list):
            for item in class_info:
                if isinstance(item, dict) and 'id' in item and 'color' in item:
                    color_map[item['id']] = item['color']

        # Format 2: dictionary with class names as keys
        elif isinstance(class_info, dict):
            for class_name, class_data in class_info.items():
                if isinstance(class_data, dict):
                    # Format: {"class_name": {"id": 1, "color": [r,g,b]}}
                    if 'id' in class_data and 'color' in class_data:
                        color_map[class_data['id']] = class_data['color']
                else:
                    # Format might be {"class_id": [r,g,b]}
                    try:
                        class_id = int(class_name)  # Try converting key to integer
                        if isinstance(class_data, list) and len(class_data) == 3:
                            color_map[class_id] = class_data
                    except (ValueError, TypeError):
                        pass  # Not a valid integer key
    except Exception as e:
        print(f"Warning: Error processing class list: {e}")
        print("Using default grayscale output instead of color mapping")
        return None

    # If no valid color map could be extracted
    if not color_map:
        print("No valid color map found in class list, using random colors")
        # Get number of classes from the data if possible
        num_classes = 0
        if isinstance(class_info, list):
            num_classes = len(class_info)
        elif isinstance(class_info, dict):
            num_classes = len(class_info)

        # Generate random colors for each class
        if num_classes > 0:
            for i in range(num_classes):
                color_map[i] = [random.randint(0, 255) for _ in range(3)]

    return color_map


# Custom loss functions
class FocalLoss(torch.nn.Module):
    """
    Focal Loss implementation for segmentation

    Args:
        gamma: Focusing parameter for hard examples (default: 2.0)
        alpha: Weighting factor (default: 0.25)
        weight: Class weight tensor for class imbalance (default: None)
    """

    def __init__(self, gamma=2.0, alpha=0.25, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        self.ce = torch.nn.CrossEntropyLoss(reduction='none', weight=weight)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class DiceLoss(torch.nn.Module):
    """
    Dice Loss implementation for segmentation

    Args:
        smooth: Smoothing factor for numerical stability (default: 1.0)
        weight: Class weight tensor (default: None)
    """

    def __init__(self, smooth=1.0, weight=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.weight = weight

    def forward(self, input, target):
        N, C = input.size(0), input.size(1)

        input_soft = torch.softmax(input, dim=1)

        # Create one-hot encoding for target
        target_one_hot = torch.zeros_like(input_soft)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)

        # Flatten all dimensions except batch
        input_flat = input_soft.view(N, C, -1)
        target_flat = target_one_hot.view(N, C, -1)

        intersection = (input_flat * target_flat).sum(dim=2)
        union = input_flat.sum(dim=2) + target_flat.sum(dim=2)

        # Apply class weights if provided
        if self.weight is not None:
            intersection = intersection * self.weight.view(1, -1)
            union = union * self.weight.view(1, -1)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice.mean()
        return loss


class CombinedLoss(torch.nn.Module):
    """
    Combined loss for segmentation, using a combination of Dice Loss and Focal Loss

    Args:
        dice_weight: Weight for Dice Loss component (default: 0.5)
        ce_weight: Weight for Focal Loss component (default: 0.5)
        gamma: Focusing parameter for Focal Loss (default: 2.0)
        alpha: Weighting factor for Focal Loss (default: 0.25)
        weight: Class weight tensor (default: None)
    """

    def __init__(self, dice_weight=0.5, ce_weight=0.5, gamma=2.0, alpha=0.25, weight=None):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(weight=weight)
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha, weight=weight)

    def forward(self, input, target):
        return self.dice_weight * self.dice_loss(input, target) + \
            self.ce_weight * self.focal_loss(input, target)