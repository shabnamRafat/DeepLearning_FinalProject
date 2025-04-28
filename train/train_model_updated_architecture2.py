import argparse
import ast
import os
import sys
import time
import warnings
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode
import segmentation_models_pytorch as smp

import torch
import gc

# Clear memory before starting
torch.cuda.empty_cache()
gc.collect()

# Enable memory-saving features
torch.backends.cudnn.benchmark = True

# allow importing your custom dataset from ../dataset-loading
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
utils_dir = os.path.join(parent_dir, "dataset-loading")
sys.path.insert(0, utils_dir)

from Data_Preprocessing import A2D2_CSV_dataset  # your CSV‐based Dataset

warnings.filterwarnings("ignore")


def fast_hist(pred, label, num_classes):
    """Confusion‐matrix histogram for one flattened batch."""
    mask = (label >= 0) & (label < num_classes)
    hist = torch.bincount(
        num_classes * label[mask] + pred[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist


def compute_metrics(hist):
    """Compute per-class IoU, mean IoU, and pixel accuracy from hist."""
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


def create_deeplabv3plus_model(num_classes, encoder_name="resnet50", encoder_weights="imagenet", activation=None):
    """
    Creates a DeepLabV3+ model using the segmentation-models-pytorch library.

    Args:
        num_classes: Number of output classes
        encoder_name: Name of the encoder backbone (default: "resnet50")
        encoder_weights: Pre-trained weights to use (default: "imagenet")
        activation: Activation function to use (default: None)

    Returns:
        A DeepLabV3+ model
    """
    try:
        # Try to load the model with specified parameters
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=activation
        )
        print(f"Successfully loaded DeepLabV3+ with {encoder_name} backbone using {encoder_weights} weights")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to default model")
        # Fallback to standard model with ResNet-50 backbone
        return smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            classes=num_classes,
            activation=activation
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model & training parameters
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1e6)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_warmup_ratio", type=float, default=1)
    parser.add_argument("--epoch_peak", type=int, default=2)
    parser.add_argument("--lr_decay_per_epoch", type=float, default=1)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--classes", type=int, default=55)
    parser.add_argument("--log-freq", type=int, default=1)
    parser.add_argument("--eval-size", type=int, default=30)
    parser.add_argument("--height", type=int, default=1208)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument(
        "--encoder", type=str,
        default="resnet50",
        help="Encoder backbone for segmentation model (resnet50, xception, etc.)"
    )
    parser.add_argument(
        "--encoder-weights", type=str,
        default="imagenet",
        choices=["imagenet", "imagenet+coco", "imagenet+pascalvoc", None],
        help="Pre-trained weights for encoder"
    )

    # Add validation split parameters
    parser.add_argument("--train-split", type=float, default=0.7,
                        help="Proportion of data to use for training")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Proportion of data to use for validation")
    parser.add_argument("--test-split", type=float, default=0.15,
                        help="Proportion of data to use for testing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for dataset splitting")

    # infra configuration
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--amp", type=str, default="True")

    # Loss function
    parser.add_argument("--loss", type=str, default="ce",
                        choices=["ce", "focal", "dice", "combined", "lovasz"],
                        help="Loss function for training")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Gamma parameter for focal loss")
    parser.add_argument("--focal-alpha", type=float, default=0.25,
                        help="Alpha parameter for focal loss")
    parser.add_argument("--dice-weight", type=float, default=0.5,
                        help="Weight for dice loss in combined loss")
    parser.add_argument("--ce-weight", type=float, default=0.5,
                        help="Weight for CE loss in combined loss")

    # Optimizer settings
    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["sgd", "adam", "adamw"],
                        help="Optimizer to use for training")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay for optimizer")

    # Data, model, and output directories
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument(
        "--class-list", type=str,
        default="class_list.json",
        help="Path to JSON file mapping colors→class IDs"
    )
    parser.add_argument(
        "--pairs-csv", type=str, required=True,
        help="Path to CSV listing image,mask pairs"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Local directory to save model checkpoints"
    )

    # Add augmentation options
    parser.add_argument("--augment", type=str, default="False",
                        help="Whether to use data augmentation")
    parser.add_argument("--mixup", type=str, default="False",
                        help="Whether to use mixup augmentation")
    parser.add_argument("--cutmix", type=str, default="False",
                        help="Whether to use cutmix augmentation")

    # Early stopping
    parser.add_argument("--early-stopping", type=str, default="False",
                        help="Whether to use early stopping")
    parser.add_argument("--patience", type=int, default=5,
                        help="Patience for early stopping")

    args, _ = parser.parse_known_args()

    # ensure checkpoint folder exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.cuda.empty_cache()

    # Transforms
    image_transform = Resize(
        (args.height, args.width),
        interpolation=InterpolationMode.BILINEAR,
    )
    target_transform = Resize(
        (args.height, args.width),
        interpolation=InterpolationMode.NEAREST,
    )

    # Load the full dataset first
    full_dataset = A2D2_CSV_dataset(
        csv_file=args.pairs_csv,
        class_list=args.class_list,
        cache=args.cache,
        height=args.height,
        width=args.width,
        transform=image_transform,
        target_transform=target_transform,
    )

    # Split the dataset into training, validation, and test sets
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))

    # Calculate split sizes
    train_size = int(args.train_split * dataset_size)
    val_size = int(args.val_split * dataset_size)
    test_size = dataset_size - train_size - val_size

    # First split into training and temp (val+test)
    temp_indices = indices.copy()
    np.random.seed(args.seed)
    np.random.shuffle(temp_indices)
    train_indices = temp_indices[:train_size]
    temp_indices = temp_indices[train_size:]  # Remaining indices

    # Now split temp indices into validation and test
    val_indices = temp_indices[:val_size]
    test_indices = temp_indices[val_size:val_size + test_size]

    # Create dataset subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    print(
        f"Dataset split: {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True, prefetch_factor=args.prefetch,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        drop_last=True, prefetch_factor=args.prefetch,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        drop_last=False, prefetch_factor=args.prefetch,
        persistent_workers=True,
    )


    # Define custom loss functions
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0, alpha=0.25):
            super(FocalLoss, self).__init__()
            self.gamma = gamma
            self.alpha = alpha
            self.ce = nn.CrossEntropyLoss(reduction='none')

        def forward(self, input, target):
            logp = self.ce(input, target)
            p = torch.exp(-logp)
            loss = (1 - p) ** self.gamma * logp
            return loss.mean()


    class DiceLoss(nn.Module):
        def __init__(self, smooth=1.0):
            super(DiceLoss, self).__init__()
            self.smooth = smooth

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

            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            loss = 1 - dice.mean()
            return loss


    class CombinedLoss(nn.Module):
        def __init__(self, dice_weight=0.5, ce_weight=0.5, gamma=2.0, alpha=0.25):
            super(CombinedLoss, self).__init__()
            self.dice_weight = dice_weight
            self.ce_weight = ce_weight
            self.dice_loss = DiceLoss()
            self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)

        def forward(self, input, target):
            return self.dice_weight * self.dice_loss(input, target) + \
                self.ce_weight * self.focal_loss(input, target)


    # Add Lovasz loss for better boundary segmentation
    class LovaszLoss(nn.Module):
        def __init__(self):
            super(LovaszLoss, self).__init__()

        def forward(self, input, target):
            # Import inside method to avoid dependency for those who don't use it
            try:
                from pytorch_toolbelt.losses import LovaszLossSoftmax
                lovasz = LovaszLossSoftmax()
                return lovasz(input, target)
            except ImportError:
                print("Warning: pytorch_toolbelt not installed, falling back to CE loss")
                return nn.CrossEntropyLoss()(input, target)


    # Model, loss, optimizer, AMP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create DeepLabV3+ model using the segmentation-models-pytorch library
    model = create_deeplabv3plus_model(
        num_classes=args.classes,
        encoder_name=args.encoder,  # Use encoder from args (default: resnet50)
        encoder_weights=args.encoder_weights,  # Use weights from args (default: imagenet)
        activation=None
    )
    model.to(device)

    # Initialize loss function based on argument
    if args.loss == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    elif args.loss == "dice":
        criterion = DiceLoss()
    elif args.loss == "combined":
        criterion = CombinedLoss(
            dice_weight=args.dice_weight,
            ce_weight=args.ce_weight,
            gamma=args.focal_gamma,
            alpha=args.focal_alpha
        )
    elif args.loss == "lovasz":
        criterion = LovaszLoss()
    else:
        criterion = nn.CrossEntropyLoss()  # Default

    # Initialize optimizer based on argument
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )

    scaler = GradScaler(enabled=ast.literal_eval(args.amp))

    # Initialize metric tracking
    best_val_miou = 0.0
    patience_counter = 0
    train_metrics_history = []
    val_metrics_history = []

    # Training loop
    for epoch in range(args.epochs):
        # LR schedule
        if epoch <= args.epoch_peak:
            start_lr = args.lr * args.lr_warmup_ratio
            lr = start_lr + (epoch / args.epoch_peak) * (args.lr - start_lr)
        else:
            lr = args.lr * (args.lr_decay_per_epoch ** (epoch - args.epoch_peak))
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        print(f"In epoch {epoch} learning rate: {lr:.6e}")

        # Training phase
        model.train()
        train_losses = []
        train_hist = torch.zeros(args.classes, args.classes, dtype=torch.int64, device=device)

        epoch_start = time.time()
        for i, (inputs, masks) in enumerate(train_loader):
            if i > args.iterations:
                break

            inputs, masks = inputs.to(device), masks.to(device)
            optimizer.zero_grad()

            with autocast(enabled=ast.literal_eval(args.amp)):
                outputs = model(inputs)
                # Handle different output formats from segmentation-models-pytorch
                if isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    # Handle dictionary output (depends on SMP version)
                    logits = outputs.get("out", outputs)

                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())

            # Update histogram for metrics
            preds = logits.argmax(dim=1).view(-1)
            gts = masks.view(-1)
            train_hist += fast_hist(preds, gts, args.classes)

            # logging & validation
            if i > 0 and (i % args.log_freq == 0):
                # Calculate training metrics so far
                train_metrics = compute_metrics(train_hist)
                avg_train_loss = np.mean(train_losses)

                # Run validation
                val_losses = []
                val_hist = torch.zeros(args.classes, args.classes, dtype=torch.int64, device=device)

                model.eval()
                with torch.no_grad():
                    for j, (v_in, v_mask) in enumerate(val_loader):
                        v_in, v_mask = v_in.to(device), v_mask.to(device)
                        out = model(v_in)

                        # Handle different output formats
                        if isinstance(out, torch.Tensor):
                            val_logits = out
                        else:
                            val_logits = out.get("out", out)

                        val_loss = criterion(val_logits, v_mask)
                        val_losses.append(val_loss.item())

                        preds = val_logits.argmax(dim=1).view(-1)
                        gts = v_mask.view(-1)
                        val_hist += fast_hist(preds, gts, args.classes)

                        if j * args.batch >= args.eval_size:
                            break

                # compute validation metrics
                val_metrics = compute_metrics(val_hist)
                avg_val_loss = np.mean(val_losses)

                # Print progress
                current_time = time.time()
                samples_per_sec = (i * args.batch) / (current_time - epoch_start)

                print(f"Epoch {epoch} - Batch {i}/{len(train_loader)}")
                print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                print(f"Train mIoU: {train_metrics['mean_iou']:.4f}, Val mIoU: {val_metrics['mean_iou']:.4f}")
                print(f"Train F1: {train_metrics['mean_f1']:.4f}, Val F1: {val_metrics['mean_f1']:.4f}")
                print(f"Train Dice: {train_metrics['mean_dice']:.4f}, Val Dice: {val_metrics['mean_dice']:.4f}")
                print(f"Boundary F1: {val_metrics['bf_score']:.4f}, Samples/sec: {samples_per_sec:.2f}")

                # Save best model
                if val_metrics['mean_iou'] > best_val_miou:
                    best_val_miou = val_metrics['mean_iou']
                    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_miou': best_val_miou,
                        'args': vars(args)
                    }, best_model_path)
                    print(f"New best model saved with mIoU: {best_val_miou:.4f}")
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Early stopping
                if ast.literal_eval(args.early_stopping) and patience_counter >= args.patience:
                    print(f"Early stopping after {patience_counter} iterations without improvement")
                    break

                # Return to training mode
                model.train()

        # End of epoch
        epoch_end = time.time()
        print(f"Epoch {epoch} completed in {epoch_end - epoch_start:.2f} seconds")

        # Check for early stopping at epoch level
        if ast.literal_eval(args.early_stopping) and patience_counter >= args.patience:
            print(f"Early stopping after {epoch + 1} epochs without improvement")
            break

        # Calculate full training and validation metrics for the epoch
        model.eval()

        # Full validation pass
        val_losses = []
        val_hist = torch.zeros(args.classes, args.classes, dtype=torch.int64, device=device)

        with torch.no_grad():
            for v_in, v_mask in val_loader:
                v_in, v_mask = v_in.to(device), v_mask.to(device)
                out = model(v_in)

                # Handle different output formats
                if isinstance(out, torch.Tensor):
                    val_logits = out
                else:
                    val_logits = out.get("out", out)

                val_loss = criterion(val_logits, v_mask)
                val_losses.append(val_loss.item())

                preds = val_logits.argmax(dim=1).view(-1)
                gts = v_mask.view(-1)
                val_hist += fast_hist(preds, gts, args.classes)

        # Compute final epoch metrics
        epoch_train_metrics = compute_metrics(train_hist)
        epoch_val_metrics = compute_metrics(val_hist)

        # Save metrics history
        train_metrics_history.append({
            'epoch': epoch,
            'loss': np.mean(train_losses),
            **{k: v if isinstance(v, (int, float)) else v.tolist()
               for k, v in epoch_train_metrics.items()}
        })

        val_metrics_history.append({
            'epoch': epoch,
            'loss': np.mean(val_losses),
            **{k: v if isinstance(v, (int, float)) else v.tolist()
               for k, v in epoch_val_metrics.items()}
        })

        # Save metrics to JSON file
        metrics_path = os.path.join(args.checkpoint_dir, f"metrics_ep{epoch}.json")
        with open(metrics_path, 'w') as f:
            json.dump({
                'train': train_metrics_history[-1],
                'val': val_metrics_history[-1],
            }, f, indent=2)

        # Save checkpoint after each epoch
        ckpt = os.path.join(args.checkpoint_dir, f"model-ep{epoch}.pth")
        if os.path.exists(ckpt):
            os.remove(ckpt)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': epoch_train_metrics,
            'val_metrics': epoch_val_metrics,
            'args': vars(args)
        }, ckpt)

        # update "latest.pth"
        latest = os.path.join(args.checkpoint_dir, "latest.pth")
        if os.path.exists(latest):
            os.remove(latest)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': vars(args)
        }, latest)

    # Final evaluation on test set
    print("Performing final evaluation on test set...")
    model.eval()
    test_hist = torch.zeros(args.classes, args.classes, dtype=torch.int64, device=device)
    test_losses = []

    with torch.no_grad():
        for inputs, masks in test_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)

            # Handle different output formats
            if isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                logits = outputs.get("out", outputs)

            test_loss = criterion(logits, masks)
            test_losses.append(test_loss.item())

            preds = logits.argmax(dim=1).view(-1)
            gts = masks.view(-1)
            test_hist += fast_hist(preds, gts, args.classes)

    test_metrics = compute_metrics(test_hist)
    avg_test_loss = np.mean(test_losses)

    print("\nFinal Test Results:")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test mIoU: {test_metrics['mean_iou']:.4f}")
    print(f"Test Pixel Accuracy: {test_metrics['pixel_acc']:.4f}")
    print(f"Test Mean F1: {test_metrics['mean_f1']:.4f}")
    print(f"Test Mean Dice: {test_metrics['mean_dice']:.4f}")
    print(f"Test Boundary F1: {test_metrics['bf_score']:.4f}")

    # Save test results
    test_results_path = os.path.join(args.checkpoint_dir, "test_results.json")
    with open(test_results_path, 'w') as f:
        json.dump({
            'loss': avg_test_loss,
            **{k: v if isinstance(v, (int, float)) else v.tolist()
               for k, v in test_metrics.items()}
        }, f, indent=2)

    # Save the final model
    final_model_path = os.path.join(args.checkpoint_dir, "final_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_metrics': test_metrics,
        'args': vars(args)
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")


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
    import numpy as np
    from PIL import Image
    import random

    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    all_metrics = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            # Handle different output formats
            if isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                logits = outputs.get("out", outputs)

            preds = logits.argmax(dim=1)

            # Compute metrics for this batch
            batch_hist = torch.zeros(logits.size(1), logits.size(1), dtype=torch.int64, device=device)
            for j in range(preds.size(0)):
                pred_flat = preds[j].view(-1)
                target_flat = targets[j].view(-1)
                batch_hist += fast_hist(pred_flat, target_flat, logits.size(1))

            batch_metrics = compute_metrics(batch_hist)

            # Convert tensors to CPU/numpy for saving results
            preds_cpu = preds.cpu().numpy()
            targets_cpu = targets.cpu().numpy()

            # Save each prediction in the batch
            for j, pred in enumerate(preds_cpu):
                img_metrics = {}

                # Calculate per-image metrics
                if save_metrics:
                    img_hist = fast_hist(
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


# Example usage after training
if __name__ == "__main__" and 'model' in locals():
    # Load color map from class list
    with open(args.class_list, 'r') as f:
        class_info = json.load(f)

    # Convert class info to color map (adjust based on your class_list.json format)
    color_map = {}
    for class_name, class_data in class_info.items():
        try:
            class_id = class_data.get('id')
            color = class_data.get('color')
            if class_id is not None and color is not None:
                color_map[class_id] = color
        except (TypeError, AttributeError):
            # Handle different JSON formats
            if isinstance(class_data, int):
                # If class_data is the ID directly
                class_id = class_data
                # You might need to generate a color or have a separate color mapping
                color_map[class_id] = [random.randint(0, 255) for _ in range(3)]

    # Create output directories for segmentation results
    train_output_dir = os.path.join(args.checkpoint_dir, "train_segmentation")
    val_output_dir = os.path.join(args.checkpoint_dir, "val_segmentation")
    test_output_dir = os.path.join(args.checkpoint_dir, "test_segmentation")

    # Run inference and save results for each split
    print("Generating validation segmentation outputs...")
    save_segmentation_results(
        model,
        val_loader,
        val_output_dir,
        device,
        color_map
    )

    print("Generating test segmentation outputs...")
    save_segmentation_results(
        model,
        test_loader,
        test_output_dir,
        device,
        color_map,
        save_metrics=True
    )

    print(f"All segmentation results saved to {args.checkpoint_dir}")


# Visualization function to create class activation maps
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
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    # Create a custom colormap for the heatmap
    colors = [(0, 0, .7), (0, .7, 1), (0, 1, 0), (.7, 1, 0), (1, .7, 0), (1, 0, 0)]
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
            else:
                logits = outputs.get("out", outputs)

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


# Example usage after training
if __name__ == "__main__" and 'model' in locals():
    # Create output directory for activation maps
    activation_maps_dir = os.path.join(args.checkpoint_dir, "activation_maps")

    # Generate activation maps
    print("Generating class activation maps...")
    generate_class_activation_maps(
        model,
        test_loader,
        activation_maps_dir,
        device,
        num_samples=10
    )


# Class-wise performance analysis
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
    num_classes = next(iter(model.parameters())).size(0)  # Get number of classes from output layer
    hist = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Handle different output formats
            if isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                logits = outputs.get("out", outputs)

            preds = logits.argmax(dim=1)

            # Update confusion matrix
            for j in range(targets.size(0)):
                pred_flat = preds[j].view(-1)
                target_flat = targets[j].view(-1)
                hist += fast_hist(pred_flat, target_flat, num_classes)

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


# Example usage after training
if __name__ == "__main__" and 'model' in locals():
    # Get class names if available
    class_names = {}
    try:
        with open(args.class_list, 'r') as f:
            class_data = json.load(f)

        # Adapt to your class_list.json format
        for name, data in class_data.items():
            if isinstance(data, dict) and 'id' in data:
                class_names[data['id']] = name
            elif isinstance(data, int):
                class_names[data] = name
    except Exception as e:
        print(f"Could not load class names: {e}")
        class_names = None

    # Analyze class performance
    print("Analyzing class performance...")
    performance_results = analyze_class_performance(
        model,
        test_loader,
        device,
        class_names
    )

    # Save results
    perf_output_path = os.path.join(args.checkpoint_dir, "class_performance.json")
    with open(perf_output_path, 'w') as f:
        json.dump(performance_results, f, indent=2)

    # Print problematic classes
    print("\nPotentially problematic classes:")
    for problem in performance_results['problem_classes'][:5]:  # Show top 5 problems
        print(f"- {problem['class_name']}: IoU = {problem['iou']:.4f}")
        print("  Confused with:")
        for confusion in problem['confusion_with']:
            print(f"  - {confusion['class']}: {confusion['percentage']:.1f}%")

    print(f"\nDetailed performance analysis saved to {perf_output_path}")


# Optional: Add a function to generate a model performance report
def generate_performance_report(model_info, metrics, class_performance, output_path):
    """
    Generate a comprehensive performance report for the model

    Args:
        model_info: Dictionary with model information
        metrics: Dictionary with overall metrics
        class_performance: Dictionary with class-wise performance
        output_path: Path to save the report
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import random

    # Create the report
    with open(output_path, 'w') as f:
        # Header
        f.write("# Semantic Segmentation Model Performance Report\n\n")

        # Model information
        f.write("## Model Information\n\n")
        f.write(f"- Architecture: DeepLabV3+\n")
        f.write(f"- Backbone: {model_info.get('encoder', 'ResNet-50')}\n")
        f.write(f"- Input Resolution: {model_info.get('height', 1208)}x{model_info.get('width', 1920)}\n")
        f.write(f"- Number of Classes: {model_info.get('classes', 'Unknown')}\n")
        f.write(f"- Training Epochs: {model_info.get('epochs', 'Unknown')}\n")
        f.write(f"- Loss Function: {model_info.get('loss', 'CrossEntropy')}\n\n")

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
        f.write("The DeepLabV3+ model ")
        if metrics.get('mean_iou', 0) > 0.7:
            f.write("demonstrates strong performance across most classes. ")
        elif metrics.get('mean_iou', 0) > 0.5:
            f.write("shows reasonable performance, but has room for improvement. ")
        else:
            f.write("shows baseline functionality, but requires significant improvement. ")

        f.write("The analysis highlights specific classes that need attention, and the ")
        f.write("recommendations provide concrete steps to improve model performance in future iterations.\n")

    print(f"Performance report generated at {output_path}")


# Example usage after training
if __name__ == "__main__" and 'model' in locals() and 'performance_results' in locals():
    # Generate performance report
    report_path = os.path.join(args.checkpoint_dir, "performance_report.md")

    # Combine all information for the report
    model_info = vars(args)
    overall_metrics = performance_results['overall_metrics']

    generate_performance_report(
        model_info,
        overall_metrics,
        performance_results,
        report_path
    )