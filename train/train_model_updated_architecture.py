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

# Import custom utility functions
# Add the utils directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
utils_dir = os.path.join(parent_dir, "utils")
sys.path.append(utils_dir)

# Import utility functions
from utils_1 import (
    _fast_hist,
    compute_metrics,
    save_segmentation_results,
    generate_class_activation_maps,
    analyze_class_performance,
    generate_performance_report,
    load_color_map,
    FocalLoss,
    DiceLoss,
    CombinedLoss
)

import torch
import gc

# Clear memory before starting
torch.cuda.empty_cache()
gc.collect()

# Enable memory-saving features
torch.backends.cudnn.benchmark = True

# allow importing your custom dataset from ../dataset-loading
dataset_dir = os.path.join(parent_dir, "dataset-loading")
sys.path.insert(0, dataset_dir)

from Data_Preprocessing import A2D2_CSV_dataset

warnings.filterwarnings("ignore")


def create_deeplabv3plus_resnet50(num_classes):
    """Creates a DeepLabV3 model with ResNet-50 backbone."""
    try:
        model = torch.hub.load(
            "pytorch/vision",
            "deeplabv3_resnet50",
            pretrained=False,
            num_classes=num_classes
        )
        print("Successfully loaded DeepLabV3 with ResNet-50 backbone")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to default model")
        return torch.hub.load(
            "pytorch/vision:v0.9.1",
            "deeplabv3_mobilenet_v3_large",
            pretrained=False,
            num_classes=num_classes
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
        "--network", type=str,
        default="deeplabv3plus_resnet50",
        help="Segmentation model architecture to use"
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
                        choices=["ce", "focal", "dice", "combined"],
                        help="Loss function for training")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Gamma parameter for focal loss")
    parser.add_argument("--focal-alpha", type=float, default=0.25,
                        help="Alpha parameter for focal loss")

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

    # Model, loss, optimizer, AMP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model based on architecture choice
    if args.network == "deeplabv3plus_resnet50":
        model = create_deeplabv3plus_resnet50(args.classes)
    else:
        # Fallback to standard torchvision models
        model = torch.hub.load(
            "pytorch/vision:v0.9.1",
            args.network,
            pretrained=False,
            num_classes=args.classes
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
        criterion = CombinedLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    else:
        criterion = nn.CrossEntropyLoss()  # Default

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scaler = GradScaler(enabled=ast.literal_eval(args.amp))

    # Initialize metric tracking
    best_val_miou = 0.0
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
                outputs = model(inputs)["out"]
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())

            # Update histogram for metrics
            preds = outputs.argmax(dim=1).view(-1)
            gts = masks.view(-1)
            train_hist += _fast_hist(preds, gts, args.classes)

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
                        out = model(v_in)["out"]
                        val_loss = criterion(out, v_mask)
                        val_losses.append(val_loss.item())

                        preds = out.argmax(dim=1).view(-1)
                        gts = v_mask.view(-1)
                        val_hist += _fast_hist(preds, gts, args.classes)

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

                # Return to training mode
                model.train()

        # End of epoch
        epoch_end = time.time()
        print(f"Epoch {epoch} completed in {epoch_end - epoch_start:.2f} seconds")

        # Calculate full training and validation metrics for the epoch
        model.eval()

        # Full validation pass
        val_losses = []
        val_hist = torch.zeros(args.classes, args.classes, dtype=torch.int64, device=device)

        with torch.no_grad():
            for v_in, v_mask in val_loader:
                v_in, v_mask = v_in.to(device), v_mask.to(device)
                out = model(v_in)["out"]
                val_loss = criterion(out, v_mask)
                val_losses.append(val_loss.item())

                preds = out.argmax(dim=1).view(-1)
                gts = v_mask.view(-1)
                val_hist += _fast_hist(preds, gts, args.classes)

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
            outputs = model(inputs)["out"]
            test_loss = criterion(outputs, masks)
            test_losses.append(test_loss.item())

            preds = outputs.argmax(dim=1).view(-1)
            gts = masks.view(-1)
            test_hist += _fast_hist(preds, gts, args.classes)

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

    # Load color map
    color_map = load_color_map(args.class_list)

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

    # Generate activation maps
    activation_maps_dir = os.path.join(args.checkpoint_dir, "activation_maps")
    generate_class_activation_maps(
        model,
        test_loader,
        activation_maps_dir,
        device,
        num_samples=10
    )

    # Analyze class performance
    print("Analyzing class performance...")
    performance_results = analyze_class_performance(
        model,
        test_loader,
        device,
        class_names
    )

    # Save performance report
    report_path = os.path.join(args.checkpoint_dir, "performance_report.md")
    generate_performance_report(
        vars(args),
        performance_results['overall_metrics'],
        performance_results,
        report_path
    )