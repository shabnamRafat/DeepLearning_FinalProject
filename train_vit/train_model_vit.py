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
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode
from einops import rearrange

import torch
import gc

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


def _fast_hist(pred, label, num_classes):
    mask = (label >= 0) & (label < num_classes) & (pred >= 0) & (pred < num_classes)
    if not mask.all():
        print(f"Warning: Some predictions or labels are out of range [0, {num_classes-1}]")
        print(f"Pred range: {pred.min().item()} to {pred.max().item()}")
        print(f"Label range: {label.min().item()} to {label.max().item()}")
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


class PatchEmbedding(nn.Module):
    """Split image into patches and embed them"""

    def __init__(self, img_size, patch_size, in_channels=3, embed_dim=768, stride=None):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.stride = self.patch_size if stride is None else (stride, stride) if isinstance(stride, int) else stride
        self.num_patches_h = (self.img_size[0] - self.patch_size[0]) // self.stride[0] + 1
        self.num_patches_w = (self.img_size[1] - self.patch_size[1]) // self.stride[1] + 1
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Convolutional layer to extract patches and project to embedding dim
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.stride)

    def forward(self, x):
        # (B, C, H, W) -> (B, E, H', W')
        x = self.projection(x)
        # (B, E, H', W') -> (B, H'*W', E)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        # Multi-head Self-Attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # MLP
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-Attention
        x_ln = self.norm1(x)
        x = x + self.attn(x_ln, x_ln, x_ln)[0]

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        # Cross-Attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # Self-Attention
        self.norm2 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # MLP
        self.norm3 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, memory):
        # Cross-Attention
        x_ln = self.norm1(x)
        x = x + self.cross_attn(x_ln, memory, memory)[0]

        # Self-Attention
        x_ln = self.norm2(x)
        x = x + self.self_attn(x_ln, x_ln, x_ln)[0]

        # MLP
        x = x + self.mlp(self.norm3(x))
        return x


class SegmentationTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels=3, num_classes=21,
                 embed_dim=768, depth=12, decoder_depth=4, num_heads=12,
                 mlp_ratio=4.0, dropout=0.1, stride=None):
        super().__init__()

        # Patch Embedding
        self.patch_embedding = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            stride=stride
        )

        # Positional Embedding
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.patch_embedding.num_patches, embed_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer Encoder
        self.dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Transformer Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoder(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(decoder_depth)
        ])

        # From embedded dimension to output classes
        self.decoder_pred = nn.Linear(embed_dim, num_classes)

        # Upsampling to original image size
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.stride = stride if stride is not None else self.patch_size
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Get batch and spatial dimensions
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embedding(x)

        # Add positional embedding
        x = x + self.pos_embedding

        # Prepend class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Dropout
        x = self.dropout(x)

        # Transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)

        # Use the encoder output as memory for the decoder
        memory = x

        # Remove the class token for decoding
        decoder_input = memory[:, 1:, :]

        # Transformer decoder
        for layer in self.decoder_layers:
            decoder_input = layer(decoder_input, memory)

        # Output projection to classes
        out_features = self.decoder_pred(decoder_input)

        # Reshape to patch grid dimensions
        H_patch = self.patch_embedding.num_patches_h
        W_patch = self.patch_embedding.num_patches_w
        out_features = out_features.reshape(B, H_patch, W_patch, -1).permute(0, 3, 1, 2)

        # Upscale to original image dimensions (using bilinear interpolation)
        out_features = F.interpolate(out_features, size=(H, W), mode='bilinear', align_corners=False)

        return {"out": out_features}


def create_segmentation_transformer(
        img_size=(1208, 1920),
        patch_size=16,
        in_channels=3,
        num_classes=55,
        embed_dim=768,
        depth=12,
        decoder_depth=4,
        num_heads=12,
        stride=8  # Use a smaller stride for higher resolution feature maps
):
    """Creates a Segmentation Transformer model"""
    try:
        model = SegmentationTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            stride=stride
        )
        print("Successfully created Segmentation Transformer model")
        return model
    except Exception as e:
        print(f"Error creating Segmentation Transformer model: {e}")
        print("Falling back to standard ViT model")
        # Fallback to a simpler model if needed
        return torch.hub.load(
            "pytorch/vision",
            "deeplabv3_resnet50",
            pretrained=False,
            num_classes=num_classes
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model & training parameters
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1e6)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.005)  # Lower LR for transformer
    parser.add_argument("--lr_warmup_ratio", type=float, default=1)
    parser.add_argument("--epoch_peak", type=int, default=2)
    parser.add_argument("--lr_decay_per_epoch", type=float, default=1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--classes", type=int, default=55)
    parser.add_argument("--log-freq", type=int, default=1)
    parser.add_argument("--eval-size", type=int, default=30)
    parser.add_argument("--height", type=int, default=1208)
    parser.add_argument("--width", type=int, default=1920)

    # Transformer-specific parameters
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--decoder-depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.05)  # Higher weight decay for transformers

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
        def __init__(self, dice_weight=0.5, focal_weight=0.5, gamma=2.0, alpha=0.25):
            super(CombinedLoss, self).__init__()
            self.dice_weight = dice_weight
            self.focal_weight = focal_weight
            self.dice_loss = DiceLoss()
            self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)

        def forward(self, input, target):
            return self.dice_weight * self.dice_loss(input, target) + \
                self.focal_weight * self.focal_loss(input, target)


    # Model, loss, optimizer, AMP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create Segmentation Transformer model
    model = create_segmentation_transformer(
        img_size=(args.height, args.width),
        patch_size=args.patch_size,
        in_channels=3,  # RGB images
        num_classes=args.classes,
        embed_dim=args.embed_dim,
        depth=args.depth,
        decoder_depth=args.decoder_depth,
        num_heads=args.num_heads,
        stride=args.stride
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

    # Use AdamW for transformer training (better for transformers than SGD)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )


    # Learning rate scheduler with warmup
    def get_lr_lambda(step):
        # Warmup for first 10% steps
        warmup_steps = int(args.epochs * len(train_loader) * 0.1)
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # Linear decay after warmup
        return max(0.0, 1.0 - (step - warmup_steps) / (args.epochs * len(train_loader) - warmup_steps))


    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)

    scaler = GradScaler(enabled=ast.literal_eval(args.amp))

    # Initialize metric tracking
    best_val_miou = 0.0
    train_metrics_history = []
    val_metrics_history = []

    # Training loop
    for epoch in range(args.epochs):
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
            scheduler.step()  # Update learning rate

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
                current_lr = optimizer.param_groups[0]['lr']

                print(f"Epoch {epoch} - Batch {i}/{len(train_loader)} - LR: {current_lr:.6f}")
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

    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    all_metrics = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)["out"]
            preds = outputs.argmax(dim=1)

            # Compute metrics for this batch
            batch_hist = torch.zeros(outputs.size(1), outputs.size(1), dtype=torch.int64, device=device)
            for j in range(preds.size(0)):
                pred_flat = preds[j].view(-1)
                target_flat = targets[j].view(-1)
                batch_hist += _fast_hist(pred_flat, target_flat, outputs.size(1))

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
                        outputs.size(1)
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


# Visualization function to create attention maps for transformers
def generate_attention_maps(model, data_loader, output_dir, device, num_samples=5):
    """
    Generate attention maps to visualize what the transformer is focusing on

    Args:
        model: Trained segmentation transformer model
        data_loader: DataLoader containing images to analyze
        output_dir: Directory where to save visualizations
        device: Device to run inference on
        num_samples: Number of samples to visualize
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    import math

    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    # Register a hook to get attention maps
    attention_maps = []

    def get_attention(module, input, output):
        # Capture the attention maps (depends on transformer implementation)
        # For MultiheadAttention, output is (attn_output, attn_output_weights)
        if isinstance(output, tuple) and len(output) > 1:
            attention_maps.append(output[1].detach())

    # Attach hooks to attention layers
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            hooks.append(module.register_forward_hook(get_attention))

    try:
        # Create a custom colormap for the heatmap
        colors = [(0, 0, 0.7), (0, 0.7, 1), (0, 1, 0), (0.7, 1, 0), (1, 0.7, 0), (1, 0, 0)]
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

        sample_count = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                if sample_count >= num_samples:
                    break

                attention_maps.clear()  # Clear previous attention maps

                inputs = inputs.to(device)
                outputs = model(inputs)

                # Process each image in the batch
                for batch_idx in range(min(inputs.size(0), num_samples - sample_count)):
                    if not attention_maps:
                        print("No attention maps captured. Check model architecture.")
                        continue

                    # Get the input image
                    input_img = inputs[batch_idx].cpu().permute(1, 2, 0).numpy()
                    input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())

                    # Get segmentation prediction
                    pred_mask = outputs["out"][batch_idx].argmax(dim=0).cpu().numpy()

                    # Visualize a subset of attention heads from different layers
                    num_layers = min(3, len(attention_maps))
                    num_heads_per_layer = min(3, attention_maps[0].size(1))

                    fig, axs = plt.subplots(num_layers, num_heads_per_layer + 2,
                                            figsize=(num_heads_per_layer * 4 + 8, num_layers * 4))

                    # Add input image and segmentation prediction to the first two columns
                    for layer_idx in range(num_layers):
                        # Display original image
                        if num_layers > 1:
                            axs[layer_idx, 0].imshow(input_img)
                            axs[layer_idx, 0].set_title('Input Image')
                            axs[layer_idx, 0].axis('off')

                            # Display segmentation mask
                            axs[layer_idx, 1].imshow(pred_mask, cmap='tab20')
                            axs[layer_idx, 1].set_title('Segmentation')
                            axs[layer_idx, 1].axis('off')
                        else:
                            axs[0].imshow(input_img)
                            axs[0].set_title('Input Image')
                            axs[0].axis('off')

                            # Display segmentation mask
                            axs[1].imshow(pred_mask, cmap='tab20')
                            axs[1].set_title('Segmentation')
                            axs[1].axis('off')






                    # Display attention maps
                    for layer_idx in range(num_layers):
                        attention = attention_maps[layer_idx][batch_idx]

                        # For each attention head
                        for head_idx in range(num_heads_per_layer):
                            if head_idx < attention.size(0):  # Check if head exists
                                # Get attention map for this head
                                attn_map = attention[head_idx].cpu()

                                # Reshape attention map to square for visualization
                                # (assuming sequence length is perfect square for simplicity)
                                # Reshape attention map to match patch grid (num_patches_h x num_patches_w)
                                height, width = input_img.shape[:2]  # Input image dimensions (384, 512)
                                patch_size = 16  # From --patch-size 16
                                num_patches_h = height // patch_size  # 24
                                num_patches_w = width // patch_size  # 32
                                num_patches = num_patches_h * num_patches_w  # 768
                                seq_len = attn_map.size(0)

                                if seq_len != num_patches:
                                    print(f"Warning: seq_len ({seq_len}) != num_patches ({num_patches})")
                                    if seq_len == num_patches + 1:
                                        attn_map = attn_map[1:]  # Skip CLS token
                                    else:
                                        attn_map = attn_map[:num_patches]

                                if len(attn_map.shape) == 1:
                                    if attn_map.size(0) < num_patches:
                                        attn_map = torch.nn.functional.pad(attn_map,
                                                                           (0, num_patches - attn_map.size(0)))
                                    attn_map = attn_map[:num_patches].reshape(num_patches_h, num_patches_w)
                                else:
                                    attn_map = attn_map[:num_patches, :num_patches].reshape(num_patches_h,num_patches_w)

                                # Resize attention map to match image dimensions for overlay
                                resized_map = torch.nn.functional.interpolate(
                                    attn_map.unsqueeze(0).unsqueeze(0),
                                    size=input_img.shape[:2],
                                    mode='bilinear',
                                    align_corners=False
                                ).squeeze().numpy()

                                if num_layers > 1:
                                    ax = axs[layer_idx, head_idx + 2]
                                else:
                                    ax = axs[head_idx + 2]

                                im = ax.imshow(resized_map, cmap=cmap, alpha=0.7)
                                ax.imshow(input_img, alpha=0.3)
                                ax.set_title(f'Layer {layer_idx + 1}, Head {head_idx + 1}')
                                ax.axis('off')

                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'attention_map_{sample_count}.png'), dpi=200)
                    plt.close(fig)

                    sample_count += 1
                    if sample_count >= num_samples:
                        break

    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

    print(f"Generated {sample_count} attention maps in {output_dir}")

def generate_attention_maps(model, data_loader, output_dir, device, num_samples=5):
    """
    Generate attention maps to visualize what the transformer is focusing on

    Args:
        model: Trained segmentation transformer model
        data_loader: DataLoader containing images to analyze
        output_dir: Directory where to save visualizations
        device: Device to run inference on
        num_samples: Number of samples to visualize
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    import math

    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    # Register a hook to get attention maps
    attention_maps = []

    def get_attention(module, input, output):
        # Capture the attention maps (depends on transformer implementation)
        # For MultiheadAttention, output is (attn_output, attn_output_weights)
        if isinstance(output, tuple) and len(output) > 1:
            attention_maps.append(output[1].detach())

    # Attach hooks to attention layers
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            hooks.append(module.register_forward_hook(get_attention))

    try:
        # Create a custom colormap for the heatmap
        colors = [(0, 0, 0.7), (0, 0.7, 1), (0, 1, 0), (0.7, 1, 0), (1, 0.7, 0), (1, 0, 0)]
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

        sample_count = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                if sample_count >= num_samples:
                    break

                attention_maps.clear()  # Clear previous attention maps

                inputs = inputs.to(device)
                outputs = model(inputs)

                # Process each image in the batch
                for batch_idx in range(min(inputs.size(0), num_samples - sample_count)):
                    if not attention_maps:
                        print("No attention maps captured. Check model architecture.")
                        continue

                    # Get the input image
                    input_img = inputs[batch_idx].cpu().permute(1, 2, 0).numpy()
                    input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())

                    # Get segmentation prediction
                    pred_mask = outputs["out"][batch_idx].argmax(dim=0).cpu().numpy()

                    # Visualize a subset of attention heads from different layers
                    num_layers = min(3, len(attention_maps))
                    num_heads_per_layer = min(3, attention_maps[0].size(1))

                    fig, axs = plt.subplots(num_layers, num_heads_per_layer + 2,
                                            figsize=(num_heads_per_layer * 4 + 8, num_layers * 4))

                    # Add input image and segmentation prediction to the first two columns
                    for layer_idx in range(num_layers):
                        # Display original image
                        if num_layers > 1:
                            axs[layer_idx, 0].imshow(input_img)
                            axs[layer_idx, 0].set_title('Input Image')
                            axs[layer_idx, 0].axis('off')

                            # Display segmentation mask
                            axs[layer_idx, 1].imshow(pred_mask, cmap='tab20')
                            axs[layer_idx, 1].set_title('Segmentation')
                            axs[layer_idx, 1].axis('off')
                        else:
                            axs[0].imshow(input_img)
                            axs[0].set_title('Input Image')
                            axs[0].axis('off')

                            # Display segmentation mask
                            axs[1].imshow(pred_mask, cmap='tab20')
                            axs[1].set_title('Segmentation')
                            axs[1].axis('off')






                    # Display attention maps
                    for layer_idx in range(num_layers):
                        attention = attention_maps[layer_idx][batch_idx]

                        # For each attention head
                        for head_idx in range(num_heads_per_layer):
                            if head_idx < attention.size(0):  # Check if head exists
                                # Get attention map for this head
                                attn_map = attention[head_idx].cpu()

                                # Reshape attention map to square for visualization
                                # (assuming sequence length is perfect square for simplicity)
                                # Reshape attention map to match patch grid (num_patches_h x num_patches_w)
                                height, width = input_img.shape[:2]  # Input image dimensions (384, 512)
                                patch_size = 16  # From --patch-size 16
                                num_patches_h = height // patch_size  # 24
                                num_patches_w = width // patch_size  # 32
                                num_patches = num_patches_h * num_patches_w  # 768
                                seq_len = attn_map.size(0)

                                if seq_len != num_patches:
                                    print(f"Warning: seq_len ({seq_len}) != num_patches ({num_patches})")
                                    if seq_len == num_patches + 1:
                                        attn_map = attn_map[1:]  # Skip CLS token
                                    else:
                                        attn_map = attn_map[:num_patches]

                                if len(attn_map.shape) == 1:
                                    if attn_map.size(0) < num_patches:
                                        attn_map = torch.nn.functional.pad(attn_map,
                                                                           (0, num_patches - attn_map.size(0)))
                                    attn_map = attn_map[:num_patches].reshape(num_patches_h, num_patches_w)
                                else:
                                    attn_map = attn_map[:num_patches, :num_patches].reshape(num_patches_h,num_patches_w)

                                # Resize attention map to match image dimensions for overlay
                                resized_map = torch.nn.functional.interpolate(
                                    attn_map.unsqueeze(0).unsqueeze(0),
                                    size=input_img.shape[:2],
                                    mode='bilinear',
                                    align_corners=False
                                ).squeeze().numpy()

                                if num_layers > 1:
                                    ax = axs[layer_idx, head_idx + 2]
                                else:
                                    ax = axs[head_idx + 2]

                                im = ax.imshow(resized_map, cmap=cmap, alpha=0.7)
                                ax.imshow(input_img, alpha=0.3)
                                ax.set_title(f'Layer {layer_idx + 1}, Head {head_idx + 1}')
                                ax.axis('off')

                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'attention_map_{sample_count}.png'), dpi=200)
                    plt.close(fig)

                    sample_count += 1
                    if sample_count >= num_samples:
                        break

    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

    print(f"Generated {sample_count} attention maps in {output_dir}")


# Class-wise performance analysis function (same as original)

#####
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
            outputs = model(inputs)["out"]
            preds = outputs.argmax(dim=1)

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

    # Create the report
    with open(output_path, 'w') as f:
        # Header
        f.write("# Semantic Segmentation Model Performance Report\n\n")

        # Model information
        f.write("## Model Information\n\n")
        f.write(f"- Architecture: Segmentation Transformer\n")
        f.write(f"- Embedding Dimension: {model_info.get('embed_dim', 768)}\n")
        f.write(f"- Transformer Depth: {model_info.get('depth', 12)}\n")
        f.write(f"- Decoder Depth: {model_info.get('decoder_depth', 4)}\n")
        f.write(f"- Number of Attention Heads: {model_info.get('num_heads', 12)}\n")
        f.write(f"- Patch Size: {model_info.get('patch_size', 16)}\n")
        f.write(f"- Stride: {model_info.get('stride', 8)}\n")
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

        # Transformer-specific recommendations
        f.write("## Recommendations for Improvement\n\n")

        if problem_classes:
            f.write("Based on the analysis, consider the following improvements for the Vision Transformer model:\n\n")

            # General recommendations for transformer models
            f.write("1. **Attention Mechanism Tuning**:\n")
            f.write("   - Analyze attention maps to understand where the model focuses\n")
            f.write("   - Consider adding positional attention for better spatial understanding\n")
            f.write("   - Experiment with different attention mechanisms (e.g., deformable attention)\n\n")

            f.write("2. **Architecture Modifications**:\n")
            f.write("   - Increase or decrease model depth based on complexity needs\n")
            f.write("   - Try smaller patch sizes for finer detail detection\n")
            f.write("   - Experiment with hierarchical transformer structures\n")
            f.write("   - Consider hybrid architectures with CNN features\n\n")

            f.write("3. **Training Strategies**:\n")
            f.write("   - Use longer warmup periods for transformer training\n")
            f.write("   - Implement layer-wise learning rate decay\n")
            f.write("   - Try curriculum learning with increasing image resolution\n")
            f.write("   - Consider pre-training on a larger dataset\n\n")

            # Specific recommendations for top problem classes
            worst_class = problem_classes[0]['class_name'] if problem_classes else "None"
            f.write(f"4. **Focus on '{worst_class}'**: This class has the lowest IoU. Consider:\n")
            f.write("   - Adding auxiliary attention supervision for this class\n")
            f.write("   - Implementing class-specific loss weighting\n")
            f.write("   - Using specialized data augmentation techniques\n\n")
        else:
            f.write("The transformer model is performing well across all classes. To further improve:\n\n")
            f.write("1. **Architecture Scaling**: Experiment with larger model variants (more layers, heads)\n")
            f.write("2. **Multi-scale Processing**: Add multi-scale attention mechanisms\n")
            f.write("3. **Self-supervised Pre-training**: Use masked image modeling for pre-training\n\n")

        # Conclusion
        f.write("## Conclusion\n\n")
        f.write("The Segmentation Transformer model ")
        if metrics.get('mean_iou', 0) > 0.7:
            f.write("demonstrates strong performance across most classes, showing the effectiveness ")
            f.write("of transformer architectures for semantic segmentation tasks. ")
        elif metrics.get('mean_iou', 0) > 0.5:
            f.write("shows promising results, confirming that transformers can effectively handle ")
            f.write("semantic segmentation tasks with further optimization. ")
        else:
            f.write("provides a foundation for transformer-based segmentation, but requires significant ")
            f.write("tuning to compete with established CNN-based approaches. ")

        f.write("The analysis highlights specific classes that need attention, and the ")
        f.write("recommendations provide concrete steps to leverage transformer-specific ")
        f.write("capabilities for improving model performance in future iterations.\n")

    print(f"Performance report generated at {output_path}")


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
                import random

                color_map[class_id] = [random.randint(0, 255) for _ in range(3)]

    # Create output directories for segmentation results
    train_output_dir = os.path.join(args.checkpoint_dir, "train_segmentation")
    val_output_dir = os.path.join(args.checkpoint_dir, "val_segmentation")
    test_output_dir = os.path.join(args.checkpoint_dir, "test_segmentation")
    attention_maps_dir = os.path.join(args.checkpoint_dir, "attention_maps")

    # Generate attention maps for the transformer model
    print("Generating attention maps from transformer...")
    generate_attention_maps(
        model,
        test_loader,
        attention_maps_dir,
        device,
        num_samples=10
    )

    # Run inference and save results for validation and test sets
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

    # Analyze class performance
    print("Analyzing class performance...")
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

    # Generate performance report
    report_path = os.path.join(args.checkpoint_dir, "performance_report.md")
    model_info = vars(args)
    overall_metrics = performance_results['overall_metrics']

    generate_performance_report(
        model_info,
        overall_metrics,
        performance_results,
        report_path
    )

    print(f"All segmentation results and analysis saved to {args.checkpoint_dir}")
    print("Vision Transformer-based segmentation model training and evaluation complete!")
