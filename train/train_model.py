import argparse
import ast
import os
import sys
import time
import warnings

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
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
    load_color_map
)

# allow importing your custom dataset from ../dataset-loading
dataset_dir = os.path.join(parent_dir, "dataset-loading")
sys.path.insert(0, dataset_dir)

from Data_Preprocessing import A2D2_CSV_dataset  # your CSV‐based Dataset

warnings.filterwarnings("ignore")

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
        default="deeplabv3_mobilenet_v3_large",
        help="Torchvision segmentation model name"
    )

    # infra configuration
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--amp", type=str, default="True")

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

    # Dataset & DataLoader
    train_data = A2D2_CSV_dataset(
        csv_file=args.pairs_csv,
        class_list=args.class_list,
        cache=args.cache,
        height=args.height,
        width=args.width,
        transform=image_transform,
        target_transform=target_transform,
    )
    val_data = A2D2_CSV_dataset(
        csv_file=args.pairs_csv,
        class_list=args.class_list,
        cache=args.cache,
        height=args.height,
        width=args.width,
        transform=image_transform,
        target_transform=target_transform,
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True, prefetch_factor=args.prefetch,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        drop_last=True, prefetch_factor=args.prefetch,
        persistent_workers=True,
    )

    # Model, loss, optimizer, AMP
    model = torch.hub.load(
        "pytorch/vision:v0.9.1",
        args.network,
        pretrained=False,
        num_classes=args.classes
    )
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    CE = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scaler = GradScaler(enabled=ast.literal_eval(args.amp))

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        bstart = time.time()
        for i, (inputs, masks) in enumerate(train_loader):
            if i > args.iterations:
                break

            model.train()
            inputs, masks = inputs.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast(enabled=ast.literal_eval(args.amp)):
                outputs = model(inputs)["out"]
                loss = CE(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # logging & validation
            if i > 0 and (i % args.log_freq == 0):
                bstop = time.time()

                # accumulate validation stats
                val_losses = []
                hist = torch.zeros(args.classes, args.classes, dtype=torch.int64, device=device)
                infer_start = time.time()
                model.eval()
                with torch.no_grad():
                    for j, (v_in, v_mask) in enumerate(val_loader):
                        v_in, v_mask = v_in.to(device), v_mask.to(device)
                        out = model(v_in)["out"]
                        val_losses.append(CE(out, v_mask))

                        preds = out.argmax(dim=1).view(-1)
                        gts = v_mask.view(-1)
                        hist += _fast_hist(preds, gts, args.classes)

                        if j * args.batch >= args.eval_size:
                            break
                infer_end = time.time()

                # compute metrics
                metrics = compute_metrics(hist)
                avg_val_loss = torch.stack(val_losses).mean().item()
                num_images = min(len(val_loader) * args.batch, args.eval_size)
                fps = num_images / (infer_end - infer_start)

                # print
                print(f"processed {i * args.batch} records in {bstop - bstart:.2f}s")
                print(
                    f"batch {i}: "
                    f"Training_loss {loss:.4f}, Val_loss {avg_val_loss:.4f}, "
                    f"Mean_IoU {metrics['mean_iou']:.4f}, PixAcc {metrics['pixel_acc']:.4f}, "
                    f"FPS {fps:.2f}"
                )

        # Save checkpoint after each epoch
        ckpt = os.path.join(args.checkpoint_dir, f"model-ep{epoch}.pth")
        if os.path.exists(ckpt):
            os.remove(ckpt)
        with open(ckpt, "wb") as f:
            torch.save(model.state_dict(), f)

        # update "latest.pth"
        latest = os.path.join(args.checkpoint_dir, "latest.pth")
        if os.path.exists(latest):
            os.remove(latest)
        with open(latest, "wb") as f:
            torch.save(model.state_dict(), f)

    # final save
    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "final_model.pth"))

    # Load color map
    color_map = load_color_map(args.class_list)

    # Create an output directory for the segmentation results
    output_dir = os.path.join(args.checkpoint_dir, "segmentation_output")

    # Run inference and save results
    print("Generating segmentation outputs...")
    save_segmentation_results(
        model,
        val_loader,
        output_dir,
        device,
        color_map
    )
    print(f"Segmentation results saved to {output_dir}")