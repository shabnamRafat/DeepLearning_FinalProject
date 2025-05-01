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

