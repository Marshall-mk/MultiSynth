"""
Training script for U-Net Super-Resolution with Multi-Stack Input

This script trains a 3D U-Net model using orthogonal low-resolution stacks
(axial, coronal, sagittal) concatenated channel-wise as input. Supports flexible
stack selection (all 3 or any 2-stack combination).

Simplified from the UHVED training pipeline - uses standard U-Net architecture
with L1 + SSIM loss instead of VAE-based approach.
"""

import os
import argparse
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple, Any
import wandb
import numpy as np

# Accelerate for easy multi-GPU training
from accelerate import Accelerator
from accelerate.utils import set_seed

# Transformers imports for scheduler
from transformers import get_cosine_schedule_with_warmup

# MONAI imports
from monai.data import DataLoader
from monai.inferers import sliding_window_inference

# Import our modules
from src.unet import CustomUNet3D
from src.data import HRLRDataGenerator, create_dataset
from src.losses import SSIMLoss, PerceptualLoss3D
from src.utils import (
    save_model_checkpoint,
    get_image_paths,
    save_training_config,
    find_latest_checkpoint,
    calculate_metrics,
    print_model_statistics,
    get_gpu_memory_stats,
    print_gpu_memory_stats,
)


class LPIPSLoss3D(nn.Module):
    """
    3D-LPIPS loss using proper 3D medical imaging networks.

    Uses PerceptualLoss3D infrastructure with MedicalNet, MONAI, or Models Genesis
    to compute perceptual similarity in true 3D feature space.
    """

    def __init__(self, backend: str = 'monai', feature_layers: List[str] = None):
        """
        Args:
            backend: 3D network backend ('medicalnet', 'monai', 'models_genesis')
            feature_layers: Layers to extract features from
        """
        super().__init__()

        if feature_layers is None:
            feature_layers = ['layer1', 'layer2', 'layer3', 'layer4']

        self.feature_layers = feature_layers
        self.perceptual_loss = PerceptualLoss3D(
            backend=backend,
            model_depth=18,
            feature_layers=feature_layers,
            weights=[1.0] * len(feature_layers),
            pretrained=False,  # Random init for metric
            normalize_input=True,
            freeze_backbone=True
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute 3D-LPIPS loss.

        Args:
            pred: Predicted volume (B, C, D, H, W)
            target: Target volume (B, C, D, H, W)

        Returns:
            LPIPS loss (scalar)
        """
        # Extract features from both volumes
        pred_features = self.perceptual_loss.extract_features(pred)
        target_features = self.perceptual_loss.extract_features(target)

        # Compute L2 distance in feature space (LPIPS formula)
        lpips_score = 0.0
        num_layers = 0

        for layer_name in self.feature_layers:
            pred_feat = None
            target_feat = None

            # Find matching features
            for name, feat in pred_features.items():
                if layer_name in name:
                    pred_feat = feat
                    break

            for name, feat in target_features.items():
                if layer_name in name:
                    target_feat = feat
                    break

            if pred_feat is not None and target_feat is not None:
                # Spatial L2 normalization (LPIPS style)
                diff = (pred_feat - target_feat) ** 2
                # Average across spatial dimensions and channels
                layer_dist = diff.mean(dim=[1, 2, 3, 4]).mean()
                lpips_score = lpips_score + layer_dist
                num_layers += 1

        # Average across layers
        if num_layers > 0:
            lpips_score = lpips_score / num_layers

        return lpips_score


class UNetLoss(nn.Module):
    """Simplified loss for U-Net without KL or orientation losses."""

    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 1.0,
        lpips_weight: float = 0.0,
        use_lpips: bool = False,
        lpips_backend: str = 'monai'
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.lpips_weight = lpips_weight
        self.use_lpips = use_lpips

        self.ssim_loss = SSIMLoss(spatial_dims=3)

        # Initialize LPIPS if requested
        if use_lpips:
            try:
                self.lpips_loss = LPIPSLoss3D(backend=lpips_backend)
                print(f"3D-LPIPS loss initialized with backend: {lpips_backend}")
            except Exception as e:
                print(f"Warning: Failed to initialize LPIPS: {e}. Skipping LPIPS loss.")
                self.lpips_loss = None
                self.use_lpips = False
        else:
            self.lpips_loss = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss between prediction and target.

        Args:
            pred: Predicted HR volume (B, 1, D, H, W)
            target: Target HR volume (B, 1, D, H, W)

        Returns:
            Dictionary with loss components
        """
        l1 = F.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)

        total_loss = self.l1_weight * l1 + self.ssim_weight * ssim

        losses = {
            'l1': l1,
            'ssim': ssim,
        }

        # Compute LPIPS if enabled
        if self.use_lpips and self.lpips_loss is not None:
            lpips_val = self.lpips_loss(pred, target)
            losses['lpips'] = lpips_val
            total_loss = total_loss + self.lpips_weight * lpips_val
        else:
            losses['lpips'] = torch.tensor(0.0, device=pred.device)

        losses['total'] = total_loss
        return losses


def parse_stack_selection(use_stacks: str) -> List[int]:
    """
    Parse stack selection string to list of indices.

    Args:
        use_stacks: "all", "012", "01", "02", "12"

    Returns:
        List of stack indices, e.g., [0, 1, 2] or [0, 1]

    Raises:
        ValueError: If invalid stack selection format
    """
    if use_stacks.lower() == "all":
        return [0, 1, 2]

    # Parse digit string
    try:
        indices = [int(c) for c in use_stacks if c.isdigit()]
        if not indices or len(indices) < 2 or len(indices) > 3:
            raise ValueError(f"Invalid stack selection: {use_stacks}")
        if any(i not in [0, 1, 2] for i in indices):
            raise ValueError(f"Stack indices must be 0, 1, or 2")
        return sorted(indices)
    except Exception as e:
        raise ValueError(
            f"Invalid --use_stacks format: {use_stacks}. "
            f"Use 'all', '012', '01', '02', or '12'"
        )


def concatenate_stacks(
    lr_stacks_list: List[torch.Tensor],
    stack_indices: List[int]
) -> torch.Tensor:
    """
    Concatenate selected LR stacks channel-wise.

    Args:
        lr_stacks_list: List of 3 tensors, each (B, 1, D, H, W)
        stack_indices: Indices of stacks to use, e.g., [0, 1, 2] or [0, 2]

    Returns:
        Concatenated tensor (B, N, D, H, W) where N = len(stack_indices)
    """
    selected_stacks = [lr_stacks_list[i] for i in stack_indices]
    return torch.cat(selected_stacks, dim=1)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    stack_indices: List[int],
    accelerator: Accelerator,
    use_sliding_window: bool = False,
    roi_size: Optional[Tuple[int, ...]] = None,
    overlap: float = 0.5,
) -> Dict[str, float]:
    """
    Run validation loop.

    Args:
        model: U-Net model
        val_loader: Validation DataLoader
        criterion: Loss function
        stack_indices: Indices of stacks to use
        accelerator: Accelerate Accelerator
        use_sliding_window: Use sliding window inference
        roi_size: ROI size for sliding window
        overlap: Overlap ratio for sliding window

    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    val_losses = []
    val_l1_losses = []
    val_ssim_losses = []
    val_lpips_losses = []
    metrics_sum = {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "psnr": 0.0, "r2": 0.0, "ssim": 0.0}
    num_batches = 0

    with torch.no_grad():
        for batch_data in tqdm(
            val_loader,
            desc="Validation",
            disable=not accelerator.is_local_main_process,
            leave=False
        ):
            # Unpack batch
            lr_stacks_list, target_img, _, _, _ = batch_data

            # Concatenate selected stacks
            lr_input = concatenate_stacks(lr_stacks_list, stack_indices)

            # Forward pass
            if use_sliding_window and roi_size is not None:
                pred = sliding_window_inference(
                    inputs=lr_input,
                    roi_size=roi_size,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=overlap,
                    mode="gaussian",
                )
            else:
                pred = model(lr_input)

            # Compute loss
            loss_dict = criterion(pred, target_img)
            val_losses.append(loss_dict['total'].item())
            val_l1_losses.append(loss_dict['l1'].item())
            val_ssim_losses.append(loss_dict['ssim'].item())
            val_lpips_losses.append(loss_dict['lpips'].item())

            # Calculate metrics
            batch_metrics = calculate_metrics(pred, target_img, max_val=1.0)
            for key in metrics_sum:
                metrics_sum[key] += batch_metrics[key]
            num_batches += 1

    # Average metrics
    val_metrics = {
        'val_loss': np.mean(val_losses),
        'val_l1': np.mean(val_l1_losses),
        'val_ssim': np.mean(val_ssim_losses),
        'val_lpips': np.mean(val_lpips_losses),
    }

    # Add other metrics
    for key in metrics_sum:
        val_metrics[f'val_{key}'] = metrics_sum[key] / num_batches

    return val_metrics


def train_unet_model(
    hr_image_paths: List[str],
    model_dir: str,
    use_stacks: str = "all",
    epochs: int = 100,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    output_shape: tuple = (128, 128, 128),
    checkpoint: str = None,
    device: str = "cuda",
    save_interval: int = 10,
    val_interval: int = 5,
    val_image_paths: List[str] = None,
    atlas_res: list = [1.0, 1.0, 1.0],
    min_resolution: list = [1.0, 1.0, 1.0],
    max_res_aniso: list = [9.0, 9.0, 9.0],
    randomise_res: bool = True,
    prob_motion: float = 0.2,
    prob_spike: float = 0.05,
    prob_aliasing: float = 0.1,
    prob_bias_field: float = 0.5,
    prob_noise: float = 0.8,
    apply_intensity_aug: bool = True,
    orientation_dropout_prob: float = 0.0,
    min_orientations: int = 1,
    drop_orientations: list = None,
    num_workers: int = None,
    use_cache: bool = False,
    use_wandb: bool = False,
    wandb_project: str = "unet-super-resolution",
    wandb_entity: str = None,
    wandb_run_name: str = None,
    nb_features: int = 24,
    nb_levels: int = 5,
    conv_size: int = 3,
    final_activation: str = "sigmoid",
    mixed_precision: str = "no",
    gradient_accumulation_steps: int = 1,
    seed: int = 42,
    l1_weight: float = 1.0,
    ssim_weight: float = 1.0,
    lpips_weight: float = 0.0,
    use_lpips: bool = False,
    lpips_backend: str = 'monai',
    use_sliding_window_val: bool = False,
    val_patch_size: tuple = (96, 96, 96),
    val_overlap: float = 0.5,
    max_grad_norm: float = 1.0,
    weight_decay: float = 1e-5,
    warmup_steps: int = 500,
    early_stopping_patience: int = 20,
    csv_log: bool = True,
    upsample_mode: str = "nearest",
):
    """
    Train 3D U-Net model with multi-stack input.

    Args:
        hr_image_paths: List of paths to high-resolution images
        model_dir: Directory to save trained models
        use_stacks: Stack selection ("all", "012", "01", "02", "12")
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        output_shape: Output volume shape
        checkpoint: Optional checkpoint to resume from
        device: 'cuda' or 'cpu'
        save_interval: Save checkpoint every N epochs
        val_interval: Run validation every N epochs
        val_image_paths: Optional list of validation image paths
        atlas_res: Physical resolution of input HR images [x, y, z] in mm
        min_resolution: Minimum resolution for randomization
        max_res_aniso: Maximum anisotropic resolution
        randomise_res: Whether to randomize resolution
        prob_motion: Probability of motion artifacts
        prob_spike: Probability of k-space spikes
        prob_aliasing: Probability of aliasing artifacts
        prob_bias_field: Probability of bias field
        prob_noise: Probability of noise
        apply_intensity_aug: Whether to apply intensity augmentation
        orientation_dropout_prob: Probability of orientation dropout
        min_orientations: Minimum orientations after dropout
        drop_orientations: Specific orientations to drop
        num_workers: Number of data loading workers
        use_cache: Whether to use CacheDataset
        use_wandb: Whether to use Weights & Biases
        wandb_project: W&B project name
        wandb_entity: W&B entity/team name
        wandb_run_name: W&B run name
        nb_features: Base number of features for U-Net
        nb_levels: Number of U-Net levels
        conv_size: Convolution kernel size
        final_activation: Final activation function
        mixed_precision: Mixed precision training ('no', 'fp16', 'bf16')
        gradient_accumulation_steps: Number of steps to accumulate gradients
        seed: Random seed
        l1_weight: Weight for L1 loss
        ssim_weight: Weight for SSIM loss
        lpips_weight: Weight for LPIPS loss
        use_lpips: Enable LPIPS perceptual loss
        lpips_backend: Backend for 3D-LPIPS ('medicalnet', 'monai', 'models_genesis')
        use_sliding_window_val: Use sliding window for validation
        val_patch_size: Patch size for sliding window validation
        val_overlap: Overlap for sliding window validation
        max_grad_norm: Max gradient norm for clipping
        weight_decay: Weight decay for optimizer
        warmup_steps: Warmup steps for scheduler
        early_stopping_patience: Early stopping patience
        csv_log: Enable CSV logging
        upsample_mode: Interpolation mode for data generation
    """
    # Parse stack selection
    stack_indices = parse_stack_selection(use_stacks)
    num_input_channels = len(stack_indices)

    # Initialize Accelerator
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with="wandb" if use_wandb else None,
    )

    # Set seed
    set_seed(seed)

    # Print configuration
    if accelerator.is_main_process:
        print("=" * 80)
        print("Training 3D U-Net with Multi-Stack Input")
        print("=" * 80)
        print(f"Stack selection: {use_stacks} → indices {stack_indices}")
        print(f"Input channels: {num_input_channels}")
        print(f"Output shape: {output_shape}")
        print(f"Distributed training: {accelerator.num_processes} process(es)")
        print(f"Mixed precision: {mixed_precision}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")

    # Create model directory
    os.makedirs(model_dir, exist_ok=True)

    # Auto-detect num_workers
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        if device == "cuda" and torch.cuda.is_available():
            num_workers = min(4, max(cpu_count // 2, 1))
        elif cpu_count >= 4:
            num_workers = 2
        else:
            num_workers = 0

    pin_memory = False  # Accelerate handles this

    if accelerator.is_main_process:
        print(f"DataLoader settings: num_workers={num_workers}, pin_memory={pin_memory}, use_cache={use_cache}")

    # Create data generator
    if accelerator.is_main_process:
        print(f"Input HR image resolution: {atlas_res} mm")
        print(f"Resolution randomization: {min_resolution} to {max_res_aniso} mm")
        print(f"Generating 3 orthogonal LR stacks per HR volume (RAS-oriented)")
        print(f"  - Stack 0 (Axial): High in-plane (R,A), Low through-plane (S)")
        print(f"  - Stack 1 (Coronal): High in-plane (R,S), Low through-plane (A)")
        print(f"  - Stack 2 (Sagittal): High in-plane (A,S), Low through-plane (R)")
        if orientation_dropout_prob > 0.0:
            print(f"Orientation dropout enabled: {orientation_dropout_prob:.2f} probability")

    generator = HRLRDataGenerator(
        atlas_res=atlas_res,
        target_res=[1.0, 1.0, 1.0],
        output_shape=list(output_shape),
        min_resolution=min_resolution,
        max_res_aniso=max_res_aniso,
        randomise_res=randomise_res,
        prob_motion=prob_motion,
        prob_spike=prob_spike,
        prob_aliasing=prob_aliasing,
        prob_bias_field=prob_bias_field,
        prob_noise=prob_noise,
        apply_intensity_aug=apply_intensity_aug,
        clip_to_unit_range=True,
        orientation_dropout_prob=orientation_dropout_prob,
        min_orientations=min_orientations,
        drop_orientations=drop_orientations,
        upsample_mode=upsample_mode,
    )

    # Create dataset
    dataset = create_dataset(
        image_paths=hr_image_paths,
        generator=generator,
        target_shape=list(output_shape),
        target_spacing=atlas_res,
        use_cache=use_cache,
        return_resolution=True,
        is_training=True,
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    if accelerator.is_main_process:
        print(f"Training dataset: {len(dataset)} images")

    # Create validation dataset
    val_dataloader = None
    if val_image_paths:
        val_dataset = create_dataset(
            image_paths=val_image_paths,
            generator=generator,
            target_shape=list(output_shape),
            target_spacing=atlas_res,
            use_cache=use_cache,
            return_resolution=True,
            is_training=False,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )

        if accelerator.is_main_process:
            print(f"Validation dataset: {len(val_dataset)} images")

    # Auto-detect checkpoint
    start_epoch = 0
    checkpoint_data = None

    if checkpoint is None:
        checkpoint = find_latest_checkpoint(model_dir, model_type="unet")
        if checkpoint and accelerator.is_main_process:
            print(f"Auto-detected checkpoint: {checkpoint}")

    # Load checkpoint if available
    if checkpoint and os.path.exists(checkpoint):
        if accelerator.is_main_process:
            print(f"Loading checkpoint from {checkpoint}")
        checkpoint_data = torch.load(checkpoint, map_location="cpu")
        start_epoch = checkpoint_data.get("epoch", 0) + 1
        if accelerator.is_main_process:
            print(f"Resuming training from epoch {start_epoch}")

    # Create U-Net model
    if accelerator.is_main_process:
        print(f"Creating U-Net model:")
        print(f"  - Input channels: {num_input_channels}")
        print(f"  - Base features: {nb_features}")
        print(f"  - Number of levels: {nb_levels}")
        print(f"  - Final activation: {final_activation}")

    model = CustomUNet3D(
        nb_features=nb_features,
        input_shape=(num_input_channels, *output_shape),
        nb_levels=nb_levels,
        conv_size=conv_size,
        nb_labels=1,
        final_pred_activation=final_activation,
    )

    # Load checkpoint weights
    if checkpoint_data is not None:
        model.load_state_dict(checkpoint_data["model_state_dict"])
        if accelerator.is_main_process:
            print(f"Loaded model weights from checkpoint")

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    criterion = UNetLoss(
        l1_weight=l1_weight,
        ssim_weight=ssim_weight,
        lpips_weight=lpips_weight,
        use_lpips=use_lpips,
        lpips_backend=lpips_backend,
    )

    if accelerator.is_main_process:
        loss_info = f"Loss weights: l1={l1_weight}, ssim={ssim_weight}"
        if use_lpips:
            loss_info += f", lpips={lpips_weight} (backend={lpips_backend})"
        print(loss_info)

    # Learning rate scheduler
    num_steps = len(dataloader) * epochs // gradient_accumulation_steps

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_steps,
    )

    if accelerator.is_main_process:
        print(f"Using Cosine LR schedule with warmup ({warmup_steps} warmup steps, {num_steps} total steps)")
        if max_grad_norm > 0:
            print(f"Gradient clipping enabled: max_norm={max_grad_norm}")

    # Load optimizer and scheduler state
    if checkpoint_data is not None:
        if "optimizer_state_dict" in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint_data:
            scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])

    # Prepare everything with accelerator
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)

    if accelerator.is_main_process:
        # Print model statistics
        input_shape = (batch_size, num_input_channels, *output_shape)
        print_model_statistics(accelerator.unwrap_model(model), [input_shape], device=str(accelerator.device))
        print(f"\nTraining for {epochs} epochs, batch size {batch_size}")
        print(f"Initial learning rate: {learning_rate}")
        print(f"Device: {accelerator.device}")

    # Track best validation loss
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    if checkpoint_data is not None and 'val_loss' in checkpoint_data and checkpoint_data['val_loss'] is not None:
        best_val_loss = checkpoint_data['val_loss']
        if accelerator.is_main_process:
            print(f"Best validation loss from checkpoint: {best_val_loss:.4f}")

    # Initialize W&B
    if use_wandb and accelerator.is_main_process:
        wandb_config = {
            "model": "unet",
            "use_stacks": use_stacks,
            "num_input_channels": num_input_channels,
            "nb_features": nb_features,
            "nb_levels": nb_levels,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "output_shape": output_shape,
            "l1_weight": l1_weight,
            "ssim_weight": ssim_weight,
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "n_train_samples": len(hr_image_paths),
            "n_val_samples": len(val_image_paths) if val_image_paths else 0,
        }

        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config=wandb_config,
            resume="allow" if checkpoint else False,
        )
        wandb.watch(model, log="gradients", log_freq=100)
        print(f"Weights & Biases initialized: {wandb.run.name}")

    # Setup CSV logging
    csv_file = None
    csv_writer = None
    if accelerator.is_main_process and csv_log:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"training_log_unet_{timestamp}.csv"
        csv_path = os.path.join(model_dir, csv_filename)

        csv_headers = ["epoch", "step", "train_loss", "train_l1", "train_ssim", "train_lpips",
                       "val_loss", "val_l1", "val_ssim", "val_lpips", "val_mae", "val_psnr", "lr", "epoch_time"]

        csv_file = open(csv_path, mode='w', newline='')
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
        csv_writer.writeheader()
        csv_file.flush()
        print(f"Logging training metrics to: {csv_path}")

    # Training loop
    global_step = 0

    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_losses = []
        epoch_l1_losses = []
        epoch_ssim_losses = []
        epoch_lpips_losses = []

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            disable=not accelerator.is_main_process,
            leave=False,
            dynamic_ncols=True,
        )

        for batch_idx, batch_data in enumerate(pbar):
            # Unpack batch
            lr_stacks_list, target_img, _, _, _ = batch_data

            # Concatenate selected stacks
            lr_input = concatenate_stacks(lr_stacks_list, stack_indices)

            # Forward and backward pass
            with accelerator.accumulate(model):
                pred = model(lr_input)
                loss_dict = criterion(pred, target_img)
                loss = loss_dict['total']

                # Backward pass
                optimizer.zero_grad()
                accelerator.backward(loss)

                # Gradient clipping
                if max_grad_norm > 0:
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

                optimizer.step()
                scheduler.step()

                global_step += 1

            # Track losses
            epoch_losses.append(loss_dict['total'].item())
            epoch_l1_losses.append(loss_dict['l1'].item())
            epoch_ssim_losses.append(loss_dict['ssim'].item())
            epoch_lpips_losses.append(loss_dict['lpips'].item())

            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            postfix_dict = {
                "loss": f"{loss_dict['total'].item():.4f}",
                "l1": f"{loss_dict['l1'].item():.4f}",
                "ssim": f"{loss_dict['ssim'].item():.4f}",
                "lr": f"{current_lr:.2e}"
            }

            # Add LPIPS to progress bar if enabled
            if use_lpips:
                postfix_dict["lpips"] = f"{loss_dict['lpips'].item():.4f}"

            # Add GPU memory
            if batch_idx % 10 == 0:
                mem_stats = get_gpu_memory_stats(accelerator.device)
                if mem_stats:
                    postfix_dict["mem"] = f"{mem_stats['allocated_mb']:.0f}MB"

            pbar.set_postfix(postfix_dict)

            # W&B logging
            if use_wandb and accelerator.is_main_process and global_step % 10 == 0:
                wandb_log = {
                    'train/loss': loss_dict['total'].item(),
                    'train/l1': loss_dict['l1'].item(),
                    'train/ssim': loss_dict['ssim'].item(),
                    'train/lr': current_lr,
                    'train/epoch': epoch,
                }
                if use_lpips:
                    wandb_log['train/lpips'] = loss_dict['lpips'].item()
                wandb.log(wandb_log, step=global_step)

        avg_loss = np.mean(epoch_losses)
        avg_l1 = np.mean(epoch_l1_losses)
        avg_ssim = np.mean(epoch_ssim_losses)
        avg_lpips = np.mean(epoch_lpips_losses) if use_lpips else 0.0

        # Validation
        val_metrics = None
        if val_dataloader and (epoch + 1) % val_interval == 0:
            val_metrics = validate(
                model=model,
                val_loader=val_dataloader,
                criterion=criterion,
                stack_indices=stack_indices,
                accelerator=accelerator,
                use_sliding_window=use_sliding_window_val,
                roi_size=tuple(val_patch_size) if use_sliding_window_val else None,
                overlap=val_overlap,
            )

            epoch_time = time.time() - epoch_start_time

            # Print validation results
            train_info = f"(L1: {avg_l1:.4f}, SSIM: {avg_ssim:.4f}"
            if use_lpips:
                train_info += f", LPIPS: {avg_lpips:.4f}"
            train_info += ")"

            accelerator.print(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_loss:.4f} {train_info} - "
                f"Val Loss: {val_metrics['val_loss']:.4f}"
            )
            accelerator.print(
                f"  Val Metrics - MAE: {val_metrics['val_mae']:.4f} | "
                f"PSNR: {val_metrics['val_psnr']:.2f} dB | "
                f"SSIM: {val_metrics['val_ssim']:.4f} | LR: {current_lr:.2e}"
            )

            # W&B logging
            if use_wandb and accelerator.is_main_process:
                wandb.log(val_metrics, step=global_step)

            # CSV logging
            if accelerator.is_main_process and csv_writer is not None:
                csv_writer.writerow({
                    "epoch": epoch + 1,
                    "step": global_step,
                    "train_loss": avg_loss,
                    "train_l1": avg_l1,
                    "train_ssim": avg_ssim,
                    "train_lpips": avg_lpips,
                    "val_loss": val_metrics['val_loss'],
                    "val_l1": val_metrics['val_l1'],
                    "val_ssim": val_metrics['val_ssim'],
                    "val_lpips": val_metrics['val_lpips'],
                    "val_mae": val_metrics['val_mae'],
                    "val_psnr": val_metrics['val_psnr'],
                    "lr": current_lr,
                    "epoch_time": epoch_time,
                })
                csv_file.flush()

            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                epochs_without_improvement = 0

                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    best_model_path = os.path.join(model_dir, "unet_best.pth")

                    unwrapped_model = accelerator.unwrap_model(model)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': unwrapped_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'global_step': global_step,
                        'model_config': {
                            'use_stacks': use_stacks,
                            'num_input_channels': num_input_channels,
                            'nb_features': nb_features,
                            'nb_levels': nb_levels,
                            'output_shape': output_shape,
                        }
                    }, best_model_path)

                    accelerator.print(f"Saved best model (val_loss: {best_val_loss:.4f}): {best_model_path}")
            else:
                epochs_without_improvement += 1

            # Early stopping
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                accelerator.print(f"Early stopping after {epochs_without_improvement} epochs without improvement")
                break
        else:
            epoch_time = time.time() - epoch_start_time
            train_info = f"(L1: {avg_l1:.4f}, SSIM: {avg_ssim:.4f}"
            if use_lpips:
                train_info += f", LPIPS: {avg_lpips:.4f}"
            train_info += ")"

            accelerator.print(
                f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} {train_info} - LR: {current_lr:.2e}"
            )

        # Periodic checkpointing
        if (epoch + 1) % save_interval == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                checkpoint_path = os.path.join(model_dir, f"unet_epoch_{epoch + 1:04d}.pth")

                unwrapped_model = accelerator.unwrap_model(model)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'global_step': global_step,
                    'val_loss': val_metrics['val_loss'] if val_metrics else None,
                }, checkpoint_path)

                accelerator.print(f"Saved checkpoint: {checkpoint_path}")

    # Close CSV file
    if accelerator.is_main_process and csv_file is not None:
        csv_file.close()

    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_path = os.path.join(model_dir, "unet_final.pth")

        unwrapped_model = accelerator.unwrap_model(model)
        torch.save({
            'epoch': epochs - 1,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'global_step': global_step,
            'model_config': {
                'use_stacks': use_stacks,
                'num_input_channels': num_input_channels,
                'nb_features': nb_features,
                'nb_levels': nb_levels,
                'output_shape': output_shape,
            }
        }, final_path)

        print(f"Training complete! Final model saved to: {final_path}")
        if best_val_loss < float('inf'):
            print(f"Best validation loss: {best_val_loss:.4f}")

        # Print memory stats
        print()
        print_gpu_memory_stats(accelerator.device, prefix="Training Complete - Peak Memory")

        if use_wandb:
            artifact = wandb.Artifact(
                name=f"unet-model-{wandb.run.id}",
                type="model",
                description="Final trained U-Net model with multi-stack input",
            )
            artifact.add_file(final_path)
            wandb.log_artifact(artifact)
            wandb.finish()

    accelerator.end_training()


if __name__ == "__main__":
    # Set multiprocessing start method
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Train 3D U-Net for Multi-Stack Super-Resolution")

    # Stack selection (NEW)
    parser.add_argument("--use_stacks", type=str, default="all",
                        help="Which stacks to use: 'all'/'012' (3 stacks), '01', '02', '12' (2 stacks)")

    # Data arguments
    parser.add_argument("--hr_image_dir", type=str, default=None, help="Directory containing HR images")
    parser.add_argument("--csv_file", type=str, default=None, help="CSV file with image metadata")
    parser.add_argument("--base_dir", type=str, default=None, help="Base directory for CSV paths")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory to save models")
    parser.add_argument("--val_image_dir", type=str, default=None, help="Validation images directory")
    parser.add_argument("--mri_classes", type=str, nargs="+", default=None,
                       help="MRI classifications to include (e.g., T1 T2 FLAIR)")
    parser.add_argument("--acquisition_types", type=str, nargs="+", default=["3D"],
                       help="Acquisition types to include (e.g., 3D 2D)")
    parser.add_argument("--no_filter_4d", action="store_true",
                       help="Don't filter out 4D images")

    # Model parameters
    parser.add_argument("--nb_features", type=int, default=24, help="Base number of features")
    parser.add_argument("--nb_levels", type=int, default=5, help="Number of U-Net levels")
    parser.add_argument("--conv_size", type=int, default=3, help="Convolution kernel size")
    parser.add_argument("--final_activation", type=str, default="sigmoid",
                        choices=["linear", "sigmoid", "tanh"],
                        help="Final activation function")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output_shape", type=int, nargs=3, default=[128, 128, 128], help="Output shape")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm (0 to disable)")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps for scheduler")
    parser.add_argument("--early_stopping_patience", type=int, default=20, help="Early stopping patience (0 to disable)")

    # Loss weights
    parser.add_argument("--l1_weight", type=float, default=1.0, help="L1 loss weight")
    parser.add_argument("--ssim_weight", type=float, default=1.0, help="SSIM loss weight")
    parser.add_argument("--lpips_weight", type=float, default=0.0, help="LPIPS loss weight")
    parser.add_argument("--use_lpips", action="store_true", help="Enable 3D-LPIPS perceptual loss")
    parser.add_argument("--lpips_backend", type=str, default="monai",
                        choices=["medicalnet", "monai", "models_genesis"],
                        help="Backend for 3D-LPIPS perceptual loss")

    # Data generation parameters
    parser.add_argument("--atlas_res", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="HR resolution")
    parser.add_argument("--min_resolution", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Min resolution")
    parser.add_argument("--max_res_aniso", type=float, nargs=3, default=[9.0, 9.0, 9.0], help="Max aniso resolution")
    parser.add_argument("--no_randomise_res", action="store_true", help="Disable resolution randomization")
    parser.add_argument("--prob_motion", type=float, default=0.2, help="Probability of motion artifacts")
    parser.add_argument("--prob_spike", type=float, default=0.05, help="Probability of k-space spikes")
    parser.add_argument("--prob_aliasing", type=float, default=0.1, help="Probability of aliasing")
    parser.add_argument("--prob_bias_field", type=float, default=0.5, help="Probability of bias field")
    parser.add_argument("--prob_noise", type=float, default=0.8, help="Probability of noise")
    parser.add_argument("--no_intensity_aug", action="store_true", help="Disable intensity augmentation")

    # Orientation dropout
    parser.add_argument("--orientation_dropout_prob", type=float, default=0.0,
                        help="Probability of orientation dropout (0.0-1.0)")
    parser.add_argument("--min_orientations", type=int, default=1,
                        help="Minimum orientations after dropout (1-3)")
    parser.add_argument("--drop_orientations", type=int, nargs="+", default=None,
                        choices=[0, 1, 2],
                        help="Specific orientations to drop (0=Axial, 1=Coronal, 2=Sagittal)")

    # Data interpolation
    parser.add_argument("--upsample_mode", type=str, default="nearest",
                        choices=["nearest", "trilinear", "nearest-exact"],
                        help="Interpolation mode for FFT upsample recovery")

    # Validation parameters
    parser.add_argument("--val_interval", type=int, default=5, help="Validate every N epochs")
    parser.add_argument("--use_sliding_window_val", action="store_true",
                       help="Use sliding window inference for validation")
    parser.add_argument("--val_patch_size", type=int, nargs=3, default=[96, 96, 96],
                       help="Patch size for sliding window validation")
    parser.add_argument("--val_overlap", type=float, default=0.5,
                       help="Overlap ratio for sliding window validation")

    # Other parameters
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--save_interval", type=int, default=10, help="Save every N epochs")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of workers")
    parser.add_argument("--use_cache", action="store_true", help="Use MONAI CacheDataset")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="unet-super-resolution", help="W&B project")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_csv_log", action="store_true", help="Disable CSV logging")

    args = parser.parse_args()

    # Process acquisition_types
    acquisition_types = args.acquisition_types
    if acquisition_types and len(acquisition_types) == 1 and acquisition_types[0].lower() == "all":
        acquisition_types = None

    # Validate orientation dropout
    if args.drop_orientations is not None and len(args.drop_orientations) > 0:
        if args.orientation_dropout_prob > 0.0:
            print("WARNING: Both --drop_orientations and --orientation_dropout_prob specified. "
                  "Using deterministic dropout (--drop_orientations).")
        if len(args.drop_orientations) >= 3:
            raise ValueError("Cannot drop all 3 orientations.")

    # Get image paths
    hr_image_paths = get_image_paths(
        image_dir=args.hr_image_dir,
        csv_file=args.csv_file,
        base_dir=args.base_dir,
        split="train",
        model_dir=args.model_dir,
        mri_classifications=args.mri_classes,
        acquisition_types=acquisition_types,
        filter_4d=not args.no_filter_4d,
    )

    val_image_paths = None
    if args.val_image_dir or args.csv_file:
        val_image_paths = get_image_paths(
            image_dir=args.val_image_dir,
            csv_file=args.csv_file,
            base_dir=args.base_dir,
            split="val",
            model_dir=args.model_dir,
            mri_classifications=args.mri_classes,
            acquisition_types=acquisition_types,
            filter_4d=not args.no_filter_4d,
        )

    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)

    # Save configuration
    save_training_config(
        model_dir=args.model_dir,
        args=args,
        n_train_samples=len(hr_image_paths),
        n_val_samples=len(val_image_paths) if val_image_paths else 0,
        training_stage="unet",
    )

    # Train model
    train_unet_model(
        hr_image_paths=hr_image_paths,
        model_dir=args.model_dir,
        use_stacks=args.use_stacks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_shape=tuple(args.output_shape),
        checkpoint=args.checkpoint,
        device=args.device,
        save_interval=args.save_interval,
        val_interval=args.val_interval,
        val_image_paths=val_image_paths,
        atlas_res=args.atlas_res,
        min_resolution=args.min_resolution,
        max_res_aniso=args.max_res_aniso,
        randomise_res=not args.no_randomise_res,
        prob_motion=args.prob_motion,
        prob_spike=args.prob_spike,
        prob_aliasing=args.prob_aliasing,
        prob_bias_field=args.prob_bias_field,
        prob_noise=args.prob_noise,
        apply_intensity_aug=not args.no_intensity_aug,
        orientation_dropout_prob=args.orientation_dropout_prob,
        min_orientations=args.min_orientations,
        drop_orientations=args.drop_orientations,
        num_workers=args.num_workers,
        use_cache=args.use_cache,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        nb_features=args.nb_features,
        nb_levels=args.nb_levels,
        conv_size=args.conv_size,
        final_activation=args.final_activation,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        l1_weight=args.l1_weight,
        ssim_weight=args.ssim_weight,
        lpips_weight=args.lpips_weight,
        use_lpips=args.use_lpips,
        lpips_backend=args.lpips_backend,
        use_sliding_window_val=args.use_sliding_window_val,
        val_patch_size=tuple(args.val_patch_size),
        val_overlap=args.val_overlap,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        early_stopping_patience=args.early_stopping_patience,
        csv_log=not args.no_csv_log,
        upsample_mode=args.upsample_mode,
    )
