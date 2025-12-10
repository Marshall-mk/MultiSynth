"""
Training script for U-HVED Super-Resolution with Orthogonal Stacks

This script trains the U-HVED model using three orthogonal low-resolution stacks
generated from high-resolution volumes. Each stack has high resolution in one
orientation (axial, coronal, sagittal).

Based on the original SynthSR training pipeline with modifications for U-HVED.
"""

import os
import argparse
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from datetime import datetime
from tqdm import tqdm
from typing import List
import wandb

# Accelerate for easy multi-GPU training
from accelerate import Accelerator
from accelerate.utils import set_seed

# Transformers imports for scheduler
from transformers import get_cosine_schedule_with_warmup

# MONAI imports
from monai.data import DataLoader

# Import our modules
from src import UHVED, UHVEDLoss, create_uhved
from src.data import HRLRDataGenerator, create_dataset
from src.utils import (
    save_model_checkpoint,
    get_image_paths,
    save_training_config,
    find_latest_checkpoint,
    calculate_metrics,
    sliding_window_inference,
    print_model_statistics,
    get_gpu_memory_stats,
    print_gpu_memory_stats,
)


def train_uhved_model(
    hr_image_paths: List[str],
    model_dir: str,
    epochs: int = 100,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    output_shape: tuple = (128, 128, 128),
    checkpoint: str = None,
    device: str = "cuda",
    save_interval: int = 10,
    val_interval: int = 1,
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
    modality_dropout_prob: float = 0.0,
    min_modalities: int = 1,
    num_workers: int = None,
    use_cache: bool = False,
    use_wandb: bool = False,
    wandb_project: str = "uhved-sr",
    wandb_entity: str = None,
    wandb_run_name: str = None,
    base_channels: int = 32,
    num_scales: int = 4,
    mixed_precision: str = "no",
    gradient_accumulation_steps: int = 1,
    seed: int = 42,
    recon_loss_type: str = "l1",
    recon_weight: float = 0.4,
    kl_weight: float = 0.1,
    perceptual_weight: float = 0.1,
    ssim_weight: float = 0.1,
    modality_weight: float = 0.4,
    use_perceptual: bool = False,
    use_ssim: bool = True,
    perceptual_backend: str = 'medicalnet',
    use_sliding_window_val: bool = False,
    val_patch_size: tuple = (128, 128, 128),
    val_overlap: float = 0.5,
):
    """
    Train U-HVED model with orthogonal LR stacks

    Args:
        hr_image_paths: List of paths to high-resolution images
        model_dir: Directory to save trained models
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
        modality_dropout_prob: Probability of applying modality dropout (0.0-1.0)
        min_modalities: Minimum number of modalities to keep after dropout (1-3)
        num_workers: Number of data loading workers
        use_cache: Whether to use CacheDataset
        use_wandb: Whether to use Weights & Biases for tracking
        wandb_project: W&B project name
        wandb_entity: W&B entity/team name
        wandb_run_name: W&B run name
        base_channels: Base number of channels for U-HVED
        num_scales: Number of hierarchical scales
        mixed_precision: Mixed precision training ('no', 'fp16', 'bf16')
        gradient_accumulation_steps: Number of steps to accumulate gradients
        seed: Random seed for reproducibility
        recon_loss_type: Type of reconstruction loss ('l1', 'l2', or 'charbonnier')
        recon_weight: Weight for reconstruction loss
        kl_weight: Weight for KL divergence loss
        perceptual_weight: Weight for perceptual loss
        ssim_weight: Weight for SSIM loss
        modality_weight: Weight for modality reconstruction loss
        use_perceptual: Whether to use perceptual loss
        use_ssim: Whether to use SSIM loss
        perceptual_backend: Backend for perceptual loss ('medicalnet' or 'monai', 'models_genesis')
        use_sliding_window_val: Use sliding window inference for validation
        val_patch_size: Patch size for sliding window validation
        val_overlap: Overlap ratio for sliding window validation
    """
    # Initialize Accelerator for multi-GPU training
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with="wandb" if use_wandb else None,
    )

    # Set seed for reproducibility
    set_seed(seed)

    # Only print from main process
    if accelerator.is_main_process:
        print("=" * 80)
        print("Training U-HVED with Orthogonal LR Stacks")
        print("=" * 80)
        print(f"Distributed training: {accelerator.num_processes} process(es)")
        print(f"Mixed precision: {mixed_precision}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")

    # Create model directory
    os.makedirs(model_dir, exist_ok=True)

    # Auto-detect optimal num_workers if not provided
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        if device == "cuda" and torch.cuda.is_available():
            num_workers = min(4, max(cpu_count // 2, 1))
        elif cpu_count >= 4:
            num_workers = 2
        else:
            num_workers = 0

    # Enable pin_memory for faster GPU transfer
    pin_memory = False  # Accelerate handles this

    if accelerator.is_main_process:
        print(
            f"DataLoader settings: num_workers={num_workers}, pin_memory={pin_memory}, use_cache={use_cache}"
        )

    # Create data generator for orthogonal stacks
    if accelerator.is_main_process:
        print(f"Input HR image resolution: {atlas_res} mm")
        print(f"Resolution randomization: {min_resolution} to {max_res_aniso} mm")
        print(f"Generating 3 orthogonal LR stacks per HR volume")
        print(f"  - Stack 0: High-res in Axial (D) direction")
        print(f"  - Stack 1: High-res in Coronal (H) direction")
        print(f"  - Stack 2: High-res in Sagittal (W) direction")
        if modality_dropout_prob > 0.0:
            print(f"Modality dropout enabled: {modality_dropout_prob:.2f} probability, min {min_modalities} modalities")
            print(f"  → Training will randomly drop views to simulate missing data")

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
        modality_dropout_prob=modality_dropout_prob,
        min_modalities=min_modalities,
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

    # Create validation dataset if provided
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
        checkpoint = find_latest_checkpoint(model_dir, model_type="uhved")
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

    # Create U-HVED model
    if accelerator.is_main_process:
        print(f"Creating U-HVED model:")
        print(f"  - Number of modalities: 3 (orthogonal stacks)")
        print(f"  - Base channels: {base_channels}")
        print(f"  - Number of scales: {num_scales}")

    model = UHVED(
        num_modalities=3,  # Fixed: axial, coronal, sagittal
        in_channels=1,
        out_channels=1,
        base_channels=base_channels,
        num_scales=num_scales,
        share_encoder=False,  # Independent encoders for each orientation
        share_decoder=False,
        reconstruct_modalities=True,  # Reconstruct input modalities for training
    )

    # Load checkpoint weights if available
    if checkpoint_data is not None:
        model.load_state_dict(checkpoint_data["model_state_dict"])
        if accelerator.is_main_process:
            print(f"✓ Loaded model weights from checkpoint")

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = UHVEDLoss(
        recon_loss_type=recon_loss_type,
        recon_weight=recon_weight,
        kl_weight=kl_weight,
        perceptual_weight=perceptual_weight,
        ssim_weight=ssim_weight,
        modality_weight=modality_weight,
        use_perceptual=use_perceptual,
        use_ssim=use_ssim,
        perceptual_backend=perceptual_backend,
    )

    if accelerator.is_main_process:
        print(f"Loss weights: recon={recon_weight}, kl={kl_weight}, perceptual={perceptual_weight}, modality={modality_weight}")

    # Calculate total training steps for scheduler
    num_steps = len(dataloader) * epochs
    warmup_steps = int(0.05 * num_steps)  # 5% warmup

    # Learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_steps,
    )

    if accelerator.is_main_process:
        print(f"Using Cosine LR schedule with warmup ({warmup_steps} warmup steps, {num_steps} total steps)")

    # Load optimizer and scheduler state if resuming
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
        # Print detailed model statistics including memory usage
        # U-HVED expects 3 orthogonal stacks: [axial, coronal, sagittal]
        input_shapes = [
            (batch_size, 1, *output_shape),  # Axial stack
            (batch_size, 1, *output_shape),  # Coronal stack
            (batch_size, 1, *output_shape),  # Sagittal stack
        ]
        print_model_statistics(accelerator.unwrap_model(model), input_shapes, device=str(accelerator.device))
        print(f"\nTraining for {epochs} epochs, batch size {batch_size}")
        print(f"Initial learning rate: {learning_rate}")
        print(f"Device: {accelerator.device}")

    # Track best validation loss for saving best model
    best_val_loss = float('inf')
    if checkpoint_data is not None and 'val_loss' in checkpoint_data and checkpoint_data['val_loss'] is not None:
        best_val_loss = checkpoint_data['val_loss']
        if accelerator.is_main_process:
            print(f"Best validation loss from checkpoint: {best_val_loss:.4f}")

    # Initialize Weights & Biases if enabled
    if use_wandb and accelerator.is_main_process:
        wandb_config = {
            "model": "uhved",
            "num_modalities": 3,
            "base_channels": base_channels,
            "num_scales": num_scales,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "output_shape": output_shape,
            "kl_weight": kl_weight,
            "perceptual_weight": perceptual_weight,
            "modality_weight": modality_weight,
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
        wandb.watch(model, criterion, log="all", log_freq=100)
        print(f"Weights & Biases initialized: {wandb.run.name}")

    # Setup CSV logging
    csv_file = None
    csv_writer = None
    if accelerator.is_main_process:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"training_log_{timestamp}.csv"
        csv_path = os.path.join(model_dir, csv_filename)

        csv_headers = ["epoch", "train_loss", "train_recon", "train_kl", "train_ssim", "train_perceptual", "train_modality",
                        "learning_rate", "epoch_time"]
        if val_dataloader:
            csv_headers.extend(["val_loss", "val_mae", "val_mse", "val_rmse", "val_psnr", "val_r2", "val_ssim", "validation_time"])

        csv_file = open(csv_path, mode='a', newline='')
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
        csv_writer.writeheader()
        csv_file.flush()
        print(f"Logging training metrics to: {csv_path}")

    # Training loop
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_ssim_loss = 0.0
        epoch_perceptual_loss = 0.0
        epoch_modality_loss = 0.0

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            disable=not accelerator.is_main_process,
            leave=False,  # Don't leave progress bar after completion
            dynamic_ncols=True,  # Adjust width dynamically
        )

        for batch_idx, batch_data in enumerate(pbar):
            # Unpack batch: (lr_stacks_list, hr, resolutions_list, thicknesses_list, modality_mask)
            lr_stacks_list, target_img, resolutions_list, thicknesses_list, modality_mask = batch_data

            # lr_stacks_list is a list of 3 tensors, each (B, C, D, H, W)
            # We need to convert to the format U-HVED expects: list of (B, C, D, H, W)
            modalities = lr_stacks_list  # Already in correct format

            # Ensure consistent dtype
            modalities = [m.float() for m in modalities]
            target_img = target_img.float()

            # Forward and backward pass
            with accelerator.accumulate(model):
                # Note: We don't pass modality_mask because we already zeroed out
                # dropped modalities in the data generator. The zeroed modalities
                # will naturally contribute near-zero to the Product of Gaussians fusion.
                outputs = model(modalities)

                # Compute loss
                losses = criterion(
                    sr_output=outputs['sr_output'],
                    sr_target=target_img,
                    posteriors=outputs['posteriors'],
                    modality_outputs=outputs.get('modality_outputs'),
                    modality_targets=modalities,
                    return_components=True
                )

                loss = losses['total']

                # Backward pass
                optimizer.zero_grad()
                accelerator.backward(loss)

                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

            epoch_loss += loss.item()
            epoch_recon_loss += losses['reconstruction'].item()
            epoch_kl_loss += losses['kl'].item()
            epoch_ssim_loss += losses['ssim'].item() if 'ssim' in losses else 0.0
            epoch_perceptual_loss += losses['perceptual'].item() if 'perceptual' in losses else 0.0
            epoch_modality_loss += losses['modality'].item() if 'modality' in losses else 0.0

            # Update progress bar with loss and memory
            current_lr = optimizer.param_groups[0]['lr']
            postfix_dict = {
                "loss": f"{loss.item():.4f}",
                "recon": f"{losses['reconstruction'].item():.4f}",
                "kl": f"{losses['kl'].item():.4f}",
                "ssim": f"{losses['ssim'].item():.4f}" if 'ssim' in losses else "N/A",
                "perceptual": f"{losses['perceptual'].item():.4f}" if 'perceptual' in losses else "N/A",
                "modality": f"{losses['modality'].item():.4f}" if 'modality' in losses else "N/A",
                "lr": f"{current_lr:.2e}"
            }

            # Add GPU memory to progress bar (update every 10 batches to avoid overhead)
            if batch_idx % 10 == 0:
                mem_stats = get_gpu_memory_stats(accelerator.device)
                if mem_stats:
                    postfix_dict["mem"] = f"{mem_stats['allocated_mb']:.0f}MB"

            pbar.set_postfix(postfix_dict)

            # Print actual GPU memory usage after first batch (once per training run)
            if epoch == start_epoch and batch_idx == 0 and accelerator.is_main_process:
                print()  # New line after progress bar
                print_gpu_memory_stats(accelerator.device, prefix="After first batch")

        avg_loss = epoch_loss / len(dataloader)
        avg_recon = epoch_recon_loss / len(dataloader)
        avg_kl = epoch_kl_loss / len(dataloader)
        avg_ssim = epoch_ssim_loss / len(dataloader) if epoch_ssim_loss > 0 else 0.0
        avg_perceptual = epoch_perceptual_loss / len(dataloader) if epoch_perceptual_loss > 0 else 0.0
        avg_modality = epoch_modality_loss / len(dataloader) if epoch_modality_loss > 0 else 0.0

        # Validation
        val_loss = None
        val_metrics = None
        validation_time = 0.0

        if val_dataloader and (epoch + 1) % val_interval == 0:
            val_start_time = time.time()
            model.eval()
            val_epoch_loss = 0.0
            metrics_sum = {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "psnr": 0.0, "r2": 0.0, "ssim": 0.0}
            num_val_batches = 0

            with torch.no_grad():
                for val_batch_data in val_dataloader:
                    lr_stacks_list, target_img, _, _, modality_mask = val_batch_data
                    modalities = [m.float() for m in lr_stacks_list]
                    target_img = target_img.float()

                    # Use sliding window inference if enabled
                    if use_sliding_window_val:
                        # For sliding window, process on main device
                        sr_output = sliding_window_inference(
                            model=accelerator.unwrap_model(model),
                            modalities=modalities,
                            patch_size=val_patch_size,
                            overlap=val_overlap,
                            batch_size=1,
                            device=accelerator.device,
                            blend_mode="gaussian",
                            progress=False,
                        )
                        outputs = {'sr_output': sr_output, 'posteriors': None}
                    else:
                        # Note: We don't pass modality_mask - zeroed modalities
                        # naturally contribute near-zero to the fusion
                        outputs = model(modalities)

                    losses = criterion(
                        sr_output=outputs['sr_output'],
                        sr_target=target_img,
                        posteriors=outputs['posteriors'],
                        return_components=True
                    )

                    val_epoch_loss += losses['total'].item()

                    # Calculate metrics
                    batch_metrics = calculate_metrics(outputs['sr_output'], target_img, max_val=1.0)
                    for key in metrics_sum:
                        metrics_sum[key] += batch_metrics[key]
                    num_val_batches += 1

            val_loss = val_epoch_loss / num_val_batches
            val_metrics = {k: v / num_val_batches for k, v in metrics_sum.items()}
            validation_time = time.time() - val_start_time

            epoch_time = time.time() - epoch_start_time
            # Print validation results (use accelerator.print to avoid tqdm interference)
            accelerator.print(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_loss:.4f} "
                f"(Recon: {avg_recon:.4f}, KL: {avg_kl:.6f}, SSIM: {avg_ssim:.4f}, Percep: {avg_perceptual:.4f}, Modality: {avg_modality:.4f}) - Val Loss: {val_loss:.4f}"
            )
            accelerator.print(
                f"  Val Metrics - MAE: {val_metrics['mae']:.4f} | RMSE: {val_metrics['rmse']:.4f} | "
                f"PSNR: {val_metrics['psnr']:.2f} dB | SSIM: {val_metrics['ssim']:.4f} | "
                f"R²: {val_metrics['r2']:.4f} | LR: {current_lr:.2e}"
            )
        else:
            epoch_time = time.time() - epoch_start_time
            # Print training summary (use accelerator.print to avoid tqdm interference)
            accelerator.print(
                f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} "
                f"(Recon: {avg_recon:.4f}, KL: {avg_kl:.6f}, SSIM: {avg_ssim:.4f}, Percep: {avg_perceptual:.4f}, Modality: {avg_modality:.4f}) - LR: {current_lr:.2e}"
            )

        # Log to CSV
        if accelerator.is_main_process and csv_writer is not None:
            log_data = {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_recon": avg_recon,
                "train_kl": avg_kl,
                "train_ssim": avg_ssim,
                "train_perceptual": avg_perceptual,
                "train_modality": avg_modality,
                "learning_rate": current_lr,
                "epoch_time": epoch_time,
            }
            if val_loss is not None and val_metrics is not None:
                log_data.update({
                    "val_loss": val_loss,
                    "val_mae": val_metrics['mae'],
                    "val_mse": val_metrics['mse'],
                    "val_rmse": val_metrics['rmse'],
                    "val_psnr": val_metrics['psnr'],
                    "val_r2": val_metrics['r2'],
                    "val_ssim": val_metrics['ssim'],
                    "validation_time": validation_time,
                })
            csv_writer.writerow(log_data)
            csv_file.flush()

        # Log to W&B
        if use_wandb and accelerator.is_main_process:
            wandb_log_data = {
                "epoch": epoch + 1,
                "train/loss": avg_loss,
                "train/reconstruction": avg_recon,
                "train/kl": avg_kl,
                "train/ssim": avg_ssim,
                "train/perceptual": avg_perceptual,
                "train/modality": avg_modality,
                "train/learning_rate": current_lr,
            }
            if val_loss is not None and val_metrics is not None:
                wandb_log_data.update({
                    "val/loss": val_loss,
                    "val/mae": val_metrics['mae'],
                    "val/mse": val_metrics['mse'],
                    "val/rmse": val_metrics['rmse'],
                    "val/psnr": val_metrics['psnr'],
                    "val/r2": val_metrics['r2'],
                    "val/ssim": val_metrics['ssim'],
                })
            wandb.log(wandb_log_data)

        # Save best model if validation loss improved
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                best_model_path = os.path.join(model_dir, "uhved_orthogonal_best.pth")

                model_config = {
                    "model_architecture": "uhved",
                    "num_modalities": 3,
                    "base_channels": base_channels,
                    "num_scales": num_scales,
                    "output_shape": output_shape,
                }

                training_config = {
                    "learning_rate": learning_rate,
                    "kl_weight": kl_weight,
                    "recon_weight": recon_weight,
                    "ssim_weight": ssim_weight,
                    "perceptual_weight": perceptual_weight,
                    "modality_weight": modality_weight,
                }

                unwrapped_model = accelerator.unwrap_model(model)
                save_model_checkpoint(
                    filepath=best_model_path,
                    model=unwrapped_model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=avg_loss,
                    val_loss=val_loss,
                    model_type="uhved",
                    model_config=model_config,
                    scheduler_state_dict=scheduler.state_dict(),
                    val_metrics=val_metrics,
                    training_config=training_config,
                )
                accelerator.print(f"✓ Saved best model (val_loss: {val_loss:.4f}): {best_model_path}")

        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                checkpoint_path = os.path.join(
                    model_dir, f"uhved_orthogonal_epoch_{epoch + 1:04d}.pth"
                )

                model_config = {
                    "model_architecture": "uhved",
                    "num_modalities": 3,
                    "base_channels": base_channels,
                    "num_scales": num_scales,
                    "output_shape": output_shape,
                }

                training_config = {
                    "learning_rate": learning_rate,
                    "kl_weight": kl_weight,
                    "perceptual_weight": perceptual_weight,
                    "modality_weight": modality_weight,
                }

                unwrapped_model = accelerator.unwrap_model(model)
                save_model_checkpoint(
                    filepath=checkpoint_path,
                    model=unwrapped_model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=avg_loss,
                    val_loss=val_loss,
                    model_type="uhved",
                    model_config=model_config,
                    scheduler_state_dict=scheduler.state_dict(),
                    val_metrics=val_metrics,
                    training_config=training_config,
                )
                accelerator.print(f"Saved checkpoint: {checkpoint_path}")

    # Close CSV file
    if accelerator.is_main_process and csv_file is not None:
        csv_file.close()

    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_path = os.path.join(model_dir, "uhved_orthogonal_final.pth")
        model_config = {
            "model_architecture": "uhved",
            "num_modalities": 3,
            "base_channels": base_channels,
            "num_scales": num_scales,
            "output_shape": output_shape,
        }

        training_config = {
            "learning_rate": learning_rate,
            "kl_weight": kl_weight,
            "perceptual_weight": perceptual_weight,
            "modality_weight": modality_weight,
        }

        unwrapped_model = accelerator.unwrap_model(model)
        save_model_checkpoint(
            filepath=final_path,
            model=unwrapped_model,
            optimizer=optimizer,
            epoch=epochs - 1,
            loss=avg_loss,
            val_loss=val_loss,
            model_type="uhved",
            model_config=model_config,
            scheduler_state_dict=scheduler.state_dict(),
            val_metrics=val_metrics,
            training_config=training_config,
        )
        print(f"Training complete! Final model saved to: {final_path}")
        if best_val_loss < float('inf'):
            print(f"Best validation loss: {best_val_loss:.4f}")

        # Print peak memory usage summary
        print()
        print_gpu_memory_stats(accelerator.device, prefix="Training Complete - Peak Memory")

        if use_wandb:
            artifact = wandb.Artifact(
                name=f"model-{wandb.run.id}",
                type="model",
                description="Final trained U-HVED model with orthogonal stacks",
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

    parser = argparse.ArgumentParser(description="Train U-HVED with Orthogonal LR Stacks")

    # Data arguments
    parser.add_argument("--hr_image_dir", type=str, default=None, help="Directory containing HR images")
    parser.add_argument("--csv_file", type=str, default=None, help="CSV file with image metadata")
    parser.add_argument("--base_dir", type=str, default=None, help="Base directory for CSV paths")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory to save models")
    parser.add_argument("--val_image_dir", type=str, default=None, help="Validation images directory")
    parser.add_argument("--mri_classes", type=str, nargs="+", default=None,
                       help="MRI classifications to include (e.g., T1 T2 FLAIR). Only for CSV mode")
    parser.add_argument("--acquisition_types", type=str, nargs="+", default=["3D"],
                       help="Acquisition types to include (e.g., 3D 2D). Use 'all' for all types. Default: 3D only")
    parser.add_argument("--no_filter_4d", action="store_true",
                       help="Don't filter out 4D images (with time dimension)")

    # Model parameters
    parser.add_argument("--base_channels", type=int, default=32, help="Base channels for U-HVED")
    parser.add_argument("--num_scales", type=int, default=4, help="Number of hierarchical scales")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output_shape", type=int, nargs=3, default=[128, 128, 128], help="Output shape")

    # Loss weights
    parser.add_argument("--recon_loss_type", type=str, default="l1", choices=["l1", "l2", "charbonnier"],
                        help="Reconstruction loss type")
    parser.add_argument("--recon_weight", type=float, default=0.4, help="Reconstruction loss weight")
    parser.add_argument("--kl_weight", type=float, default=0.1, help="KL divergence weight")
    parser.add_argument("--perceptual_weight", type=float, default=0.1, help="Perceptual loss weight")
    parser.add_argument("--ssim_weight", type=float, default=0.1, help="SSIM loss weight")
    parser.add_argument("--modality_weight", type=float, default=0.4, help="Modality reconstruction weight")
    parser.add_argument("--use_perceptual", action="store_true", help="Use perceptual loss")
    parser.add_argument("--use_ssim", action="store_true", help="Use SSIM loss")
    parser.add_argument("--perceptual_backend", type=str, default="medicalnet",
                        choices=["medicalnet", "monai", "models_genesis"], help="Perceptual loss backend")

    # Data generation parameters
    parser.add_argument("--atlas_res", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="HR resolution")
    parser.add_argument("--min_resolution", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Min resolution")
    parser.add_argument("--max_res_aniso", type=float, nargs=3, default=[9.0, 9.0, 9.0], help="Max aniso resolution")
    parser.add_argument("--no_randomise_res", action="store_true", help="Disable resolution randomization")
    parser.add_argument("--prob_motion", type=float, default=0.5, help="Probability of motion artifacts")
    parser.add_argument("--prob_spike", type=float, default=0.5, help="Probability of k-space spikes")
    parser.add_argument("--prob_aliasing", type=float, default=0.02, help="Probability of aliasing")
    parser.add_argument("--prob_bias_field", type=float, default=0.5, help="Probability of bias field")
    parser.add_argument("--prob_noise", type=float, default=0.8, help="Probability of noise")
    parser.add_argument("--no_intensity_aug", action="store_true", help="Disable intensity augmentation")

    # Modality dropout (for robust training with missing views)
    parser.add_argument("--modality_dropout_prob", type=float, default=0.0,
                        help="Probability of applying modality dropout (0.0-1.0). "
                             "Randomly drops 1-2 orthogonal views to simulate missing data during inference. "
                             "Default: 0.0 (no dropout)")
    parser.add_argument("--min_modalities", type=int, default=1,
                        help="Minimum number of modalities to keep after dropout (1-3). "
                             "Default: 1 (allows training with single views)")

    # Other parameters
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--save_interval", type=int, default=10, help="Save every N epochs")
    parser.add_argument("--val_interval", type=int, default=1, help="Validate every N epochs")
    parser.add_argument("--use_sliding_window_val", action="store_true",
                       help="Use sliding window inference for validation (slower but more accurate for large volumes)")
    parser.add_argument("--val_patch_size", type=int, nargs=3, default=[128, 128, 128],
                       help="Patch size for sliding window validation")
    parser.add_argument("--val_overlap", type=float, default=0.5,
                       help="Overlap ratio for sliding window validation (0.0-1.0)")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of workers")
    parser.add_argument("--use_cache", action="store_true", help="Use MONAI CacheDataset")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="uhved-sr", help="W&B project")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Process acquisition_types: convert "all" to None for the function
    acquisition_types = args.acquisition_types
    if acquisition_types and len(acquisition_types) == 1 and acquisition_types[0].lower() == "all":
        acquisition_types = None

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
        training_stage="uhved",
    )

    # Train model
    train_uhved_model(
        hr_image_paths=hr_image_paths,
        model_dir=args.model_dir,
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
        modality_dropout_prob=args.modality_dropout_prob,
        min_modalities=args.min_modalities,
        num_workers=args.num_workers,
        use_cache=args.use_cache,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        base_channels=args.base_channels,
        num_scales=args.num_scales,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        recon_loss_type=args.recon_loss_type,
        recon_weight=args.recon_weight,
        kl_weight=args.kl_weight,
        perceptual_weight=args.perceptual_weight,
        ssim_weight=args.ssim_weight,
        modality_weight=args.modality_weight,
        use_perceptual=args.use_perceptual,
        use_ssim=args.use_ssim,
        perceptual_backend=args.perceptual_backend,
        use_sliding_window_val=args.use_sliding_window_val,
        val_patch_size=tuple(args.val_patch_size),
        val_overlap=args.val_overlap,
    )
