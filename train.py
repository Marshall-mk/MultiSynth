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


def calculate_metrics(pred, target, max_val=1.0):
    """Calculate regression metrics."""
    with torch.no_grad():
        mae = torch.mean(torch.abs(pred - target)).item()
        mse = torch.mean((pred - target) ** 2).item()
        rmse = torch.sqrt(torch.tensor(mse)).item()

        # PSNR
        if mse > 0:
            psnr = 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(torch.tensor(mse))).item()
        else:
            psnr = 100.0

        # R-squared
        ss_res = torch.sum((target - pred) ** 2).item()
        ss_tot = torch.sum((target - torch.mean(target)) ** 2).item()
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        # SSIM (simplified)
        ssim = 1.0 - mae / max_val  # Placeholder

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'psnr': psnr,
            'r2': r2,
            'ssim': ssim
        }


def get_image_paths(image_dir, csv_file=None, base_dir=None, split="train",
                   model_dir=None, mri_classifications=None):
    """Get image paths from directory or CSV file."""
    import glob

    if csv_file and os.path.exists(csv_file):
        # Load from CSV
        import pandas as pd
        df = pd.read_csv(csv_file)

        # Filter by split
        if 'split' in df.columns:
            df = df[df['split'] == split]

        # Filter by MRI classification if specified
        if mri_classifications and 'mri_classification' in df.columns:
            df = df[df['mri_classification'].isin(mri_classifications)]

        # Get paths
        if 'path' in df.columns:
            paths = df['path'].tolist()
            if base_dir:
                paths = [os.path.join(base_dir, p) if not os.path.isabs(p) else p for p in paths]
            return paths

    # Load from directory
    if image_dir and os.path.exists(image_dir):
        extensions = ['*.nii', '*.nii.gz', '*.mgz', '*.mgh']
        paths = []
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))
        return sorted(paths)

    return []


def save_training_config(model_dir, args, n_train_samples, n_val_samples, training_stage):
    """Save training configuration to JSON."""
    import json

    config = {
        'training_stage': training_stage,
        'model_architecture': 'uhved',
        'num_modalities': 3,  # Fixed: axial, coronal, sagittal
        'n_train_samples': n_train_samples,
        'n_val_samples': n_val_samples,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'output_shape': args.output_shape,
        'atlas_res': args.atlas_res,
        'min_resolution': args.min_resolution,
        'max_res_aniso': args.max_res_aniso,
    }

    config_path = os.path.join(model_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Training configuration saved to: {config_path}")


def find_latest_checkpoint(model_dir, model_type="uhved"):
    """Find the latest checkpoint in the model directory."""
    checkpoints = glob.glob(os.path.join(model_dir, f"{model_type}_*_epoch_*.pth"))
    if not checkpoints:
        return None

    # Sort by epoch number
    def get_epoch(path):
        try:
            return int(path.split('epoch_')[-1].split('.pth')[0])
        except:
            return -1

    checkpoints.sort(key=get_epoch)
    return checkpoints[-1] if checkpoints else None


def save_model_checkpoint(filepath, model, optimizer, epoch, loss, val_loss=None,
                          model_type="uhved", model_config=None, scheduler_state_dict=None):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'val_loss': val_loss,
        'model_type': model_type,
        'model_config': model_config or {},
    }

    if scheduler_state_dict:
        checkpoint['scheduler_state_dict'] = scheduler_state_dict

    torch.save(checkpoint, filepath)


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
    kl_weight: float = 0.001,
    perceptual_weight: float = 0.0,
    modality_weight: float = 0.5,
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
        kl_weight: Weight for KL divergence loss
        perceptual_weight: Weight for perceptual loss
        modality_weight: Weight for modality reconstruction loss
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
        reconstruction_weight=1.0,
        kl_weight=kl_weight,
        perceptual_weight=perceptual_weight,
        modality_weight=modality_weight,
        use_perceptual=perceptual_weight > 0,
    )

    if accelerator.is_main_process:
        print(f"Loss weights: recon=1.0, kl={kl_weight}, perceptual={perceptual_weight}, modality={modality_weight}")

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
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Training for {epochs} epochs, batch size {batch_size}")
        print(f"Initial learning rate: {learning_rate}")
        print(f"Device: {accelerator.device}")

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

        csv_headers = ["epoch", "train_loss", "train_recon", "train_kl", "learning_rate", "epoch_time"]
        if val_dataloader:
            csv_headers.extend(["val_loss", "val_mae", "val_psnr", "validation_time"])

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

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", disable=not accelerator.is_main_process)

        for batch_idx, batch_data in enumerate(pbar):
            # Unpack batch: (lr_stacks_list, hr, resolutions_list, thicknesses_list)
            lr_stacks_list, target_img, resolutions_list, thicknesses_list = batch_data

            # lr_stacks_list is a list of 3 tensors, each (B, C, D, H, W)
            # We need to convert to the format U-HVED expects: list of (B, C, D, H, W)
            modalities = lr_stacks_list  # Already in correct format

            # Ensure consistent dtype
            modalities = [m.float() for m in modalities]
            target_img = target_img.float()

            # Forward and backward pass
            with accelerator.accumulate(model):
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

            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{losses['reconstruction'].item():.4f}",
                "kl": f"{losses['kl'].item():.4f}",
                "lr": f"{current_lr:.2e}"
            })

        avg_loss = epoch_loss / len(dataloader)
        avg_recon = epoch_recon_loss / len(dataloader)
        avg_kl = epoch_kl_loss / len(dataloader)

        # Validation
        val_loss = None
        val_metrics = None
        validation_time = 0.0

        if val_dataloader and (epoch + 1) % val_interval == 0:
            val_start_time = time.time()
            model.eval()
            val_epoch_loss = 0.0
            metrics_sum = {"mae": 0.0, "mse": 0.0, "psnr": 0.0}
            num_val_batches = 0

            with torch.no_grad():
                for val_batch_data in val_dataloader:
                    lr_stacks_list, target_img, _, _ = val_batch_data
                    modalities = [m.float() for m in lr_stacks_list]
                    target_img = target_img.float()

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
            if accelerator.is_main_process:
                print(
                    f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_loss:.4f} "
                    f"(Recon: {avg_recon:.4f}, KL: {avg_kl:.6f}) - Val Loss: {val_loss:.4f}"
                )
                print(
                    f"  Val Metrics - MAE: {val_metrics['mae']:.4f} | "
                    f"PSNR: {val_metrics['psnr']:.2f} dB | LR: {current_lr:.2e}"
                )
        else:
            epoch_time = time.time() - epoch_start_time
            if accelerator.is_main_process:
                print(
                    f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} "
                    f"(Recon: {avg_recon:.4f}, KL: {avg_kl:.6f}) - LR: {current_lr:.2e}"
                )

        # Log to CSV
        if accelerator.is_main_process and csv_writer is not None:
            log_data = {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_recon": avg_recon,
                "train_kl": avg_kl,
                "learning_rate": current_lr,
                "epoch_time": epoch_time,
            }
            if val_loss is not None and val_metrics is not None:
                log_data.update({
                    "val_loss": val_loss,
                    "val_mae": val_metrics['mae'],
                    "val_psnr": val_metrics['psnr'],
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
                "train/learning_rate": current_lr,
            }
            if val_loss is not None and val_metrics is not None:
                wandb_log_data.update({
                    "val/loss": val_loss,
                    "val/mae": val_metrics['mae'],
                    "val/psnr": val_metrics['psnr'],
                })
            wandb.log(wandb_log_data)

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
                )
                print(f"Saved checkpoint: {checkpoint_path}")

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

        unwrapped_model = accelerator.unwrap_model(model)
        save_model_checkpoint(
            filepath=final_path,
            model=unwrapped_model,
            optimizer=optimizer,
            epoch=epochs - 1,
            model_type="uhved",
            model_config=model_config,
            scheduler_state_dict=scheduler.state_dict(),
        )
        print(f"Training complete! Final model saved to: {final_path}")

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

    # Model parameters
    parser.add_argument("--base_channels", type=int, default=32, help="Base channels for U-HVED")
    parser.add_argument("--num_scales", type=int, default=4, help="Number of hierarchical scales")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output_shape", type=int, nargs=3, default=[128, 128, 128], help="Output shape")

    # Loss weights
    parser.add_argument("--kl_weight", type=float, default=0.001, help="KL divergence weight")
    parser.add_argument("--perceptual_weight", type=float, default=0.0, help="Perceptual loss weight")
    parser.add_argument("--modality_weight", type=float, default=0.5, help="Modality reconstruction weight")

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

    # Other parameters
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--save_interval", type=int, default=10, help="Save every N epochs")
    parser.add_argument("--val_interval", type=int, default=1, help="Validate every N epochs")
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

    # Get image paths
    hr_image_paths = get_image_paths(
        image_dir=args.hr_image_dir,
        csv_file=args.csv_file,
        base_dir=args.base_dir,
        split="train",
        model_dir=args.model_dir,
    )

    val_image_paths = None
    if args.val_image_dir or args.csv_file:
        val_image_paths = get_image_paths(
            image_dir=args.val_image_dir,
            csv_file=args.csv_file,
            base_dir=args.base_dir,
            split="val",
            model_dir=args.model_dir,
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
        kl_weight=args.kl_weight,
        perceptual_weight=args.perceptual_weight,
        modality_weight=args.modality_weight,
    )
