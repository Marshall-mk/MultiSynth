"""
Training script for U-HVED Super-Resolution

This script provides utilities for training the U-HVED model on super-resolution tasks.
The key idea is to force the model to learn the same feature representation for
different input modalities (degradation types).
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple, Optional
import random
from pathlib import Path

from models import UHVED, UHVEDLoss


class DegradationGenerator:
    """
    Generates different degradation types to create multiple modalities
    from a single high-resolution image.

    For super-resolution, modalities can be:
    - Different blur kernels (Gaussian, motion, etc.)
    - Different noise levels
    - Different downsampling factors
    - Different compression artifacts
    """

    def __init__(
        self,
        degradation_types: List[str] = None,
        scale_factor: int = 4
    ):
        """
        Args:
            degradation_types: List of degradation types to apply
            scale_factor: Downsampling factor
        """
        if degradation_types is None:
            degradation_types = ['bicubic', 'gaussian_blur', 'noise', 'jpeg']

        self.degradation_types = degradation_types
        self.scale_factor = scale_factor

    def bicubic_downsample(self, img: torch.Tensor) -> torch.Tensor:
        """Simple bicubic downsampling."""
        h, w = img.shape[-2:]
        new_h, new_w = h // self.scale_factor, w // self.scale_factor
        return torch.nn.functional.interpolate(
            img.unsqueeze(0) if img.dim() == 3 else img,
            size=(new_h, new_w),
            mode='bicubic',
            align_corners=False
        ).squeeze(0) if img.dim() == 3 else torch.nn.functional.interpolate(
            img, size=(new_h, new_w), mode='bicubic', align_corners=False
        )

    def add_gaussian_blur(self, img: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
        """Add Gaussian blur before downsampling."""
        kernel_size = int(4 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create Gaussian kernel
        x = torch.arange(kernel_size, dtype=img.dtype, device=img.device) - kernel_size // 2
        kernel_1d = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)

        # Apply blur
        if img.dim() == 3:
            img = img.unsqueeze(0)
        c = img.shape[1]
        kernel_2d = kernel_2d.repeat(c, 1, 1, 1)
        padding = kernel_size // 2
        blurred = torch.nn.functional.conv2d(img, kernel_2d, padding=padding, groups=c)

        return self.bicubic_downsample(blurred.squeeze(0))

    def add_noise(self, img: torch.Tensor, std: float = 0.05) -> torch.Tensor:
        """Add Gaussian noise after downsampling."""
        lr = self.bicubic_downsample(img)
        noise = torch.randn_like(lr) * std
        return torch.clamp(lr + noise, -1, 1)

    def add_jpeg_artifacts(self, img: torch.Tensor, quality: int = 50) -> torch.Tensor:
        """Simulate JPEG compression artifacts (simplified)."""
        # This is a simplified version - for real JPEG artifacts,
        # use torchvision.io or PIL
        lr = self.bicubic_downsample(img)
        # Simple approximation: add block artifacts
        block_size = 8
        if lr.dim() == 3:
            lr = lr.unsqueeze(0)
        b, c, h, w = lr.shape

        # Quantization simulation
        quantization_factor = (100 - quality) / 100 * 0.1
        noise = torch.randn(b, c, h // block_size, w // block_size, device=lr.device)
        noise = torch.nn.functional.interpolate(
            noise, size=(h, w), mode='nearest'
        ) * quantization_factor

        return torch.clamp((lr + noise).squeeze(0), -1, 1)

    def __call__(self, hr_img: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate multiple degraded versions of the input image.

        Args:
            hr_img: High-resolution image tensor (C, H, W) or (B, C, H, W)

        Returns:
            List of degraded images (modalities)
        """
        modalities = []

        for deg_type in self.degradation_types:
            if deg_type == 'bicubic':
                lr = self.bicubic_downsample(hr_img)
            elif deg_type == 'gaussian_blur':
                sigma = random.uniform(0.5, 2.5)
                lr = self.add_gaussian_blur(hr_img, sigma)
            elif deg_type == 'noise':
                std = random.uniform(0.02, 0.1)
                lr = self.add_noise(hr_img, std)
            elif deg_type == 'jpeg':
                quality = random.randint(30, 80)
                lr = self.add_jpeg_artifacts(hr_img, quality)
            else:
                lr = self.bicubic_downsample(hr_img)

            modalities.append(lr)

        return modalities


class SyntheticSRDataset(Dataset):
    """
    Synthetic dataset for super-resolution training.

    Creates multiple degraded versions (modalities) from high-resolution images
    to train the U-HVED model.
    """

    def __init__(
        self,
        image_dir: str,
        scale_factor: int = 4,
        patch_size: int = 256,
        num_modalities: int = 4,
        degradation_types: List[str] = None,
        transform=None
    ):
        """
        Args:
            image_dir: Directory containing HR images
            scale_factor: Downsampling factor
            patch_size: Size of HR patches to extract
            num_modalities: Number of degraded versions to generate
            degradation_types: Types of degradation to apply
            transform: Additional transforms
        """
        self.image_dir = Path(image_dir)
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.transform = transform

        # Find all images
        self.image_paths = list(self.image_dir.glob('*.png')) + \
                          list(self.image_dir.glob('*.jpg')) + \
                          list(self.image_dir.glob('*.jpeg'))

        if degradation_types is None:
            degradation_types = ['bicubic', 'gaussian_blur', 'noise', 'jpeg'][:num_modalities]

        self.degradation = DegradationGenerator(degradation_types, scale_factor)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with:
                - 'hr': High-resolution target
                - 'modalities': List of degraded inputs
                - 'mask': Modality presence mask
        """
        # Load image (placeholder - implement actual loading)
        # For demonstration, create random tensor
        hr = torch.randn(1, self.patch_size, self.patch_size)

        # Generate degraded modalities
        modalities = self.degradation(hr)

        # Random modality dropout during training
        mask = torch.ones(len(modalities), dtype=torch.bool)
        if self.training:
            # Randomly drop some modalities (but keep at least one)
            num_keep = random.randint(1, len(modalities))
            drop_indices = random.sample(range(len(modalities)), len(modalities) - num_keep)
            for idx in drop_indices:
                mask[idx] = False

        return {
            'hr': hr,
            'modalities': modalities,
            'mask': mask
        }

    @property
    def training(self) -> bool:
        return True  # Override in actual implementation


class Trainer:
    """
    Training manager for U-HVED Super-Resolution.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        loss_fn: Optional[nn.Module] = None,
        device: str = 'cuda',
        log_dir: str = 'runs',
        checkpoint_dir: str = 'checkpoints'
    ):
        """
        Args:
            model: U-HVED model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer (default: Adam)
            scheduler: Learning rate scheduler
            loss_fn: Loss function (default: UHVEDLoss)
            device: Device to train on
            log_dir: TensorBoard log directory
            checkpoint_dir: Checkpoint save directory
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=1e-4)
        self.scheduler = scheduler
        self.loss_fn = loss_fn or UHVEDLoss()

        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir)
        self.global_step = 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {'total': 0, 'recon': 0, 'kl': 0, 'perceptual': 0}
        num_batches = 0

        for batch in self.train_loader:
            # Move data to device
            hr = batch['hr'].to(self.device)
            modalities = [m.to(self.device) for m in batch['modalities']]
            mask = batch['mask'].to(self.device) if 'mask' in batch else None

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(modalities, mask)

            # Compute loss
            losses = self.loss_fn(
                sr_output=outputs['sr_output'],
                sr_target=hr,
                posteriors=outputs['posteriors'],
                modality_outputs=outputs.get('modality_outputs'),
                modality_targets=modalities if outputs.get('modality_outputs') else None,
                return_components=True
            )

            # Backward pass
            losses['total'].backward()
            self.optimizer.step()

            # Accumulate losses
            for k, v in losses.items():
                if k in epoch_losses:
                    epoch_losses[k] += v.item()
            num_batches += 1
            self.global_step += 1

            # Log to tensorboard
            if self.global_step % 100 == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f'train/{k}', v.item(), self.global_step)

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= max(num_batches, 1)

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_losses = {'total': 0, 'recon': 0, 'kl': 0}
        num_batches = 0

        for batch in self.val_loader:
            hr = batch['hr'].to(self.device)
            modalities = [m.to(self.device) for m in batch['modalities']]

            outputs = self.model(modalities, deterministic=True)

            losses = self.loss_fn(
                sr_output=outputs['sr_output'],
                sr_target=hr,
                posteriors=outputs['posteriors'],
                return_components=True
            )

            for k, v in losses.items():
                if k in val_losses:
                    val_losses[k] += v.item()
            num_batches += 1

        for k in val_losses:
            val_losses[k] /= max(num_batches, 1)

        return val_losses

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step
        }

        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)

        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        return checkpoint.get('epoch', 0)

    def train(self, num_epochs: int, start_epoch: int = 0):
        """Full training loop."""
        best_val_loss = float('inf')

        for epoch in range(start_epoch, num_epochs):
            # Train
            train_losses = self.train_epoch(epoch)
            print(f"Epoch {epoch}: Train Loss = {train_losses['total']:.4f}")

            # Validate
            val_losses = self.validate()
            if val_losses:
                print(f"Epoch {epoch}: Val Loss = {val_losses['total']:.4f}")

                # Log validation
                for k, v in val_losses.items():
                    self.writer.add_scalar(f'val/{k}', v, epoch)

                # Save best model
                is_best = val_losses['total'] < best_val_loss
                if is_best:
                    best_val_loss = val_losses['total']
                self.save_checkpoint(epoch, is_best)
            else:
                self.save_checkpoint(epoch)

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train U-HVED for Super-Resolution')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to HR images')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_modalities', type=int, default=4, help='Number of modalities')
    parser.add_argument('--scale_factor', type=int, default=4, help='Super-resolution scale')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    args = parser.parse_args()

    # Create model
    model = UHVED(
        num_modalities=args.num_modalities,
        in_channels=3,
        out_channels=3,
        base_channels=32,
        num_scales=4
    )

    # Create dataset
    train_dataset = SyntheticSRDataset(
        image_dir=args.data_dir,
        scale_factor=args.scale_factor,
        num_modalities=args.num_modalities
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        device=args.device
    )

    # Resume if specified
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(args.epochs, start_epoch)


if __name__ == '__main__':
    main()
