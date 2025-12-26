"""
Evaluation script for U-HVED Medical Image Super-Resolution

This script evaluates the U-HVED model on test datasets, computing comprehensive
metrics and supporting both HR-only and paired HR/LR data modes.

Features:
- Two data modes: HR volumes only (generate LR) or paired HR/LR stacks
- Multiple output formats: CSV, JSON, console summary, NIfTI files
- Per-volume and aggregate metrics (mean, std, median, min, max)
- Orientation dropout testing (1, 2, or 3 orientations)
- Multi-GPU support via Accelerate
- Timing and memory profiling
"""

import os
import sys
import argparse
import csv
import json
import time
import logging
from glob import glob
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

# Accelerate for multi-GPU
from accelerate import Accelerator

# MONAI imports
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
)

# Import from project
from src import UHVED
from src.data import HRLRDataGenerator
from src.utils import (
    calculate_metrics,
    calculate_metrics_with_lpips,
    pad_to_multiple_of_32,
    unpad_volume,
    sliding_window_inference,
)


# ============================================================================
# Utility Functions from test.py (reused)
# ============================================================================

def load_uhved_from_checkpoint(checkpoint_path, device="cuda"):
    """
    Load U-HVED model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        device: Device to load model on

    Returns:
        Tuple of (model, checkpoint_data)
    """
    logging.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model configuration
    model_config = checkpoint.get('model_config', {})

    # Required parameters
    num_orientations = model_config.get('num_orientations', 3)
    base_channels = model_config.get('base_channels', 32)
    num_scales = model_config.get('num_scales', 4)

    # Architectural parameters (with backward compatibility defaults)
    reconstruct_orientations = model_config.get('reconstruct_orientations', True)
    use_prior = model_config.get('use_prior', True)
    use_encoder_outputs_as_skip = model_config.get('use_encoder_outputs_as_skip', True)
    decoder_upsample_mode = model_config.get('decoder_upsample_mode', 'trilinear')
    final_activation = model_config.get('final_activation', 'sigmoid')
    share_encoder = model_config.get('share_encoder', False)
    share_decoder = model_config.get('share_decoder', False)
    activation = model_config.get('activation', 'leakyrelu')
    in_channels = model_config.get('in_channels', 1)
    out_channels = model_config.get('out_channels', 1)

    logging.info(f"Model configuration:")
    logging.info(f"  - num_orientations: {num_orientations}")
    logging.info(f"  - base_channels: {base_channels}")
    logging.info(f"  - num_scales: {num_scales}")
    logging.info(f"  - reconstruct_orientations: {reconstruct_orientations}")
    logging.info(f"  - use_prior: {use_prior}")
    logging.info(f"  - use_encoder_outputs_as_skip: {use_encoder_outputs_as_skip}")
    logging.info(f"  - decoder_upsample_mode: {decoder_upsample_mode}")
    logging.info(f"  - final_activation: {final_activation}")

    # Create model with ALL parameters from config
    model = UHVED(
        num_orientations=num_orientations,
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        num_scales=num_scales,
        share_encoder=share_encoder,
        share_decoder=share_decoder,
        use_prior=use_prior,
        use_encoder_outputs_as_skip=use_encoder_outputs_as_skip,
        activation=activation,
        upsample_mode=decoder_upsample_mode,
        reconstruct_orientations=reconstruct_orientations,
        final_activation=final_activation,
    )

    # Remap checkpoint keys for backward compatibility
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('modality_decoders', 'orientation_decoders')
        new_state_dict[new_key] = value

    # Load weights
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    logging.info("Model loaded successfully")
    return model, checkpoint


def create_inference_transforms(target_res=[1.0, 1.0, 1.0]):
    """Create MONAI preprocessing transforms for inference."""
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=target_res, mode="bilinear"),
    ])
    return transforms


def generate_orthogonal_stacks(hr_volume, generator):
    """
    Generate 3 orthogonal LR stacks from HR volume.

    Args:
        hr_volume: High-resolution volume (D, H, W) or (C, D, H, W)
        generator: HRLRDataGenerator instance

    Returns:
        List of 3 LR stacks (each as tensor with shape (C, D, H, W))
    """
    # Convert to tensor if needed
    if isinstance(hr_volume, np.ndarray):
        hr_volume = torch.from_numpy(hr_volume).float()

    # Ensure we have batch and channel dimensions: (B, C, D, H, W)
    if hr_volume.ndim == 3:
        hr_volume = hr_volume.unsqueeze(0).unsqueeze(0)
    elif hr_volume.ndim == 4:
        hr_volume = hr_volume.unsqueeze(0)

    # Generate orthogonal stacks
    lr_stacks_list, _, orientation_mask = generator.generate_paired_data(
        hr_volume, return_resolution=False
    )

    # Remove batch dimension and return as list
    lr_stacks_tensors = [stack.squeeze(0) for stack in lr_stacks_list]

    return lr_stacks_tensors


# ============================================================================
# Data Loading Functions
# ============================================================================

def match_lr_stacks_to_hr(lr_dir, gt_dir, pattern):
    """
    Match LR stacks to HR volumes using pattern.

    Args:
        lr_dir: Directory containing LR stack files
        gt_dir: Directory containing HR ground truth files
        pattern: Pattern string like "{case}_axial.nii.gz,{case}_coronal.nii.gz,{case}_sagittal.nii.gz"

    Returns:
        List of tuples: [(hr_path, [lr_axial, lr_coronal, lr_sagittal]), ...]
    """
    hr_files = sorted(glob(os.path.join(gt_dir, "*.nii*")))
    matched_cases = []

    for hr_path in hr_files:
        case_name = os.path.basename(hr_path).replace(".nii.gz", "").replace(".nii", "")

        # Parse pattern to find LR stack files
        stack_names = pattern.replace("{case}", case_name).split(",")
        stack_paths = [os.path.join(lr_dir, name.strip()) for name in stack_names]

        # Validate all stacks exist
        if all(os.path.exists(p) for p in stack_paths):
            matched_cases.append((hr_path, stack_paths))
        else:
            logging.warning(f"Missing LR stacks for {case_name}, skipping")

    return matched_cases


def load_test_dataset(args, generator=None):
    """
    Load test dataset in either mode: HR-only or paired HR/LR.

    Args:
        args: Command-line arguments
        generator: HRLRDataGenerator instance (required for Mode 1)

    Returns:
        List of tuples: [(lr_stacks, hr_gt, volume_name, affine), ...]
    """
    test_cases = []
    transforms = create_inference_transforms(args.target_res)

    # Mode detection
    if args.input_lr_dir:
        # Mode 2: Paired HR/LR stacks
        logging.info("Mode 2: Loading paired HR and LR stacks")
        matched_pairs = match_lr_stacks_to_hr(
            args.input_lr_dir,
            args.ground_truth,
            args.lr_stack_pattern
        )

        for hr_path, lr_stack_paths in tqdm(matched_pairs, desc="Loading data"):
            volume_name = os.path.basename(hr_path).replace(".nii.gz", "").replace(".nii", "")

            # Load HR ground truth
            hr_data = transforms({"image": hr_path})
            hr_volume = hr_data["image"]
            affine = hr_data["image_meta_dict"]["affine"]

            # Normalize HR
            hr_np = hr_volume.cpu().numpy().squeeze()
            hr_np = (hr_np - hr_np.min()) / (hr_np.max() - hr_np.min() + 1e-8)
            hr_tensor = torch.from_numpy(hr_np).float()

            # Load LR stacks
            lr_stacks = []
            for lr_path in lr_stack_paths:
                lr_data = transforms({"image": lr_path})
                lr_volume = lr_data["image"]
                lr_np = lr_volume.cpu().numpy().squeeze()
                lr_np = (lr_np - lr_np.min()) / (lr_np.max() - lr_np.min() + 1e-8)
                lr_stacks.append(torch.from_numpy(lr_np).float().unsqueeze(0))

            test_cases.append((lr_stacks, hr_tensor, volume_name, affine))

    else:
        # Mode 1: HR volumes only (generate LR stacks)
        logging.info("Mode 1: Loading HR volumes and generating LR stacks")

        if generator is None:
            raise ValueError("Generator required for Mode 1 (HR-only)")

        # Get HR file paths
        if os.path.isfile(args.input):
            hr_files = [args.input]
        else:
            hr_files = sorted(glob(os.path.join(args.input, "*.nii*")))

        for hr_path in tqdm(hr_files, desc="Loading and generating LR stacks"):
            volume_name = os.path.basename(hr_path).replace(".nii.gz", "").replace(".nii", "")

            # Load HR volume
            hr_data = transforms({"image": hr_path})
            hr_volume = hr_data["image"]
            affine = hr_data["image_meta_dict"]["affine"]

            # Normalize
            hr_np = hr_volume.cpu().numpy().squeeze()
            hr_np = (hr_np - hr_np.min()) / (hr_np.max() - hr_np.min() + 1e-8)
            hr_tensor = torch.from_numpy(hr_np).float()

            # Generate LR stacks
            lr_stacks = generate_orthogonal_stacks(hr_np, generator)

            test_cases.append((lr_stacks, hr_tensor, volume_name, affine))

    logging.info(f"Loaded {len(test_cases)} test cases")
    return test_cases


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_single_volume(
    model: nn.Module,
    lr_stacks: List[torch.Tensor],
    hr_ground_truth: torch.Tensor,
    orientation_mask: torch.Tensor,
    device: str,
    use_sliding_window: bool = False,
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    overlap: float = 0.5,
    track_time: bool = True,
    track_memory: bool = False,
    compute_lpips: bool = True,
    lpips_backend: str = 'monai',
) -> Dict[str, Any]:
    """
    Evaluate a single volume.

    Returns:
        Dictionary with metrics, timing, memory, and SR output
    """
    # Timing
    if track_time:
        start_time = time.time()
        if device == 'cuda':
            torch.cuda.synchronize()

    # Memory baseline
    if track_memory and device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    # Padding
    original_shape = lr_stacks[0].squeeze().shape
    lr_stacks_padded = []

    for stack in lr_stacks:
        stack_np = stack.squeeze().cpu().numpy()
        padded, pad_before, orig_shape = pad_to_multiple_of_32(stack_np)
        lr_stacks_padded.append(
            torch.from_numpy(padded).float().unsqueeze(0).unsqueeze(0)
        )

    # Move to device
    lr_stacks_padded = [stack.to(device) for stack in lr_stacks_padded]
    orientation_mask_tensor = orientation_mask.unsqueeze(0).to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        if use_sliding_window:
            sr_output = sliding_window_inference(
                model=model,
                orientations=lr_stacks_padded,
                patch_size=patch_size,
                overlap=overlap,
                batch_size=1,
                device=device,
                blend_mode="gaussian",
                progress=False,
                orientation_mask=orientation_mask_tensor
            )
            orientation_outputs = []
        else:
            outputs = model(lr_stacks_padded, orientation_mask=orientation_mask_tensor)
            sr_output = outputs['sr_output']
            orientation_outputs = outputs.get('orientation_outputs', [])

    # Timing
    if track_time:
        if device == 'cuda':
            torch.cuda.synchronize()
        inference_time = time.time() - start_time

    # Memory
    if track_memory and device == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB

    # Unpad
    sr_output = sr_output.squeeze().cpu().numpy()
    sr_output = unpad_volume(sr_output, pad_before, orig_shape)
    sr_output = np.clip(sr_output, 0, 1)

    # Compute metrics
    sr_tensor = torch.from_numpy(sr_output).unsqueeze(0).unsqueeze(0)
    hr_tensor = hr_ground_truth.unsqueeze(0).unsqueeze(0)
    metrics = calculate_metrics_with_lpips(
        sr_tensor, hr_tensor,
        max_val=1.0,
        compute_lpips=compute_lpips,
        lpips_backend=lpips_backend,
        device=device
    )

    result = {
        'metrics': metrics,
        'sr_output': sr_output,
    }

    if track_time:
        result['timing'] = {'inference_time_sec': inference_time}

    if track_memory and device == 'cuda':
        result['memory'] = {'peak_allocated_mb': peak_memory}

    if len(orientation_outputs) > 0:
        result['orientation_outputs'] = [
            unpad_volume(o.squeeze().cpu().numpy(), pad_before, orig_shape)
            for o in orientation_outputs
        ]

    return result


# ============================================================================
# Orientation Dropout Functions
# ============================================================================

def generate_orientation_configs(mode='all'):
    """
    Generate orientation dropout configurations to test.

    Args:
        mode: 'all', 'single', 'pairs', or custom list

    Returns:
        List of (mask, description) tuples
    """
    if mode == 'all':
        configs = [
            ([1, 1, 1], "All 3 orientations (Axial+Coronal+Sagittal)"),
            ([1, 1, 0], "Axial+Coronal only"),
            ([1, 0, 1], "Axial+Sagittal only"),
            ([0, 1, 1], "Coronal+Sagittal only"),
            ([1, 0, 0], "Axial only"),
            ([0, 1, 0], "Coronal only"),
            ([0, 0, 1], "Sagittal only"),
        ]
    elif mode == 'single':
        configs = [
            ([1, 0, 0], "Axial only"),
            ([0, 1, 0], "Coronal only"),
            ([0, 0, 1], "Sagittal only"),
        ]
    elif mode == 'pairs':
        configs = [
            ([1, 1, 0], "Axial+Coronal"),
            ([1, 0, 1], "Axial+Sagittal"),
            ([0, 1, 1], "Coronal+Sagittal"),
        ]
    else:
        # Parse custom configs (e.g., "111,110,101")
        configs = []
        for config_str in mode.split(','):
            mask = [int(c) for c in config_str.strip()]
            orientations = []
            if mask[0]: orientations.append("Axial")
            if mask[1]: orientations.append("Coronal")
            if mask[2]: orientations.append("Sagittal")
            description = "+".join(orientations) if orientations else "None"
            configs.append((mask, description))

    return configs


# ============================================================================
# Metrics Aggregation
# ============================================================================

def aggregate_metrics(volume_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Compute aggregate statistics across all volumes.

    Returns:
        Dictionary of {metric_name: {mean, std, median, min, max}, ...}
    """
    if not volume_results:
        return {}

    # Base metrics always present
    metric_names = ['mae', 'mse', 'rmse', 'psnr', 'r2', 'ssim']

    # Add LPIPS metrics if they exist in results
    first_result = volume_results[0]['metrics']
    if 'lpips_3d' in first_result:
        metric_names.extend(['lpips_3d', 'rlpips'])

    aggregates = {}

    for metric_name in metric_names:
        if metric_name in first_result:
            values = [r['metrics'][metric_name] for r in volume_results]
            aggregates[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }

    # Timing statistics
    if 'timing' in volume_results[0]:
        times = [r['timing']['inference_time_sec'] for r in volume_results]
        aggregates['inference_time_sec'] = {
            'mean': float(np.mean(times)),
            'std': float(np.std(times)),
            'median': float(np.median(times)),
            'min': float(np.min(times)),
            'max': float(np.max(times))
        }

    # Memory statistics
    if 'memory' in volume_results[0]:
        mems = [r['memory']['peak_allocated_mb'] for r in volume_results]
        aggregates['peak_memory_mb'] = {
            'mean': float(np.mean(mems)),
            'std': float(np.std(mems)),
            'median': float(np.median(mems)),
            'min': float(np.min(mems)),
            'max': float(np.max(mems))
        }

    return aggregates


# ============================================================================
# Output Generation Functions
# ============================================================================

def save_csv_results(
    volume_results: List[Dict[str, Any]],
    aggregates: Dict[str, Dict[str, float]],
    output_path: str,
    volume_names: List[str]
):
    """Save results to CSV with per-volume and aggregate sections."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Per-volume section
        writer.writerow(['# Per-Volume Metrics'])
        headers = ['volume_name', 'mae', 'mse', 'rmse', 'psnr', 'r2', 'ssim']

        # Add LPIPS metrics if present
        first_result = volume_results[0] if volume_results else {}
        if first_result and 'lpips_3d' in first_result.get('metrics', {}):
            headers.extend(['lpips_3d', 'rlpips'])

        if volume_results and 'timing' in volume_results[0]:
            headers.append('inference_time_sec')
        if volume_results and 'memory' in volume_results[0]:
            headers.append('peak_memory_mb')

        writer.writerow(headers)

        for name, result in zip(volume_names, volume_results):
            base_metrics = ['mae', 'mse', 'rmse', 'psnr', 'r2', 'ssim']
            row = [name] + [
                f"{result['metrics'][m]:.6f}"
                for m in base_metrics
            ]

            # Add LPIPS metrics if present
            if 'lpips_3d' in result['metrics']:
                lpips_metrics = ['lpips_3d', 'rlpips']
                row.extend([
                    f"{result['metrics'][m]:.6f}"
                    for m in lpips_metrics
                ])

            if 'timing' in result:
                row.append(f"{result['timing']['inference_time_sec']:.4f}")
            if 'memory' in result:
                row.append(f"{result['memory']['peak_allocated_mb']:.2f}")
            writer.writerow(row)

        # Aggregate section
        writer.writerow([])
        writer.writerow(['# Aggregate Statistics'])
        writer.writerow(['metric', 'mean', 'std', 'median', 'min', 'max'])

        # Base metrics
        base_metrics = ['mae', 'mse', 'rmse', 'psnr', 'r2', 'ssim']
        for metric_name in base_metrics:
            if metric_name in aggregates:
                stats = aggregates[metric_name]
                writer.writerow([
                    metric_name,
                    f"{stats['mean']:.6f}",
                    f"{stats['std']:.6f}",
                    f"{stats['median']:.6f}",
                    f"{stats['min']:.6f}",
                    f"{stats['max']:.6f}"
                ])

        # LPIPS metrics
        lpips_metrics = ['lpips_3d', 'rlpips']
        for metric_name in lpips_metrics:
            if metric_name in aggregates:
                stats = aggregates[metric_name]
                writer.writerow([
                    metric_name,
                    f"{stats['mean']:.6f}",
                    f"{stats['std']:.6f}",
                    f"{stats['median']:.6f}",
                    f"{stats['min']:.6f}",
                    f"{stats['max']:.6f}"
                ])

        if 'inference_time_sec' in aggregates:
            stats = aggregates['inference_time_sec']
            writer.writerow([
                'inference_time_sec',
                f"{stats['mean']:.4f}",
                f"{stats['std']:.4f}",
                f"{stats['median']:.4f}",
                f"{stats['min']:.4f}",
                f"{stats['max']:.4f}"
            ])

        if 'peak_memory_mb' in aggregates:
            stats = aggregates['peak_memory_mb']
            writer.writerow([
                'peak_memory_mb',
                f"{stats['mean']:.2f}",
                f"{stats['std']:.2f}",
                f"{stats['median']:.2f}",
                f"{stats['min']:.2f}",
                f"{stats['max']:.2f}"
            ])

    logging.info(f"CSV results saved to: {output_path}")


def save_json_results(
    volume_results: List[Dict[str, Any]],
    aggregates: Dict[str, Dict[str, float]],
    output_path: str,
    volume_names: List[str],
    metadata: Dict[str, Any]
):
    """Save structured JSON results."""
    output_data = {
        'metadata': {
            **metadata,
            'evaluation_date': datetime.now().isoformat(),
            'num_volumes': len(volume_results)
        },
        'per_volume_results': [
            {
                'volume_name': name,
                'metrics': result['metrics'],
                **(({'timing': result['timing']} if 'timing' in result else {})),
                **(({'memory': result['memory']} if 'memory' in result else {}))
            }
            for name, result in zip(volume_names, volume_results)
        ],
        'aggregate_statistics': aggregates
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logging.info(f"JSON results saved to: {output_path}")


def print_evaluation_summary(
    aggregates: Dict[str, Dict[str, float]],
    num_volumes: int,
    orientation_config: Optional[List[int]] = None,
    model_path: str = "",
    device: str = "cuda"
):
    """Print formatted summary table to console."""
    console = Console()

    # Header
    console.print("=" * 80, style="bold")
    console.print("EVALUATION SUMMARY", style="bold cyan", justify="center")
    console.print("=" * 80, style="bold")
    console.print(f"Model: {model_path}")
    console.print(f"Device: {device}")
    console.print(f"Volumes Evaluated: {num_volumes}")

    if orientation_config:
        config_str = str(orientation_config)
        orientations = []
        if orientation_config[0]: orientations.append("Axial")
        if orientation_config[1]: orientations.append("Coronal")
        if orientation_config[2]: orientations.append("Sagittal")
        console.print(f"Orientation Config: {config_str} ({'+'.join(orientations)})")

    console.print()

    # Metrics table
    console.print("=" * 80, style="bold")
    console.print("METRICS (mean ± std)", style="bold cyan", justify="center")
    console.print("=" * 80, style="bold")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=15)
    table.add_column("Mean ± Std", justify="right", width=25)
    table.add_column("Range", justify="right", width=30)

    metric_configs = [
        ('MAE', 'mae', '.4f', ''),
        ('MSE', 'mse', '.6f', ''),
        ('RMSE', 'rmse', '.4f', ''),
        ('PSNR', 'psnr', '.2f', ' dB'),
        ('R²', 'r2', '.4f', ''),
        ('SSIM', 'ssim', '.4f', ''),
    ]

    # Add LPIPS metrics if available
    if 'lpips_3d' in aggregates:
        metric_configs.extend([
            ('3D-LPIPS', 'lpips_3d', '.4f', ''),
            ('R-LPIPS', 'rlpips', '.4f', ''),
        ])

    for config in metric_configs:
        name = config[0]
        key = config[1]
        fmt = config[2]
        suffix = config[3]

        if key in aggregates:
            stats = aggregates[key]
            mean_std = f"{stats['mean']:{fmt}} ± {stats['std']:{fmt}}{suffix}"
            range_str = f"[{stats['min']:{fmt}} - {stats['max']:{fmt}}]{suffix}"
            table.add_row(name, mean_std, range_str)

    console.print(table)

    # Performance table (if available)
    if 'inference_time_sec' in aggregates or 'peak_memory_mb' in aggregates:
        console.print("\n" + "=" * 80, style="bold")
        console.print("PERFORMANCE", style="bold cyan", justify="center")
        console.print("=" * 80, style="bold")

        perf_table = Table(show_header=True, header_style="bold magenta")
        perf_table.add_column("Metric", style="cyan", width=20)
        perf_table.add_column("Mean ± Std", justify="right", width=25)
        perf_table.add_column("Range", justify="right", width=30)

        # Timing
        if 'inference_time_sec' in aggregates:
            stats = aggregates['inference_time_sec']
            perf_table.add_row(
                "Inference Time",
                f"{stats['mean']:.2f} ± {stats['std']:.2f} sec",
                f"[{stats['min']:.2f} - {stats['max']:.2f}] sec"
            )

        # Memory
        if 'peak_memory_mb' in aggregates:
            stats = aggregates['peak_memory_mb']
            perf_table.add_row(
                "Peak Memory",
                f"{stats['mean']:.1f} ± {stats['std']:.1f} MB",
                f"[{stats['min']:.1f} - {stats['max']:.1f}] MB"
            )

        console.print(perf_table)

    console.print("=" * 80, style="bold")


def save_sr_outputs(
    volume_results: List[Dict[str, Any]],
    volume_names: List[str],
    output_dir: str,
    affines: List[np.ndarray],
    save_reconstructions: bool = False
):
    """Save SR outputs and optional orientation reconstructions as NIfTI files."""
    sr_dir = os.path.join(output_dir, 'sr_outputs')
    os.makedirs(sr_dir, exist_ok=True)

    if save_reconstructions:
        recon_dir = os.path.join(output_dir, 'reconstructions')
        os.makedirs(recon_dir, exist_ok=True)

    for name, result, affine in zip(volume_names, volume_results, affines):
        # Save SR output
        sr_path = os.path.join(sr_dir, f"{name}_sr.nii.gz")
        sr_nii = nib.Nifti1Image(result['sr_output'], affine)
        nib.save(sr_nii, sr_path)

        # Save orientation reconstructions
        if save_reconstructions and 'orientation_outputs' in result:
            orientation_names = ['axial', 'coronal', 'sagittal']
            for i, recon in enumerate(result['orientation_outputs']):
                recon_path = os.path.join(recon_dir, f"{name}_recon_{orientation_names[i]}.nii.gz")
                recon_nii = nib.Nifti1Image(recon, affine)
                nib.save(recon_nii, recon_path)

    logging.info(f"SR outputs saved to: {sr_dir}")
    if save_reconstructions:
        logging.info(f"Reconstructions saved to: {recon_dir}")


# ============================================================================
# Main Evaluation Loop
# ============================================================================

def evaluate_model(
    model: nn.Module,
    test_dataset: List[Tuple],
    orientation_configs: List[Tuple[List[int], str]],
    args: argparse.Namespace,
    accelerator: Accelerator
):
    """
    Main evaluation loop.

    Returns:
        Dictionary with all results
    """
    all_results = {}

    for mask, description in orientation_configs:
        logging.info(f"\nEvaluating configuration: {description}")
        logging.info(f"  Mask: {mask}")

        orientation_mask = torch.tensor(mask, dtype=torch.bool)
        config_results = []
        volume_names = []
        affines = []

        for lr_stacks, hr_gt, volume_name, affine in tqdm(
            test_dataset,
            desc=f"Evaluating {description}"
        ):
            try:
                # Evaluate single volume
                result = evaluate_single_volume(
                    model=model,
                    lr_stacks=lr_stacks,
                    hr_ground_truth=hr_gt,
                    orientation_mask=orientation_mask,
                    device=args.device,
                    use_sliding_window=args.use_sliding_window,
                    patch_size=tuple(args.patch_size) if args.patch_size else (128, 128, 128),
                    overlap=args.overlap,
                    track_time=True,
                    track_memory=args.track_memory,
                    compute_lpips=args.compute_lpips,
                    lpips_backend=args.lpips_backend
                )

                config_results.append(result)
                volume_names.append(volume_name)
                affines.append(affine)

            except Exception as e:
                logging.error(f"Failed to evaluate {volume_name}: {str(e)}")
                logging.exception(e)
                continue

        # Aggregate metrics
        aggregates = aggregate_metrics(config_results)

        # Store results
        all_results[description] = {
            'volume_results': config_results,
            'volume_names': volume_names,
            'affines': affines,
            'aggregates': aggregates,
            'orientation_mask': mask
        }

        # Print summary for this config
        if args.verbose:
            print_evaluation_summary(
                aggregates=aggregates,
                num_volumes=len(config_results),
                orientation_config=mask,
                model_path=args.model,
                device=args.device
            )

    return all_results


# ============================================================================
# Argument Parser
# ============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate U-HVED super-resolution model on test datasets"
    )

    # Input/Output arguments
    parser.add_argument('--input', type=str,
                        help='HR volume file or directory (Mode 1)')
    parser.add_argument('--input_lr_dir', type=str,
                        help='Directory containing LR stacks (Mode 2)')
    parser.add_argument('--ground_truth', type=str, required=True,
                        help='HR ground truth file or directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save all results')
    parser.add_argument('--save_sr_outputs', action='store_true',
                        help='Save SR volumes as .nii.gz')
    parser.add_argument('--save_reconstructions', action='store_true',
                        help='Save orientation reconstructions')

    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for inference')

    # Data processing arguments
    parser.add_argument('--target_res', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help='Target resolution [x, y, z] in mm')
    parser.add_argument('--output_shape', type=int, nargs=3,
                        help='Output shape for generated stacks')
    parser.add_argument('--lr_stack_pattern', type=str,
                        default='{case}_axial.nii.gz,{case}_coronal.nii.gz,{case}_sagittal.nii.gz',
                        help='Pattern for LR stack files (Mode 2)')

    # Inference arguments
    parser.add_argument('--use_sliding_window', action='store_true',
                        help='Enable sliding window inference')
    parser.add_argument('--patch_size', type=int, nargs=3,
                        help='Patch size for sliding window')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Overlap ratio (0.0-1.0)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation')

    # Orientation dropout arguments
    parser.add_argument('--test_all_orientations', action='store_true',
                        help='Test all 7 orientation combinations')
    parser.add_argument('--orientation_configs', type=str,
                        help='Specific configs to test (e.g., "111,110,101")')
    parser.add_argument('--orientation_mask', type=int, nargs=3,
                        help='Single mask to test (e.g., 1 1 0)')

    # Output format arguments
    parser.add_argument('--csv_output', type=str,
                        help='Path to save CSV results')
    parser.add_argument('--json_output', type=str,
                        help='Path to save JSON results')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed per-volume results')

    # Performance arguments
    parser.add_argument('--track_memory', action='store_true',
                        help='Track GPU memory usage')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')

    # LPIPS metric arguments
    parser.add_argument('--compute_lpips', action='store_true', default=True,
                        help='Compute 3D-LPIPS and R-LPIPS metrics (default: True)')
    parser.add_argument('--no_lpips', dest='compute_lpips', action='store_false',
                        help='Disable LPIPS metrics computation')
    parser.add_argument('--lpips_backend', type=str, default='monai',
                        choices=['monai', 'medicalnet', 'models_genesis'],
                        help='3D network backend for LPIPS computation (default: monai)')

    # Data generator arguments (for Mode 1)
    parser.add_argument('--thickness_range', type=float, nargs=2, default=[2.0, 5.0],
                        help='LR slice thickness range [min, max] in mm')
    parser.add_argument('--slice_profile', type=str, default='trapezoid',
                        choices=['trapezoid', 'gaussian', 'boxcar'],
                        help='Slice profile type')
    parser.add_argument("--upsample_mode", type=str, default="nearest",
                        choices=["nearest", "trilinear", "nearest-exact"],
                        help="Interpolation mode for FFT upsample recovery (default: nearest)")

    return parser.parse_args()


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main evaluation function."""
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'evaluation.log')),
            logging.StreamHandler()
        ]
    )

    logging.info("=" * 80)
    logging.info("U-HVED Evaluation Script")
    logging.info("=" * 80)
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Device: {args.device}")

    # Initialize Accelerator for multi-GPU support
    accelerator = Accelerator()

    # Load model
    model, checkpoint = load_uhved_from_checkpoint(args.model, args.device)
    model = accelerator.prepare(model)

    # Setup data generator (for Mode 1)
    generator = None
    if not args.input_lr_dir:
        logging.info("Initializing HRLRDataGenerator for Mode 1")
        from src.data import HRLRDataGenerator
        generator = HRLRDataGenerator(
            output_shape=tuple(args.output_shape) if args.output_shape else (160, 224, 160),
            thickness_range=tuple(args.thickness_range),
            slice_profile=args.slice_profile,
            simulate_bias_field=True,
            simulate_motion=True,
            simulate_k_space_spikes=True,
            upsample_mode=args.upsample_mode,
        )

    # Load test dataset
    test_dataset = load_test_dataset(args, generator)

    if not test_dataset:
        logging.error("No test cases found!")
        return

    # Determine orientation configurations to test
    if args.test_all_orientations:
        orientation_configs = generate_orientation_configs('all')
    elif args.orientation_configs:
        orientation_configs = generate_orientation_configs(args.orientation_configs)
    elif args.orientation_mask:
        mask = args.orientation_mask
        orientations = []
        if mask[0]: orientations.append("Axial")
        if mask[1]: orientations.append("Coronal")
        if mask[2]: orientations.append("Sagittal")
        description = "+".join(orientations) if orientations else "None"
        orientation_configs = [(mask, description)]
    else:
        # Default: all 3 orientations
        orientation_configs = [([1, 1, 1], "All 3 orientations (Axial+Coronal+Sagittal)")]

    logging.info(f"Testing {len(orientation_configs)} orientation configuration(s)")

    # Run evaluation
    all_results = evaluate_model(
        model=model,
        test_dataset=test_dataset,
        orientation_configs=orientation_configs,
        args=args,
        accelerator=accelerator
    )

    # Save results for each configuration
    for config_name, config_data in all_results.items():
        config_suffix = config_name.replace(" ", "_").replace("+", "_")

        # Save CSV
        if args.csv_output or len(orientation_configs) > 1:
            csv_path = args.csv_output if args.csv_output else os.path.join(
                args.output_dir, f'results_{config_suffix}.csv'
            )
            save_csv_results(
                volume_results=config_data['volume_results'],
                aggregates=config_data['aggregates'],
                output_path=csv_path,
                volume_names=config_data['volume_names']
            )

        # Save JSON
        if args.json_output or len(orientation_configs) > 1:
            json_path = args.json_output if args.json_output else os.path.join(
                args.output_dir, f'results_{config_suffix}.json'
            )
            metadata = {
                'model_path': args.model,
                'device': args.device,
                'orientation_config': config_data['orientation_mask'],
                'orientation_description': config_name,
                'sliding_window': args.use_sliding_window,
                'patch_size': args.patch_size if args.patch_size else None,
            }
            save_json_results(
                volume_results=config_data['volume_results'],
                aggregates=config_data['aggregates'],
                output_path=json_path,
                volume_names=config_data['volume_names'],
                metadata=metadata
            )

        # Print summary
        print_evaluation_summary(
            aggregates=config_data['aggregates'],
            num_volumes=len(config_data['volume_results']),
            orientation_config=config_data['orientation_mask'],
            model_path=args.model,
            device=args.device
        )

        # Save SR outputs (only for main config or if explicitly requested)
        if args.save_sr_outputs and (len(orientation_configs) == 1 or config_name == orientation_configs[0][1]):
            save_sr_outputs(
                volume_results=config_data['volume_results'],
                volume_names=config_data['volume_names'],
                output_dir=args.output_dir,
                affines=config_data['affines'],
                save_reconstructions=args.save_reconstructions
            )

    logging.info("\nEvaluation complete!")
    logging.info(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
