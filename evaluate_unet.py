#!/usr/bin/env python3
"""
Evaluation script for U-Net Super-Resolution Model

This script performs comprehensive evaluation on validation/test sets with multiple metrics.
Computes standard metrics (MAE, MSE, PSNR, SSIM) and perceptual metrics (3D-LPIPS, R-LPIPS).
Generates CSV, JSON, and console outputs with aggregate statistics.

Usage examples:
    # Evaluate with stack generation
    python evaluate_unet.py --ground_truth /path/to/test_volumes/ \
                            --output_dir ./evaluation_results \
                            --model checkpoints_unet/unet_best.pth \
                            --compute_lpips

    # Evaluate with pre-existing LR stacks
    python evaluate_unet.py --ground_truth /path/to/hr_volumes/ \
                            --input_lr_dir /path/to/lr_stacks/ \
                            --output_dir ./evaluation_results \
                            --model checkpoints_unet/unet_best.pth

    # Full evaluation with all metrics and timing
    python evaluate_unet.py --ground_truth /path/to/test_volumes/ \
                            --output_dir ./evaluation_results \
                            --model checkpoints_unet/unet_best.pth \
                            --compute_lpips \
                            --lpips_backend monai \
                            --track_memory \
                            --save_sr_outputs \
                            --verbose
"""

import os
import csv
import json
import time
import argparse
import numpy as np
import torch
import nibabel as nib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from datetime import datetime

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd
from monai.inferers import sliding_window_inference

from src.unet import CustomUNet3D
from src.data import HRLRDataGenerator
from src.utils import (
    pad_to_multiple_of_32,
    unpad_volume,
    calculate_metrics,
    calculate_metrics_with_lpips,
)

# Import helper functions from test_unet.py
from test_unet import (
    load_unet_from_checkpoint,
    concatenate_stacks,
    create_inference_transforms,
    load_and_generate_stacks,
    load_existing_stacks,
)


def get_gpu_memory_stats(device: str = "cuda") -> Dict:
    """
    Get GPU memory statistics if CUDA is available.

    Args:
        device: Device string

    Returns:
        Dictionary with memory statistics in MB
    """
    if device == "cpu" or not torch.cuda.is_available():
        return {}

    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        'allocated_mb': allocated,
        'reserved_mb': reserved,
        'peak_memory_mb': max_allocated,
    }


def aggregate_metrics(all_metrics: List[Dict]) -> Dict:
    """
    Aggregate metrics across all volumes.

    Args:
        all_metrics: List of metric dictionaries from each volume

    Returns:
        Dictionary with {metric_name: {mean, std, median, min, max}}
    """
    if not all_metrics:
        return {}

    metrics_dict = {}
    metric_names = all_metrics[0].keys()

    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics if metric_name in m]
        if values:
            metrics_dict[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }

    return metrics_dict


def evaluate_single_volume(
    model: torch.nn.Module,
    hr_path: str,
    lr_stack_paths: Optional[List[str]],
    generator: Optional[HRLRDataGenerator],
    transforms,
    device: str,
    use_sliding_window: bool,
    patch_size: Tuple[int, int, int],
    overlap: float,
    compute_lpips: bool,
    lpips_backend: str,
    track_memory: bool,
) -> Tuple[Dict, np.ndarray]:
    """
    Evaluate model on a single volume.

    Args:
        model: Trained U-Net model
        hr_path: Path to ground truth HR volume
        lr_stack_paths: Optional pre-existing LR stack paths
        generator: Optional data generator for creating LR stacks
        transforms: MONAI transform chain
        device: Inference device
        use_sliding_window: Use sliding window inference
        patch_size: Patch size for sliding window
        overlap: Overlap ratio for sliding window
        compute_lpips: Whether to compute LPIPS metrics
        lpips_backend: Backend for LPIPS computation
        track_memory: Whether to track GPU memory usage

    Returns:
        metrics: Dictionary with all computed metrics
        sr_output: Super-resolved output as numpy array
    """
    # Reset peak memory stats
    if track_memory and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Load ground truth
    hr_dict = {"image": hr_path}
    hr_transformed = transforms(hr_dict)
    hr_volume = hr_transformed["image"]
    affine = hr_transformed["image_meta_dict"]["affine"]

    # Ensure correct dimensions
    if hr_volume.ndim == 3:
        hr_volume = hr_volume.unsqueeze(0)  # Add channel

    # Normalize
    hr_min = hr_volume.min()
    hr_max = hr_volume.max()
    hr_volume = (hr_volume - hr_min) / (hr_max - hr_min + 1e-8)
    hr_volume = hr_volume.unsqueeze(0).to(device)  # (1, 1, D, H, W)

    # Load or generate LR stacks
    if lr_stack_paths is not None:
        lr_stacks_padded, _, original_shape, pad_before = load_existing_stacks(
            lr_stack_paths, transforms, device
        )
    else:
        # Need to pass the HR volume to the generator
        # First create a temporary HR volume without normalization for generation
        hr_dict_gen = {"image": hr_path}
        hr_transformed_gen = transforms(hr_dict_gen)
        hr_volume_gen = hr_transformed_gen["image"]

        if hr_volume_gen.ndim == 3:
            hr_volume_gen = hr_volume_gen.unsqueeze(0)
        hr_volume_gen = hr_volume_gen.unsqueeze(0)

        # Normalize for generation
        hr_min_gen = hr_volume_gen.min()
        hr_max_gen = hr_volume_gen.max()
        hr_volume_gen = (hr_volume_gen - hr_min_gen) / (hr_max_gen - hr_min_gen + 1e-8)

        # Generate LR stacks
        lr_stacks_list, _, _, _, _ = generator.generate_paired_data(
            hr_volume_gen, return_resolution=True
        )

        # Pad LR stacks
        lr_stacks_padded = []
        original_shape = None
        pad_before = None

        for stack in lr_stacks_list:
            stack_np = stack.squeeze(0).squeeze(0).cpu().numpy()

            if original_shape is None:
                original_shape = np.array(stack_np.shape)

            padded, pb, _ = pad_to_multiple_of_32(stack_np)
            if pad_before is None:
                pad_before = pb

            lr_stacks_padded.append(
                torch.from_numpy(padded).float().unsqueeze(0).unsqueeze(0).to(device)
            )

    # Pad HR to match LR
    hr_np = hr_volume.squeeze().cpu().numpy()
    hr_padded, hr_pad_before, _ = pad_to_multiple_of_32(hr_np)
    hr_padded = torch.from_numpy(hr_padded).float().unsqueeze(0).unsqueeze(0).to(device)

    # Concatenate stacks
    lr_concat = concatenate_stacks(lr_stacks_padded)

    # Inference with timing
    start_time = time.time()

    with torch.no_grad():
        if use_sliding_window:
            sr_output = sliding_window_inference(
                inputs=lr_concat,
                roi_size=patch_size,
                sw_batch_size=1,
                predictor=model,
                overlap=overlap,
                mode="gaussian",
            )
        else:
            sr_output = model(lr_concat)

    inference_time = time.time() - start_time

    # Memory tracking
    memory_stats = {}
    if track_memory:
        memory_stats = get_gpu_memory_stats(device)

    # Unpad both
    sr_np = sr_output.squeeze().cpu().numpy()
    sr_np = unpad_volume(sr_np, pad_before, original_shape)

    hr_np = hr_padded.squeeze().cpu().numpy()
    hr_np = unpad_volume(hr_np, hr_pad_before, original_shape)

    # Convert to tensors for metrics
    sr_tensor = torch.from_numpy(sr_np).unsqueeze(0).unsqueeze(0).to(device)
    hr_tensor = torch.from_numpy(hr_np).unsqueeze(0).unsqueeze(0).to(device)

    # Compute metrics
    if compute_lpips:
        metrics = calculate_metrics_with_lpips(
            sr_tensor, hr_tensor,
            max_val=1.0,
            compute_lpips=True,
            lpips_backend=lpips_backend,
            device=device,
        )
    else:
        metrics = calculate_metrics(sr_tensor, hr_tensor, max_val=1.0)

    # Add timing and memory
    metrics['inference_time_sec'] = inference_time
    if memory_stats:
        for key, value in memory_stats.items():
            metrics[key] = value

    return metrics, sr_np, affine


def save_csv_results(
    output_path: str,
    per_volume_results: List[Dict],
    aggregate_stats: Dict,
):
    """
    Save results to CSV format.

    Args:
        output_path: Path to save CSV file
        per_volume_results: List of per-volume result dictionaries
        aggregate_stats: Aggregate statistics dictionary
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Per-volume section
        metric_names = list(per_volume_results[0]['metrics'].keys())
        writer.writerow(['volume_name'] + metric_names)

        # Data rows
        for result in per_volume_results:
            row = [result['volume_name']]
            row.extend([f"{result['metrics'][m]:.6f}" if isinstance(result['metrics'][m], (int, float)) else result['metrics'][m]
                       for m in metric_names])
            writer.writerow(row)

        # Separator
        writer.writerow([])
        writer.writerow(['=== AGGREGATE STATISTICS ==='])
        writer.writerow(['metric', 'mean', 'std', 'median', 'min', 'max'])

        # Aggregate rows
        for metric_name, stats in aggregate_stats.items():
            writer.writerow([
                metric_name,
                f"{stats['mean']:.6f}",
                f"{stats['std']:.6f}",
                f"{stats['median']:.6f}",
                f"{stats['min']:.6f}",
                f"{stats['max']:.6f}",
            ])

    print(f"Saved CSV results: {output_path}")


def save_json_results(
    output_path: str,
    per_volume_results: List[Dict],
    aggregate_stats: Dict,
    metadata: Dict,
):
    """
    Save results to JSON format.

    Args:
        output_path: Path to save JSON file
        per_volume_results: List of per-volume result dictionaries
        aggregate_stats: Aggregate statistics dictionary
        metadata: Evaluation metadata dictionary
    """
    results = {
        'metadata': metadata,
        'per_volume_results': per_volume_results,
        'aggregate_statistics': aggregate_stats,
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved JSON results: {output_path}")


def print_evaluation_summary(aggregate_stats: Dict):
    """
    Print formatted summary to console.

    Args:
        aggregate_stats: Aggregate statistics dictionary
    """
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    # Standard metrics
    standard_metrics = ['mae', 'mse', 'rmse', 'psnr', 'r2', 'ssim']
    if any(m in aggregate_stats for m in standard_metrics):
        print("\nStandard Metrics:")
        print(f"{'Metric':<15} {'Mean ± Std':<25} {'[Min - Max]':<25}")
        print("-" * 65)

        for metric_name in standard_metrics:
            if metric_name in aggregate_stats:
                stats = aggregate_stats[metric_name]
                print(
                    f"{metric_name.upper():<15} "
                    f"{stats['mean']:>8.4f} ± {stats['std']:<8.4f}   "
                    f"[{stats['min']:>8.4f} - {stats['max']:<8.4f}]"
                )

    # Perceptual metrics
    perceptual_metrics = ['lpips_3d', 'rlpips']
    if any(m in aggregate_stats for m in perceptual_metrics):
        print("\nPerceptual Metrics:")
        print(f"{'Metric':<15} {'Mean ± Std':<25} {'[Min - Max]':<25}")
        print("-" * 65)

        for metric_name in perceptual_metrics:
            if metric_name in aggregate_stats:
                stats = aggregate_stats[metric_name]
                print(
                    f"{metric_name.upper():<15} "
                    f"{stats['mean']:>8.4f} ± {stats['std']:<8.4f}   "
                    f"[{stats['min']:>8.4f} - {stats['max']:<8.4f}]"
                )

    # Performance metrics
    performance_metrics = ['inference_time_sec', 'peak_memory_mb']
    if any(m in aggregate_stats for m in performance_metrics):
        print("\nPerformance Metrics:")
        print(f"{'Metric':<25} {'Mean ± Std':<20}")
        print("-" * 45)

        if 'inference_time_sec' in aggregate_stats:
            stats = aggregate_stats['inference_time_sec']
            print(f"{'Inference Time (s)':<25} {stats['mean']:>6.2f} ± {stats['std']:<6.2f}")

        if 'peak_memory_mb' in aggregate_stats:
            stats = aggregate_stats['peak_memory_mb']
            print(f"{'Peak Memory (MB)':<25} {stats['mean']:>6.1f} ± {stats['std']:<6.1f}")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate U-Net Super-Resolution Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input/Output
    parser.add_argument("--ground_truth", type=str, required=True,
                       help="Ground truth HR volume file or directory")
    parser.add_argument("--input_lr_dir", type=str,
                       help="Directory with pre-existing LR stacks (optional)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--save_sr_outputs", action="store_true",
                       help="Save SR output volumes as NIfTI")

    # Model
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained U-Net checkpoint (.pth)")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device for inference")

    # Data processing (if generating stacks)
    parser.add_argument("--target_res", type=float, nargs=3,
                       default=[1.0, 1.0, 1.0],
                       help="Target resolution in mm (x y z)")
    parser.add_argument("--output_shape", type=int, nargs=3,
                       default=[128, 128, 128],
                       help="Output volume shape (x y z)")
    parser.add_argument("--max_res_aniso", type=float, nargs=3,
                       default=[9.0, 9.0, 9.0],
                       help="Max anisotropic resolution for LR generation (x y z)")
    parser.add_argument("--upsample_mode", type=str, default="nearest",
                       choices=["nearest", "trilinear", "nearest-exact"],
                       help="Upsampling mode for LR generation")

    # Inference
    parser.add_argument("--use_sliding_window", action="store_true",
                       help="Use sliding window inference for large volumes")
    parser.add_argument("--patch_size", type=int, nargs=3,
                       default=[128, 128, 128],
                       help="Patch size for sliding window (x y z)")
    parser.add_argument("--overlap", type=float, default=0.5,
                       help="Overlap ratio for sliding window (0.0-1.0)")

    # Metrics
    parser.add_argument("--compute_lpips", action="store_true", default=True,
                       help="Compute LPIPS perceptual metrics (default: True)")
    parser.add_argument("--no_lpips", dest='compute_lpips', action="store_false",
                       help="Disable LPIPS computation")
    parser.add_argument("--lpips_backend", type=str, default="monai",
                       choices=["monai", "medicalnet", "models_genesis"],
                       help="Backend for 3D-LPIPS computation")

    # Performance
    parser.add_argument("--track_memory", action="store_true",
                       help="Track GPU memory usage during inference")

    # Output
    parser.add_argument("--csv_output", type=str, default=None,
                       help="Custom path for CSV output (default: output_dir/results.csv)")
    parser.add_argument("--json_output", type=str, default=None,
                       help="Custom path for JSON output (default: output_dir/results.json)")
    parser.add_argument("--verbose", action="store_true",
                       help="Print per-volume metrics during evaluation")

    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"

    # Load model
    print("\nLoading model...")
    model, checkpoint = load_unet_from_checkpoint(args.model, device)

    # Create generator if needed
    generator = None
    if args.input_lr_dir is None:
        print("\nInitializing data generator...")
        generator = HRLRDataGenerator(
            atlas_res=args.target_res,
            target_res=[1.0, 1.0, 1.0],
            output_shape=args.output_shape,
            max_res_aniso=args.max_res_aniso,
            randomise_res=False,
            apply_intensity_aug=False,
            orientation_dropout_prob=0.0,
            upsample_mode=args.upsample_mode,
        )

    # Get file list
    gt_path = Path(args.ground_truth)
    if gt_path.is_file():
        hr_files = [gt_path]
    else:
        hr_files = sorted(gt_path.glob("*.nii.gz"))

    print(f"\nFound {len(hr_files)} volumes to evaluate")

    # Evaluation loop
    per_volume_results = []
    transforms = create_inference_transforms(args.target_res)

    for hr_file in tqdm(hr_files, desc="Evaluating"):
        volume_name = hr_file.stem.replace('.nii', '')

        # Get LR stacks if provided
        lr_stack_paths = None
        if args.input_lr_dir:
            # Match pattern: {volume}_axial.nii.gz, {volume}_coronal.nii.gz, {volume}_sagittal.nii.gz
            lr_dir = Path(args.input_lr_dir)
            lr_stack_paths = [
                str(lr_dir / f"{volume_name}_axial.nii.gz"),
                str(lr_dir / f"{volume_name}_coronal.nii.gz"),
                str(lr_dir / f"{volume_name}_sagittal.nii.gz"),
            ]

            # Verify files exist
            if not all(Path(p).exists() for p in lr_stack_paths):
                print(f"\nWarning: Missing LR stacks for {volume_name}, skipping...")
                continue

        # Evaluate
        try:
            metrics, sr_output, affine = evaluate_single_volume(
                model=model,
                hr_path=str(hr_file),
                lr_stack_paths=lr_stack_paths,
                generator=generator,
                transforms=transforms,
                device=device,
                use_sliding_window=args.use_sliding_window,
                patch_size=tuple(args.patch_size),
                overlap=args.overlap,
                compute_lpips=args.compute_lpips,
                lpips_backend=args.lpips_backend,
                track_memory=args.track_memory,
            )

            per_volume_results.append({
                'volume_name': volume_name,
                'metrics': metrics,
            })

            # Save SR output if requested
            if args.save_sr_outputs:
                sr_dir = os.path.join(args.output_dir, 'sr_outputs')
                os.makedirs(sr_dir, exist_ok=True)
                sr_path = os.path.join(sr_dir, f"{volume_name}_sr.nii.gz")

                sr_nii = nib.Nifti1Image(sr_output, affine)
                nib.save(sr_nii, sr_path)

            if args.verbose:
                psnr = metrics.get('psnr', 0)
                ssim = metrics.get('ssim', 0)
                print(f"\n{volume_name}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")

        except Exception as e:
            print(f"\nError processing {volume_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not per_volume_results:
        print("\nNo volumes were successfully evaluated!")
        return

    # Aggregate statistics
    all_metrics = [r['metrics'] for r in per_volume_results]
    aggregate_stats = aggregate_metrics(all_metrics)

    # Metadata
    metadata = {
        'model_path': args.model,
        'device': device,
        'sliding_window': args.use_sliding_window,
        'patch_size': list(args.patch_size) if args.use_sliding_window else None,
        'overlap': args.overlap if args.use_sliding_window else None,
        'compute_lpips': args.compute_lpips,
        'lpips_backend': args.lpips_backend if args.compute_lpips else None,
        'evaluation_date': datetime.now().isoformat(),
        'num_volumes': len(per_volume_results),
    }

    # Save results
    csv_path = args.csv_output or os.path.join(args.output_dir, 'results.csv')
    json_path = args.json_output or os.path.join(args.output_dir, 'results.json')

    save_csv_results(csv_path, per_volume_results, aggregate_stats)
    save_json_results(json_path, per_volume_results, aggregate_stats, metadata)
    print_evaluation_summary(aggregate_stats)

    print(f"\nEvaluation complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
