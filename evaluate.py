"""
Calculate Metrics for Pre-Generated Model Predictions

This script computes evaluation metrics between saved model predictions and ground truth.
It expects predictions and ground truth to be saved in subject directories.

Directory Structure:
    test_dir/
    ├── subject001/
    │   ├── HR_groundtruth.nii.gz
    │   └── {prediction_name}.nii.gz
    ├── subject002/
    │   ├── HR_groundtruth.nii.gz
    │   └── {prediction_name}.nii.gz
    └── ...

Usage:
    python evaluate.py \\
        --test_dir /path/to/test \\
        --prediction_name model1_sr.nii.gz \\
        --output_csv results.csv \\
        --output_json results.json
"""

import os
import argparse
import csv
import json
import logging
from glob import glob
from typing import Dict, Any, List

import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm

# Import metrics from project
from src.utils import calculate_metrics


def load_nifti_volume(file_path: str) -> np.ndarray:
    """
    Load and normalize a NIfTI volume.

    Args:
        file_path: Path to .nii or .nii.gz file

    Returns:
        Normalized numpy array (values in [0, 1])
    """
    nii = nib.load(file_path)
    volume = nii.get_fdata()

    # Normalize to [0, 1]
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

    return volume


def find_subject_pairs(test_dir: str, prediction_name: str, gt_name: str = "HR_groundtruth.nii.gz") -> List[tuple]:
    """
    Find all subject directories with both ground truth and predictions.

    Args:
        test_dir: Root directory containing subject subdirectories
        prediction_name: Name of prediction file (e.g., "model1_sr.nii.gz")
        gt_name: Name of ground truth file (default: "HR_groundtruth.nii.gz")

    Returns:
        List of tuples: [(subject_id, gt_path, pred_path), ...]
    """
    pairs = []

    # Iterate through subdirectories
    for subject_dir in sorted(os.listdir(test_dir)):
        subject_path = os.path.join(test_dir, subject_dir)

        if not os.path.isdir(subject_path):
            continue

        gt_path = os.path.join(subject_path, gt_name)
        pred_path = os.path.join(subject_path, prediction_name)

        # Check both files exist
        if os.path.exists(gt_path) and os.path.exists(pred_path):
            pairs.append((subject_dir, gt_path, pred_path))
        else:
            missing = []
            if not os.path.exists(gt_path):
                missing.append(gt_name)
            if not os.path.exists(pred_path):
                missing.append(prediction_name)
            logging.warning(f"Skipping {subject_dir}: missing {', '.join(missing)}")

    return pairs


def center_crop_to_match(volume1: np.ndarray, volume2: np.ndarray) -> tuple:
    """
    Center crop both volumes to their minimum overlapping size.

    Args:
        volume1: First volume (e.g., ground truth)
        volume2: Second volume (e.g., prediction)

    Returns:
        Tuple of (cropped_volume1, cropped_volume2)
    """
    shape1 = np.array(volume1.shape)
    shape2 = np.array(volume2.shape)

    # Determine minimum shape along each dimension
    min_shape = np.minimum(shape1, shape2)

    # Calculate crop indices for volume1
    start1 = (shape1 - min_shape) // 2
    end1 = start1 + min_shape

    # Calculate crop indices for volume2
    start2 = (shape2 - min_shape) // 2
    end2 = start2 + min_shape

    # Perform center crop
    cropped1 = volume1[start1[0]:end1[0], start1[1]:end1[1], start1[2]:end1[2]]
    cropped2 = volume2[start2[0]:end2[0], start2[1]:end2[1], start2[2]:end2[2]]

    return cropped1, cropped2


def compute_metrics_for_pair(
    gt_path: str,
    pred_path: str,
    device: str = 'cuda',
    disable_intensity_filter: bool = False,
    eval_thr_hi: float = 0.2,
    eval_thr_lo: float = 0.1,
    center_keep_radius: float = 0.45,
    min_eval_voxels: int = 1000,
    use_perceptual: bool = False,
    perceptual_network: str = 'alex',
    is_fake_3d: bool = True
) -> Dict[str, float]:

    gt_volume = load_nifti_volume(gt_path)
    pred_volume = load_nifti_volume(pred_path)

    if gt_volume.shape != pred_volume.shape:
        logging.warning(
            f"Shape mismatch detected: GT {gt_volume.shape} vs Pred {pred_volume.shape}. "
            f"Applying center crop to minimum size."
        )
        gt_volume, pred_volume = center_crop_to_match(gt_volume, pred_volume)
        logging.info(f"Cropped to common shape: {gt_volume.shape}")

    # Build center-aware eval mask (or use all voxels if filtering is disabled)
    if disable_intensity_filter:
        # Evaluate all voxels when filtering is disabled
        eval_mask = np.ones(pred_volume.shape, dtype=bool)
        num_vox = int(np.prod(pred_volume.shape))
    else:
        # Build center-aware eval mask:
        # - always keep pred > hi
        # - also keep pred in (lo, hi] if inside the central ellipsoid
        center_mask = central_ellipsoid_mask(pred_volume.shape, radius_frac=center_keep_radius)

        eval_mask = (pred_volume > eval_thr_hi) | (
            (pred_volume > eval_thr_lo) & (pred_volume <= eval_thr_hi) & center_mask
        )

        num_vox = int(np.count_nonzero(eval_mask))
        if num_vox < min_eval_voxels:
            raise RuntimeError(
                f"Too few voxels to evaluate after masking: {num_vox} < {min_eval_voxels}. "
                f"(hi={eval_thr_hi}, lo={eval_thr_lo}, center_radius={center_keep_radius})"
            )

    # Zero-out outside-mask voxels in BOTH volumes (keeps tensor shapes for SSIM, etc.)
    gt_volume_f = gt_volume.copy()
    pred_volume_f = pred_volume.copy()
    gt_volume_f[~eval_mask] = 0.0
    pred_volume_f[~eval_mask] = 0.0

    gt_tensor = torch.from_numpy(gt_volume_f).float().unsqueeze(0).unsqueeze(0)
    pred_tensor = torch.from_numpy(pred_volume_f).float().unsqueeze(0).unsqueeze(0)

    metrics = calculate_metrics(
        pred_tensor,
        gt_tensor,
        max_val=1.0,
        use_perceptual=use_perceptual,
        perceptual_network=perceptual_network,
        is_fake_3d=is_fake_3d
    )
    metrics["eval_voxels"] = float(num_vox)
    metrics["intensity_filter_disabled"] = bool(disable_intensity_filter)
    if not disable_intensity_filter:
        metrics["eval_thr_hi"] = float(eval_thr_hi)
        metrics["eval_thr_lo"] = float(eval_thr_lo)
        metrics["center_keep_radius"] = float(center_keep_radius)

    return metrics


def central_ellipsoid_mask(shape: tuple, radius_frac: float = 0.45) -> np.ndarray:
    """
    Create a central ellipsoid mask.
    radius_frac is relative to half-size along each axis.
      - 1.0 means it reaches the borders
      - 0.5 means half of the half-size (i.e., quarter of full size)
    """
    assert len(shape) == 3
    cx, cy, cz = (np.array(shape) - 1) / 2.0
    rx, ry, rz = (np.array(shape) / 2.0) * float(radius_frac)

    # Avoid division by zero
    rx = max(rx, 1e-6); ry = max(ry, 1e-6); rz = max(rz, 1e-6)

    x = np.arange(shape[0])[:, None, None]
    y = np.arange(shape[1])[None, :, None]
    z = np.arange(shape[2])[None, None, :]

    ell = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 + ((z - cz) / rz) ** 2 <= 1.0
    return ell

def aggregate_metrics(volume_results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Compute aggregate statistics across all volumes.

    Args:
        volume_results: List of per-volume metric dictionaries

    Returns:
        Dictionary of {metric_name: {mean, std, median, min, max}, ...}
    """
    if not volume_results:
        return {}

    # Get all metric names from first result
    metric_names = list(volume_results[0].keys())

    aggregates = {}
    for metric_name in metric_names:
        values = [r[metric_name] for r in volume_results]
        aggregates[metric_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }

    return aggregates


def save_csv_results(
    volume_results: List[Dict[str, float]],
    subject_ids: List[str],
    aggregates: Dict[str, Dict[str, float]],
    output_path: str
):
    """
    Save per-volume and aggregate results to CSV.

    Args:
        volume_results: List of per-volume metric dictionaries
        subject_ids: List of subject IDs
        aggregates: Aggregate statistics
        output_path: Path to save CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Per-volume section
        writer.writerow(['# Per-Volume Metrics'])

        # Get metric names from first result
        if volume_results:
            metric_names = list(volume_results[0].keys())
            headers = ['subject_id'] + metric_names
            writer.writerow(headers)

            for subject_id, result in zip(subject_ids, volume_results):
                row = [subject_id] + [f"{result[m]:.6f}" for m in metric_names]
                writer.writerow(row)

        # Aggregate section
        writer.writerow([])
        writer.writerow(['# Aggregate Statistics'])
        writer.writerow(['metric', 'mean', 'std', 'median', 'min', 'max'])

        for metric_name, stats in aggregates.items():
            writer.writerow([
                metric_name,
                f"{stats['mean']:.6f}",
                f"{stats['std']:.6f}",
                f"{stats['median']:.6f}",
                f"{stats['min']:.6f}",
                f"{stats['max']:.6f}"
            ])

    logging.info(f"CSV results saved to: {output_path}")


def save_json_results(
    volume_results: List[Dict[str, float]],
    subject_ids: List[str],
    aggregates: Dict[str, Dict[str, float]],
    output_path: str,
    metadata: Dict[str, Any]
):
    """
    Save structured JSON results.

    Args:
        volume_results: List of per-volume metric dictionaries
        subject_ids: List of subject IDs
        aggregates: Aggregate statistics
        output_path: Path to save JSON file
        metadata: Additional metadata to include
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_data = {
        'metadata': metadata,
        'per_volume_results': [
            {
                'subject_id': subject_id,
                'metrics': result
            }
            for subject_id, result in zip(subject_ids, volume_results)
        ],
        'aggregate_statistics': aggregates
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logging.info(f"JSON results saved to: {output_path}")


def print_summary(aggregates: Dict[str, Dict[str, float]], num_volumes: int):
    """Print formatted summary to console."""
    print("\n" + "=" * 80)
    print("METRICS SUMMARY")
    print("=" * 80)
    print(f"Number of volumes evaluated: {num_volumes}")
    print("\n" + "-" * 80)
    print(f"{'Metric':<15} {'Mean ± Std':<25} {'Range':<30}")
    print("-" * 80)

    for metric_name, stats in aggregates.items():
        mean_std = f"{stats['mean']:.6f} ± {stats['std']:.6f}"
        range_str = f"[{stats['min']:.6f} - {stats['max']:.6f}]"
        print(f"{metric_name:<15} {mean_std:<25} {range_str:<30}")

    print("=" * 80 + "\n")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate metrics between saved predictions and ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate model predictions
  python evaluate.py \\
    --test_dir /data/test \\
    --prediction_name model1_sr.nii.gz \\
    --output_csv model1_metrics.csv \\
    --output_json model1_metrics.json

"""
    )

    # Input/Output arguments
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Directory containing subject subdirectories')
    parser.add_argument('--prediction_name', type=str, required=True,
                        help='Name of prediction file in each subject directory (e.g., "model1_sr.nii.gz")')
    parser.add_argument('--gt_name', type=str, default='HR_groundtruth.nii.gz',
                        help='Name of ground truth file (default: HR_groundtruth.nii.gz)')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Path to save CSV results')
    parser.add_argument('--output_json', type=str,
                        help='Path to save JSON results (optional)')
    # Center-aware artifact filtering
    parser.add_argument('--disable_intensity_filter', action='store_true',
                        help='Disable intensity-based filtering and evaluate all voxels (default: False)')
    parser.add_argument('--eval_thr_hi', type=float, default=0.25,
                        help='Primary threshold: always evaluate voxels where pred > eval_thr_hi (default 0.25)')
    parser.add_argument('--eval_thr_lo', type=float, default=0.15,
                        help='Secondary threshold: evaluate voxels where pred > eval_thr_lo AND inside center region (default 0.15)')
    parser.add_argument('--center_keep_radius', type=float, default=0.35,
                        help=("Keep low-intensity voxels within a central ellipsoid. "
                              "Value is fraction of half-size per axis (0-1). Default 0.35"))
    parser.add_argument('--min_eval_voxels', type=int, default=1000,
                        help='Skip a volume if fewer than this many voxels are evaluated (default 1000)')

    # Perceptual loss arguments
    parser.add_argument('--use_perceptual', action='store_true',
                        help='Compute MONAI perceptual loss metric')
    parser.add_argument('--perceptual_network', type=str, default='alex',
                        choices=['alex', 'vgg', 'squeeze', 'radimagenet', 'medicalnet', 'resnet50'],
                        help='Network backbone for perceptual loss (default: alex)')
    parser.add_argument('--perceptual_fake_3d', action='store_true', default=True,
                        help='Use 2.5D slice-based processing for perceptual loss (default: True, faster)')

    # Computation arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for computation')

    # Output control
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-volume results')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info("=" * 80)
    logging.info("Metrics Calculation Script")
    logging.info("=" * 80)
    logging.info(f"Test directory: {args.test_dir}")
    logging.info(f"Prediction name: {args.prediction_name}")
    logging.info(f"Ground truth name: {args.gt_name}")
    logging.info(f"Device: {args.device}")

    # Find all subject pairs
    logging.info("\nSearching for subject pairs...")
    subject_pairs = find_subject_pairs(args.test_dir, args.prediction_name, args.gt_name)

    if not subject_pairs:
        logging.error("No valid subject pairs found!")
        return

    logging.info(f"Found {len(subject_pairs)} subject pairs\n")

    # Compute metrics for each pair
    volume_results = []
    subject_ids = []

    for subject_id, gt_path, pred_path in tqdm(subject_pairs, desc="Computing metrics"):
        try:
            metrics = compute_metrics_for_pair(
                        gt_path=gt_path,
                        pred_path=pred_path,
                        device=args.device,
                        disable_intensity_filter=args.disable_intensity_filter,
                        eval_thr_hi=args.eval_thr_hi,
                        eval_thr_lo=args.eval_thr_lo,
                        center_keep_radius=args.center_keep_radius,
                        min_eval_voxels=args.min_eval_voxels,
                        use_perceptual=args.use_perceptual,
                        perceptual_network=args.perceptual_network,
                        is_fake_3d=args.perceptual_fake_3d
                    )


            volume_results.append(metrics)
            subject_ids.append(subject_id)

            if args.verbose:
                logging.info(f"\n{subject_id}:")
                for metric_name, value in metrics.items():
                    logging.info(f"  {metric_name}: {value:.6f}")

        except Exception as e:
            logging.error(f"Failed to process {subject_id}: {str(e)}")
            continue

    if not volume_results:
        logging.error("No results computed successfully!")
        return

    # Compute aggregate statistics
    logging.info("\nComputing aggregate statistics...")
    aggregates = aggregate_metrics(volume_results)

    # Save CSV results
    logging.info(f"\nSaving results...")
    save_csv_results(
        volume_results=volume_results,
        subject_ids=subject_ids,
        aggregates=aggregates,
        output_path=args.output_csv
    )

    # Save JSON results if requested
    if args.output_json:
        metadata = {
            'test_dir': args.test_dir,
            'prediction_name': args.prediction_name,
            'gt_name': args.gt_name,
            'num_volumes': len(volume_results),
            'device': args.device,
            'intensity_filter_disabled': args.disable_intensity_filter,
            'eval_thr_hi': args.eval_thr_hi if not args.disable_intensity_filter else None,
            'eval_thr_lo': args.eval_thr_lo if not args.disable_intensity_filter else None,
            'center_keep_radius': args.center_keep_radius if not args.disable_intensity_filter else None,
            'use_perceptual': args.use_perceptual,
            'perceptual_network': args.perceptual_network if args.use_perceptual else None,
            'perceptual_fake_3d': args.perceptual_fake_3d if args.use_perceptual else None
        }
        save_json_results(
            volume_results=volume_results,
            subject_ids=subject_ids,
            aggregates=aggregates,
            output_path=args.output_json,
            metadata=metadata
        )

    # Print summary
    print_summary(aggregates, len(volume_results))

    logging.info("=" * 80)
    logging.info("COMPUTATION COMPLETE")
    logging.info("=" * 80)


if __name__ == '__main__':
    main()
