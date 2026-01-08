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
    python calc_metrics.py \\
        --test_dir /path/to/test \\
        --prediction_name model1_sr.nii.gz \\
        --output_csv results.csv \\
        --output_json results.json \\
        --compute_lpips
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
from src.utils import calculate_metrics_with_lpips


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


def compute_metrics_for_pair(
    gt_path: str,
    pred_path: str,
    compute_lpips: bool = True,
    lpips_backend: str = 'monai',
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Compute metrics between ground truth and prediction.

    Args:
        gt_path: Path to ground truth NIfTI file
        pred_path: Path to prediction NIfTI file
        compute_lpips: Whether to compute LPIPS metrics
        lpips_backend: Backend for LPIPS computation
        device: Device for computation

    Returns:
        Dictionary of metrics
    """
    # Load volumes
    gt_volume = load_nifti_volume(gt_path)
    pred_volume = load_nifti_volume(pred_path)

    # Validate shapes match
    if gt_volume.shape != pred_volume.shape:
        raise ValueError(
            f"Shape mismatch: GT {gt_volume.shape} vs Pred {pred_volume.shape}"
        )

    # Convert to tensors with batch and channel dimensions
    gt_tensor = torch.from_numpy(gt_volume).float().unsqueeze(0).unsqueeze(0)
    pred_tensor = torch.from_numpy(pred_volume).float().unsqueeze(0).unsqueeze(0)

    # Compute metrics
    metrics = calculate_metrics_with_lpips(
        pred_tensor,
        gt_tensor,
        max_val=1.0,
        compute_lpips=compute_lpips,
        lpips_backend=lpips_backend,
        device=device
    )

    return metrics


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
  python calc_metrics.py \\
    --test_dir /data/test \\
    --prediction_name model1_sr.nii.gz \\
    --output_csv model1_metrics.csv \\
    --output_json model1_metrics.json

  # Evaluate without LPIPS metrics
  python calc_metrics.py \\
    --test_dir /data/test \\
    --prediction_name model2_output.nii.gz \\
    --output_csv model2_metrics.csv \\
    --no_lpips
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

    # Computation arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for computation')
    parser.add_argument('--compute_lpips', action='store_true', default=True,
                        help='Compute LPIPS metrics (default: True)')
    parser.add_argument('--no_lpips', dest='compute_lpips', action='store_false',
                        help='Disable LPIPS metrics computation')
    parser.add_argument('--lpips_backend', type=str, default='monai',
                        choices=['monai', 'medicalnet', 'models_genesis'],
                        help='Backend for LPIPS computation (default: monai)')

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
    logging.info(f"Compute LPIPS: {args.compute_lpips}")

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
                compute_lpips=args.compute_lpips,
                lpips_backend=args.lpips_backend,
                device=args.device
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
            'compute_lpips': args.compute_lpips,
            'lpips_backend': args.lpips_backend if args.compute_lpips else None,
            'device': args.device
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
