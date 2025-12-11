#!/usr/bin/env python3
"""
Verification script for U-HVED Data Generation Pipeline

This script tests and saves examples of the low-resolution orthogonal stacks
produced by the HRLRDataGenerator to verify the data pipeline is working
correctly.

Usage:
    python verify_data_pipeline.py --input /path/to/hr_volume.nii.gz --output ./verification_output
    python verify_data_pipeline.py --input_dir /path/to/hr_volumes --output ./verification_output --num_samples 5
"""

import os
import argparse
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict
import json
from datetime import datetime

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRangePercentilesd,
)

from src.data import HRLRDataGenerator
from src.utils import calculate_metrics


def create_verification_transforms(target_res=[1.0, 1.0, 1.0], target_shape=[128, 128, 128]):
    """Create transforms for loading and preprocessing HR volumes (using dict-based transforms)."""
    return Compose([
        LoadImaged(keys=["image"], image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS", labels=None),
        Spacingd(keys=["image"], pixdim=target_res, mode="bilinear"),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
    ])


def load_and_preprocess_volume(volume_path: str, target_res: List[float], target_shape: List[int]) -> Tuple[torch.Tensor, Dict]:
    """Load and preprocess a single HR volume using dict-based transforms."""
    transforms = create_verification_transforms(target_res, target_shape)

    # Create dict input for dict-based transforms
    data_dict = {"image": volume_path}

    # Apply transforms
    transformed = transforms(data_dict)

    # Extract volume and metadata
    volume = transformed["image"]
    metadata = {k: v for k, v in transformed.items() if k != "image"}

    # Ensure correct shape
    if volume.ndim == 4 and volume.shape[0] == 1:
        volume = volume.squeeze(0)  # Remove channel dim for processing

    # Crop/pad to target shape if needed
    current_shape = volume.shape
    if list(current_shape) != target_shape:
        print(f"  Warning: Volume shape {current_shape} differs from target {target_shape}")
        # Simple center crop for verification
        volume = volume[:target_shape[0], :target_shape[1], :target_shape[2]]

    return volume, metadata


def generate_orthogonal_stacks_with_info(
    hr_volume: torch.Tensor,
    generator: HRLRDataGenerator,
    batch_idx: int = 0
) -> Tuple[List[torch.Tensor], Dict]:
    """
    Generate orthogonal LR stacks and collect detailed information.
    
    Returns:
        Tuple of (lr_stacks, metadata_dict)
    """
    # Ensure proper shape: (batch=1, channels=1, D, H, W)
    if hr_volume.ndim == 3:
        # (D, H, W) -> (1, 1, D, H, W)
        hr_volume = hr_volume.unsqueeze(0).unsqueeze(0)
    elif hr_volume.ndim == 4 and hr_volume.shape[0] == 1:
        # (1, D, H, W) -> (1, 1, D, H, W)
        hr_volume = hr_volume.unsqueeze(1)
    elif hr_volume.ndim == 5:
        # Already (1, 1, D, H, W)
        pass
    else:
        raise ValueError(f"Unexpected volume shape: {hr_volume.shape}")
    
    # Generate paired data with resolution info
    result = generator.generate_paired_data(
        hr_volume,
        return_resolution=True
    )
    
    # Handle different return formats
    if len(result) == 5:
        lr_stacks, hr_augmented, resolutions, thicknesses, orientation_mask = result
    elif len(result) == 3:
        lr_stacks, hr_augmented, orientation_mask = result
        # Create dummy resolution/thickness data
        resolutions = [[1.0, 1.0, 1.0] for _ in range(3)]
        thicknesses = [[1.0, 1.0, 1.0] for _ in range(3)]
    else:
        raise ValueError(f"Unexpected result format: {len(result)} items")
    
    # Collect metadata
    metadata = {
        'batch_idx': batch_idx,
        'input_shape': list(hr_volume.shape),
        'output_shape': list(lr_stacks[0].shape),
        'orientation_mask': orientation_mask.tolist() if hasattr(orientation_mask, 'tolist') else orientation_mask,
        'resolutions': {
            'axial': resolutions[0] if isinstance(resolutions[0], list) else resolutions[0].tolist() if hasattr(resolutions[0], 'tolist') else [1.0, 1.0, 1.0],
            'coronal': resolutions[1] if isinstance(resolutions[1], list) else resolutions[1].tolist() if hasattr(resolutions[1], 'tolist') else [1.0, 1.0, 1.0],
            'sagittal': resolutions[2] if isinstance(resolutions[2], list) else resolutions[2].tolist() if hasattr(resolutions[2], 'tolist') else [1.0, 1.0, 1.0],
        },
        'thicknesses': {
            'axial': thicknesses[0] if isinstance(thicknesses[0], list) else thicknesses[0].tolist() if hasattr(thicknesses[0], 'tolist') else [1.0, 1.0, 1.0],
            'coronal': thicknesses[1] if isinstance(thicknesses[1], list) else thicknesses[1].tolist() if hasattr(thicknesses[1], 'tolist') else [1.0, 1.0, 1.0],
            'sagittal': thicknesses[2] if isinstance(thicknesses[2], list) else thicknesses[2].tolist() if hasattr(thicknesses[2], 'tolist') else [1.0, 1.0, 1.0],
        },
        'hr_stats': {
            'min': float(hr_augmented.min()),
            'max': float(hr_augmented.max()),
            'mean': float(hr_augmented.mean()),
            'std': float(hr_augmented.std()),
        },
        'lr_stats': []
    }
    
    # Calculate stats for each LR stack
    stack_names = ['axial', 'coronal', 'sagittal']
    for i, (stack, name) in enumerate(zip(lr_stacks, stack_names)):
        stack_np = stack.squeeze().cpu().numpy()
        stats = {
            'name': name,
            'min': float(stack_np.min()),
            'max': float(stack_np.max()),
            'mean': float(stack_np.mean()),
            'std': float(stack_np.std()),
            'shape': list(stack_np.shape),
            'is_present': bool(metadata['orientation_mask'][i]) if i < len(metadata['orientation_mask']) else True,
        }
        metadata['lr_stats'].append(stats)
    
    return lr_stacks, metadata


def save_volume_as_nifti(volume: torch.Tensor, filepath: str, affine: np.ndarray = None):
    """Save a volume tensor as NIfTI file."""
    # Convert to numpy
    if isinstance(volume, torch.Tensor):
        volume_np = volume.squeeze().cpu().numpy()
    else:
        volume_np = np.array(volume)
    
    # Create affine if not provided
    if affine is None:
        affine = np.eye(4)
    
    # Create and save NIfTI
    nii = nib.Nifti1Image(volume_np, affine)
    nib.save(nii, filepath)


def create_comparison_slices(
    hr_volume: torch.Tensor,
    lr_stacks: List[torch.Tensor],
    output_dir: Path,
    sample_name: str,
    slice_indices: Dict[str, int] = None
):
    """
    Create and save comparison slice images showing HR vs LR stacks.

    After RAS orientation:
      Axis 0 = R (Right-Left)      → Sagittal slices
      Axis 1 = A (Anterior-Post.)  → Coronal slices
      Axis 2 = S (Superior-Inf.)   → Axial slices

    Args:
        hr_volume: High-resolution volume (D, H, W) in RAS orientation
        lr_stacks: List of 3 LR stacks in RAS orientation
        output_dir: Output directory
        sample_name: Sample identifier
        slice_indices: Dict with slice indices for each orientation
    """
    if slice_indices is None:
        # Default to middle slices
        D, H, W = hr_volume.shape
        slice_indices = {
            'sagittal': D // 2,  # Axis 0 = R → Sagittal
            'coronal': H // 2,   # Axis 1 = A → Coronal
            'axial': W // 2      # Axis 2 = S → Axial
        }
    
    # Convert to numpy for plotting
    hr_np = hr_volume.cpu().numpy()
    lr_np = [stack.squeeze().cpu().numpy() for stack in lr_stacks]

    # Create comparison plots
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'{sample_name} - HR vs Orthogonal LR Stacks (RAS Orientation)', fontsize=16)

    # Stack 0 = Axial, Stack 1 = Coronal, Stack 2 = Sagittal
    stack_names = ['Stack 0 (Axial)', 'Stack 1 (Coronal)', 'Stack 2 (Sagittal)']

    # Row 1: AXIAL slices (slice along axis 2 = S direction)
    axes[0, 0].imshow(hr_np[:, :, slice_indices['axial']], cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('HR Axial View')
    axes[0, 0].axis('off')

    for i, (lr_stack, name) in enumerate(zip(lr_np, stack_names)):
        axes[0, i+1].imshow(lr_stack[:, :, slice_indices['axial']], cmap='gray', vmin=0, vmax=1)
        axes[0, i+1].set_title(f'LR {name}')
        axes[0, i+1].axis('off')

    # Row 2: CORONAL slices (slice along axis 1 = A direction)
    axes[1, 0].imshow(hr_np[:, slice_indices['coronal'], :], cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('HR Coronal View')
    axes[1, 0].axis('off')

    for i, (lr_stack, name) in enumerate(zip(lr_np, stack_names)):
        axes[1, i+1].imshow(lr_stack[:, slice_indices['coronal'], :], cmap='gray', vmin=0, vmax=1)
        axes[1, i+1].set_title(f'LR {name}')
        axes[1, i+1].axis('off')

    # Row 3: SAGITTAL slices (slice along axis 0 = R direction)
    axes[2, 0].imshow(hr_np[slice_indices['sagittal'], :, :], cmap='gray', vmin=0, vmax=1)
    axes[2, 0].set_title('HR Sagittal View')
    axes[2, 0].axis('off')

    for i, (lr_stack, name) in enumerate(zip(lr_np, stack_names)):
        axes[2, i+1].imshow(lr_stack[slice_indices['sagittal'], :, :], cmap='gray', vmin=0, vmax=1)
        axes[2, i+1].set_title(f'LR {name}')
        axes[2, i+1].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save comparison
    comparison_path = output_dir / f'{sample_name}_comparison.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved comparison image: {comparison_path}")


def create_resolution_analysis_plot(
    metadata: Dict,
    output_dir: Path,
    sample_name: str
):
    """Create a plot showing the resolution configurations.

    After RAS orientation:
      Axis 0 (D) = R (Right-Left)
      Axis 1 (H) = A (Anterior-Posterior)
      Axis 2 (W) = S (Superior-Inferior)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{sample_name} - Resolution Analysis (RAS Orientation)', fontsize=16)

    # Extract resolution data
    resolutions = metadata['resolutions']
    axes_labels = ['Axis 0 (R)', 'Axis 1 (A)', 'Axis 2 (S)']
    stack_names = ['Axial', 'Coronal', 'Sagittal']
    
    # Plot 1: Resolution per axis for each stack
    x = np.arange(len(axes_labels))
    width = 0.25
    
    for i, stack_name in enumerate(stack_names):
        res_values = resolutions[stack_name.lower()]
        # Ensure res_values is a list of scalars
        if isinstance(res_values, (list, tuple)):
            res_values = [float(v) if isinstance(v, (int, float, np.number)) else v[0] if isinstance(v, (list, tuple)) else v for v in res_values]
        ax1.bar(x + i*width, res_values, width, label=stack_name)
    
    ax1.set_xlabel('Spatial Axis')
    ax1.set_ylabel('Resolution (mm)')
    ax1.set_title('Resolution per Axis')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(axes_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Anisotropy factors
    anisotropy_factors = []
    for stack_name in stack_names:
        res_values = resolutions[stack_name.lower()]
        # Ensure res_values is a list of scalars
        if isinstance(res_values, (list, tuple)):
            res_values = [float(v) if isinstance(v, (int, float, np.number)) else v[0] if isinstance(v, (list, tuple)) else v for v in res_values]
        max_res = max(res_values)
        min_res = min(res_values)
        anisotropy = max_res / min_res if min_res > 0 else 1.0
        anisotropy_factors.append(anisotropy)
    
    ax2.bar(stack_names, anisotropy_factors)
    ax2.set_ylabel('Anisotropy Factor (max/min)')
    ax2.set_title('Resolution Anisotropy')
    ax2.grid(True, alpha=0.3)
    
    # Add values on bars
    for i, factor in enumerate(anisotropy_factors):
        ax2.text(i, factor + 0.1, f'{factor:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f'{sample_name}_resolution_analysis.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved resolution analysis: {plot_path}")


def verify_single_volume(
    volume_path: str,
    generator: HRLRDataGenerator,
    output_dir: Path,
    sample_idx: int = 0,
    save_nifti: bool = True,
    save_plots: bool = True
) -> Dict:
    """Verify data generation on a single volume."""
    volume_name = Path(volume_path).stem
    sample_name = f"sample_{sample_idx:03d}_{volume_name}"
    sample_dir = output_dir / sample_name
    sample_dir.mkdir(exist_ok=True)
    
    print(f"\n[{sample_idx + 1}] Processing: {volume_name}")
    print(f"  Output directory: {sample_dir}")
    
    # Load and preprocess HR volume
    print("  Loading HR volume...")
    hr_volume, metadata = load_and_preprocess_volume(
        volume_path,
        target_res=generator.atlas_res,
        target_shape=generator.output_shape
    )
    print(f"  HR volume shape: {hr_volume.shape}")
    
    # Generate orthogonal LR stacks
    print("  Generating orthogonal LR stacks...")
    lr_stacks, gen_metadata = generate_orthogonal_stacks_with_info(
        hr_volume, generator, sample_idx
    )
    
    # Update metadata with volume info
    gen_metadata.update({
        'volume_name': volume_name,
        'volume_path': str(volume_path),
        'sample_name': sample_name,
        'generator_config': {
            'atlas_res': generator.atlas_res,
            'target_res': generator.target_res,
            'output_shape': generator.output_shape,
            'min_resolution': generator.min_res if hasattr(generator, 'min_res') else None,
            'max_res_aniso': generator.max_res_aniso if hasattr(generator, 'max_res_aniso') else None,
            'randomise_res': generator.randomise_res,
        }
    })
    
    print(f"  Generated {len(lr_stacks)} LR stacks")
    for i, stats in enumerate(gen_metadata['lr_stats']):
        status = "✓ Present" if stats['is_present'] else "✗ Dropped"
        print(f"    {stats['name']}: {stats['shape']} - {status}")
    
    # Save volumes as NIfTI if requested
    if save_nifti:
        print("  Saving NIfTI files...")
        # Save HR volume
        hr_path = sample_dir / f'{sample_name}_hr.nii.gz'
        save_volume_as_nifti(hr_volume, str(hr_path))
        print(f"    ✓ HR: {hr_path}")
        
        # Save LR stacks
        stack_names = ['axial', 'coronal', 'sagittal']
        for i, (stack, name) in enumerate(zip(lr_stacks, stack_names)):
            if gen_metadata['lr_stats'][i]['is_present']:
                lr_path = sample_dir / f'{sample_name}_lr_{name}.nii.gz'
                save_volume_as_nifti(stack, str(lr_path))
                print(f"    ✓ LR {name}: {lr_path}")
    
    # Create comparison plots if requested
    if save_plots:
        print("  Creating visualization plots...")
        try:
            create_comparison_slices(hr_volume, lr_stacks, sample_dir, sample_name)
            create_resolution_analysis_plot(gen_metadata, sample_dir, sample_name)
        except Exception as e:
            print(f"    ⚠️  Plot creation failed: {e}")
    
    # Save metadata
    metadata_path = sample_dir / f'{sample_name}_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(gen_metadata, f, indent=2, default=str)
    print(f"  ✓ Saved metadata: {metadata_path}")
    
    return gen_metadata


def create_summary_report(all_metadata: List[Dict], output_dir: Path):
    """Create a summary report of all verification samples."""
    print("\nCreating summary report...")
    
    # Calculate statistics
    total_samples = len(all_metadata)
    total_orientations = sum(len(m['lr_stats']) for m in all_metadata)
    dropped_orientations = sum(sum(1 for stats in m['lr_stats'] if not stats['is_present']) for m in all_metadata)
    
    # Resolution statistics
    all_resolutions = []
    anisotropy_factors = []
    
    for metadata in all_metadata:
        for stack_name in ['axial', 'coronal', 'sagittal']:
            res_values = metadata['resolutions'][stack_name]
            # Ensure res_values are scalars
            if isinstance(res_values, (list, tuple)):
                res_values = [float(v) if isinstance(v, (int, float, np.number)) else v[0] if isinstance(v, (list, tuple)) else v for v in res_values]
            all_resolutions.extend(res_values)
            
            max_res = max(res_values)
            min_res = min(res_values)
            anisotropy = max_res / min_res if min_res > 0 else 1.0
            anisotropy_factors.append(anisotropy)
    
    # Create summary
    summary = {
        'verification_timestamp': datetime.now().isoformat(),
        'total_samples': total_samples,
        'total_orientations_generated': total_orientations,
        'dropped_orientations': dropped_orientations,
        'orientation_dropout_rate': dropped_orientations / total_orientations if total_orientations > 0 else 0,
        'resolution_statistics': {
            'min_resolution': min(all_resolutions),
            'max_resolution': max(all_resolutions),
            'mean_resolution': np.mean(all_resolutions),
            'std_resolution': np.std(all_resolutions),
        },
        'anisotropy_statistics': {
            'min_anisotropy': min(anisotropy_factors),
            'max_anisotropy': max(anisotropy_factors),
            'mean_anisotropy': np.mean(anisotropy_factors),
            'std_anisotropy': np.std(anisotropy_factors),
        },
        'samples': [m['sample_name'] for m in all_metadata]
    }
    
    # Save summary
    summary_path = output_dir / 'verification_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"✓ Summary report saved: {summary_path}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    print(f"Total samples processed: {total_samples}")
    print(f"Total orientations generated: {total_orientations}")
    print(f"Dropped orientations: {dropped_orientations} ({summary['orientation_dropout_rate']:.2%})")
    print(f"Resolution range: {summary['resolution_statistics']['min_resolution']:.2f} - {summary['resolution_statistics']['max_resolution']:.2f} mm")
    print(f"Mean anisotropy factor: {summary['anisotropy_statistics']['mean_anisotropy']:.2f}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Verify U-HVED data generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single volume verification
    python verify_data_pipeline.py --input /path/to/volume.nii.gz --output ./verification
    
    # Batch verification
    python verify_data_pipeline.py --input_dir /path/to/volumes --output ./verification --num_samples 5
    
    # Custom configuration
    python verify_data_pipeline.py --input /path/to/volume.nii.gz --output ./verification \\
        --output_shape 96 96 96 --max_res_aniso 6.0 6.0 6.0 --orientation_dropout_prob 0.3
        """
    )
    
    # Input arguments
    parser.add_argument("--input", type=str, help="Single HR volume file")
    parser.add_argument("--input_dir", type=str, help="Directory containing HR volumes")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to process (for batch mode)")
    
    # Data generation configuration
    parser.add_argument("--output_shape", type=int, nargs=3, default=[128, 128, 128], 
                       help="Output volume shape (D H W)")
    parser.add_argument("--atlas_res", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                       help="Atlas resolution in mm (x y z)")
    parser.add_argument("--min_resolution", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                       help="Minimum resolution in mm (x y z)")
    parser.add_argument("--max_res_aniso", type=float, nargs=3, default=[9.0, 9.0, 9.0],
                       help="Maximum anisotropic resolution in mm (x y z)")
    parser.add_argument("--no_randomise_res", action="store_true", help="Disable resolution randomization")
    
    # Artifact probabilities
    parser.add_argument("--prob_motion", type=float, default=0.2, help="Motion artifact probability")
    parser.add_argument("--prob_spike", type=float, default=0.05, help="K-space spike probability")
    parser.add_argument("--prob_aliasing", type=float, default=0.1, help="Aliasing artifact probability")
    parser.add_argument("--prob_bias_field", type=float, default=0.5, help="Bias field probability")
    parser.add_argument("--prob_noise", type=float, default=0.8, help="Noise probability")
    parser.add_argument("--no_intensity_aug", action="store_true", help="Disable intensity augmentation")
    
    # orientation dropout
    parser.add_argument("--orientation_dropout_prob", type=float, default=0.0,
                       help="Probability of orientation dropout (0.0-1.0)")
    parser.add_argument("--min_orientations", type=int, default=1,
                       help="Minimum orientations to keep after dropout")
    
    # Output options
    parser.add_argument("--save_nifti", action="store_true", default=True, help="Save NIfTI files")
    parser.add_argument("--no_save_plots", action="store_true", help="Don't save visualization plots")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate inputs
    if not args.input and not args.input_dir:
        raise ValueError("Must specify either --input or --input_dir")
    
    if args.input and args.input_dir:
        raise ValueError("Cannot specify both --input and --input_dir")
    
    # Get input paths
    if args.input:
        input_paths = [args.input]
    else:
        input_dir = Path(args.input_dir)
        input_paths = sorted([
            str(p) for p in input_dir.glob("*.nii.gz")
        ] + [
            str(p) for p in input_dir.glob("*.nii")
        ])
        
        if len(input_paths) == 0:
            raise ValueError(f"No NIfTI files found in {input_dir}")
        
        input_paths = input_paths[:args.num_samples]
    
    print(f"Found {len(input_paths)} volumes to process")
    
    # Create data generator
    print("\nInitializing data generator...")
    generator = HRLRDataGenerator(
        atlas_res=args.atlas_res,
        target_res=[1.0, 1.0, 1.0],
        output_shape=list(args.output_shape),
        min_resolution=args.min_resolution,
        max_res_aniso=args.max_res_aniso,
        randomise_res=not args.no_randomise_res,
        prob_motion=args.prob_motion,
        prob_spike=args.prob_spike,
        prob_aliasing=args.prob_aliasing,
        prob_bias_field=args.prob_bias_field,
        prob_noise=args.prob_noise,
        apply_intensity_aug=not args.no_intensity_aug,
        clip_to_unit_range=True,
        orientation_dropout_prob=args.orientation_dropout_prob,
        min_orientations=args.min_orientations,
    )
    
    print(f"Generator configuration:")
    print(f"  Output shape: {args.output_shape}")
    print(f"  Resolution range: {args.min_resolution} - {args.max_res_aniso} mm")
    print(f"  Randomize resolution: {generator.randomise_res}")
    print(f"  orientation dropout: {args.orientation_dropout_prob:.2f}")
    
    # Process volumes
    all_metadata = []
    for i, volume_path in enumerate(input_paths):
        try:
            metadata = verify_single_volume(
                volume_path=volume_path,
                generator=generator,
                output_dir=output_dir,
                sample_idx=i,
                save_nifti=args.save_nifti,
                save_plots=not args.no_save_plots
            )
            all_metadata.append(metadata)
        except Exception as e:
            print(f"✗ Failed to process {volume_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create summary report
    if all_metadata:
        create_summary_report(all_metadata, output_dir)
        print(f"\n✓ Verification complete! Results saved to: {output_dir}")
    else:
        print(f"\n✗ No samples were successfully processed!")


if __name__ == "__main__":
    main()