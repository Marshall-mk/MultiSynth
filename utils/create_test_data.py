#!/usr/bin/env python3
"""
create_test_data.py

Generate LR stacks from test subset for model evaluation.
Creates orthogonal LR stacks (axial, coronal, sagittal) and HR ground truth
for each test subject using the same artifact simulation as training.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    Spacingd,
    ToTensord,
)
from tqdm import tqdm

from src.data import HRLRDataGenerator
from src.utils import load_image_paths_from_csv


def compute_affine_from_resolution(shape: tuple, resolution: List[float]) -> np.ndarray:
    """
    Compute NIfTI affine matrix from shape and voxel resolution.

    Args:
        shape: Volume shape (D, H, W)
        resolution: Voxel resolution [x, y, z]

    Returns:
        4x4 affine matrix
    """
    affine = np.eye(4)
    affine[0, 0] = resolution[0]
    affine[1, 1] = resolution[1]
    affine[2, 2] = resolution[2]
    # Center the volume
    affine[:3, 3] = -np.array(shape) * np.array(resolution) / 2
    return affine


def save_patient_data(
    lr_stacks: List[torch.Tensor],
    true_lr_stacks: List[torch.Tensor],
    hr_groundtruth: torch.Tensor,
    resolutions: List[torch.Tensor],
    thicknesses: List[torch.Tensor],
    patient_dir: Path,
    patient_id: str,
    save_mode: str,
    atlas_res: List[float],
    source_path: str,
):
    """
    Save LR stacks and HR ground truth with proper NIfTI affine.

    Args:
        lr_stacks: List of 3 LR stack tensors (1, C, D, H, W)
        hr_groundtruth: HR tensor (1, C, D, H, W)
        patient_dir: Output directory for this patient
        patient_id: Patient identifier
        save_mode: One of ['upsampled_only', 'true_lr_only', 'both']
        atlas_res: HR voxel resolution
        source_path: Path to source HR volume
    """
    stack_names = ['axial', 'coronal', 'sagittal']
    if save_mode == 'both':
        # Save each LR stack
        for stack_idx, (lr_tensor, true_lr_tensor, resolution, stack_name) in enumerate(zip(lr_stacks, true_lr_stacks, resolutions, stack_names)):
            # Extract volume: (1, C, D, H, W) -> (D, H, W)
            lr_volume = lr_tensor[0, 0].cpu().numpy()
            true_lr_volume = true_lr_tensor[0, 0].cpu().numpy()

            # Compute affine matrix
            lr_affine = compute_affine_from_resolution(lr_volume.shape, atlas_res)
            true_lr_affine = compute_affine_from_resolution(true_lr_volume.shape, resolution.squeeze())
            lr_filename = f"{stack_name}_upsampled.nii.gz"
            true_lr_filename = f"{stack_name}.nii.gz"

            # Save NIfTI upsampled
            nifti_img = nib.Nifti1Image(lr_volume, lr_affine)
            nib.save(nifti_img, patient_dir / lr_filename)
            # Save NIfTI true
            nifti_img_true = nib.Nifti1Image(true_lr_volume, true_lr_affine)
            nib.save(nifti_img_true, patient_dir / true_lr_filename)
        
        # Save HR ground truth
        hr_volume = hr_groundtruth[0, 0].cpu().numpy()
        hr_affine = compute_affine_from_resolution(hr_volume.shape, atlas_res)
        nib.save(nib.Nifti1Image(hr_volume, hr_affine), patient_dir / "HR_groundtruth.nii.gz")

        # Get saved shape from first LR stack (all should be same shape)
        first_lr_shape = list(lr_stacks[0][0, 0].cpu().numpy().shape)

        # Save metadata
        metadata = {
            'patient_id': patient_id,
            'source_path': str(source_path),
            'stack_names': stack_names,
            'save_mode': save_mode,
            'atlas_res': atlas_res,
            'resolutions': resolutions[0].squeeze().tolist(),
            'saved_shape': first_lr_shape,
            'timestamp': datetime.now().isoformat(),
        }
        with open(patient_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    elif save_mode == 'upsampled_only':
        # Save each LR stack
        for stack_idx, (lr_tensor, stack_name) in enumerate(zip(lr_stacks, stack_names)):
            # Extract volume: (1, C, D, H, W) -> (D, H, W)
            lr_volume = lr_tensor[0, 0].cpu().numpy()

            # Compute affine matrix
            affine = compute_affine_from_resolution(lr_volume.shape, atlas_res)
            filename = f"{stack_name}_upsampled.nii.gz"
            # Save NIfTI
            nifti_img = nib.Nifti1Image(lr_volume, affine)
            nib.save(nifti_img, patient_dir / filename)
        
        # Save HR ground truth
        hr_volume = hr_groundtruth[0, 0].cpu().numpy()
        hr_affine = compute_affine_from_resolution(hr_volume.shape, atlas_res)
        nib.save(nib.Nifti1Image(hr_volume, hr_affine), patient_dir / "HR_groundtruth.nii.gz")

        # Get saved shape from first LR stack (all should be same shape)
        first_lr_shape = list(lr_stacks[0][0, 0].cpu().numpy().shape)

        # Save metadata
        metadata = {
            'patient_id': patient_id,
            'source_path': str(source_path),
            'stack_names': stack_names,
            'save_mode': save_mode,
            'atlas_res': atlas_res,
            'saved_shape': first_lr_shape,
            'timestamp': datetime.now().isoformat(),
        }
        with open(patient_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    elif save_mode == 'true_lr_only':
        # Save each LR stack
        for stack_idx, (lr_tensor, resolution, stack_name) in enumerate(zip(true_lr_stacks, resolutions, stack_names)):
            # Extract volume: (1, C, D, H, W) -> (D, H, W)
            lr_volume = lr_tensor[0, 0].cpu().numpy()
            # Compute affine matrix
            affine = compute_affine_from_resolution(lr_volume.shape, resolution.squeeze())
            filename = f"{stack_name}.nii.gz"
            # Save NIfTI
            nifti_img = nib.Nifti1Image(lr_volume, affine)
            nib.save(nifti_img, patient_dir / filename)
        
        # Save HR ground truth
        hr_volume = hr_groundtruth[0, 0].cpu().numpy()
        hr_affine = compute_affine_from_resolution(hr_volume.shape, atlas_res)
        nib.save(nib.Nifti1Image(hr_volume, hr_affine), patient_dir / "HR_groundtruth.nii.gz")

        # Get saved shape from first LR stack (all should be same shape)
        first_lr_shape = list(true_lr_stacks[0][0, 0].cpu().numpy().shape)

        # Save metadata
        metadata = {
            'patient_id': patient_id,
            'source_path': str(source_path),
            'stack_names': stack_names,
            'save_mode': save_mode,
            'atlas_res': atlas_res,
            'resolutions': resolutions[0].squeeze().tolist(),
            'saved_shape': first_lr_shape,
            'timestamp': datetime.now().isoformat(),
        }
        with open(patient_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Generate test data (LR stacks) from HR volumes for model evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output arguments
    parser.add_argument("--csv_file", type=str, required=True,
                        help="CSV file with image metadata (must have split='test' rows)")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Base directory for relative paths in CSV")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for generated LR stacks")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Optional limit on number of samples to process (for debugging)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing patient directories")

    # LR saving mode
    parser.add_argument("--save_mode", type=str, default="true_lr_only",
                        choices=['upsampled_only', 'true_lr_only', 'both'],
                        help="Save mode: upsampled_only (matches training), true_lr_only (native LR), both")

    # Data preprocessing (must match training)
    parser.add_argument("--atlas_res", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="HR voxel resolution")
    parser.add_argument("--target_spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="Target spacing for resampling")
    parser.add_argument("--output_shape", type=int, nargs=3, default=None,
                        help="Output crop/pad shape. If None and --use_input_shape not set, "
                             "uses input volume shape.")

    # Output shape behavior
    parser.add_argument("--use_input_shape", action="store_true",
                        help="Use each input volume's shape instead of fixed output_shape. "
                             "Overrides --output_shape for upsampling target.")

    # Artifact simulation parameters (match training defaults)
    parser.add_argument("--min_resolution", type=float, nargs=3, default=[2.0, 2.0, 2.0],
                        help="Minimum resolution for LR generation")
    parser.add_argument("--max_res_aniso", type=float, nargs=3, default=[9.0, 9.0, 9.0],
                        help="Maximum anisotropic resolution")
    parser.add_argument("--no_randomise_res", action="store_true",
                        help="Disable resolution randomization")
    parser.add_argument("--prob_motion", type=float, default=0.5,
                        help="Probability of motion artifacts")
    parser.add_argument("--prob_spike", type=float, default=0.5,
                        help="Probability of k-space spikes")
    parser.add_argument("--prob_aliasing", type=float, default=0.02,
                        help="Probability of aliasing artifacts")
    parser.add_argument("--prob_bias_field", type=float, default=0.5,
                        help="Probability of bias field")
    parser.add_argument("--prob_noise", type=float, default=0.8,
                        help="Probability of noise")
    parser.add_argument("--no_intensity_aug", action="store_true",
                        help="Disable intensity augmentation")
    parser.add_argument("--upsample_mode", type=str, default="nearest",
                        choices=['nearest', 'trilinear', 'nearest-exact'],
                        help="Upsampling mode for LR stacks")

    # CSV filtering
    parser.add_argument("--acquisition_types", type=str, nargs="+", default=["3D"],
                        help="Acquisition types to include (e.g., 3D 2D)")
    parser.add_argument("--mri_classifications", type=str, nargs="+", default=None,
                        help="MRI classifications to include (e.g., T1 T2 FLAIR)")
    parser.add_argument("--no_filter_4d", action="store_true",
                        help="Don't filter out 4D images")

    # Other
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (default 1 for memory efficiency)")

    args = parser.parse_args()

    # Set random seeds for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed set to {args.seed}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("TEST DATA GENERATION - create_test_data.py")
    print("="*60)
    print(f"CSV file: {args.csv_file}")
    print(f"Base directory: {args.base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Save mode: {args.save_mode}")
    print()

    # Load test paths from CSV
    print("Loading test paths from CSV...")
    try:
        test_paths = load_image_paths_from_csv(
            csv_path=args.csv_file,
            base_dir=args.base_dir,
            split='test',
            acquisition_types=args.acquisition_types,
            mri_classifications=args.mri_classifications,
            filter_4d=not args.no_filter_4d,
        )
    except Exception as e:
        print(f"ERROR: Failed to load CSV: {e}")
        sys.exit(1)

    if len(test_paths) == 0:
        print("ERROR: No test images found in CSV. Check split='test' and filtering.")
        sys.exit(1)

    print(f"Found {len(test_paths)} test images")

    # Apply num_samples limit if specified
    if args.num_samples is not None:
        test_paths = test_paths[:args.num_samples]
        print(f"Processing first {len(test_paths)} samples (--num_samples limit)")

    print()

    # Create MONAI transform pipeline (deterministic for test data)
    print("Creating preprocessing pipeline...")
    transforms = [
        LoadImaged(keys=["image"], image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS", labels=None),
        Spacingd(keys=["image"], pixdim=args.target_spacing, mode="bilinear"),
        ToTensord(keys=["image"]),
    ]
    transform = Compose(transforms)

    # Initialize HRLRDataGenerator
    print("Initializing HRLRDataGenerator...")
    generator = HRLRDataGenerator(
        atlas_res=args.atlas_res,
        target_res=args.atlas_res,
        output_shape=args.output_shape,       
        preserve_input_shape=args.use_input_shape, 
        min_resolution=args.min_resolution,
        max_res_aniso=args.max_res_aniso,
        randomise_res=not args.no_randomise_res,
        prob_motion=args.prob_motion,
        prob_spike=args.prob_spike,
        prob_aliasing=args.prob_aliasing,
        prob_bias_field=args.prob_bias_field,
        prob_noise=args.prob_noise,
        apply_intensity_aug=not args.no_intensity_aug,
        upsample_mode=args.upsample_mode,
    )

    print("Configuration:")
    print(f"  Atlas resolution: {args.atlas_res}")
    if args.use_input_shape:
        print(f"  Output shape mode: Use input volume shape (adaptive)")
    elif args.output_shape:
        print(f"  Output shape mode: Fixed shape {args.output_shape}")
    else:
        print(f"  Output shape mode: Use input volume shape (default)")
    print(f"  Resolution range: {args.min_resolution} to {args.max_res_aniso}")
    print(f"  Artifact probabilities: motion={args.prob_motion}, spike={args.prob_spike}, "
          f"aliasing={args.prob_aliasing}, bias={args.prob_bias_field}, noise={args.prob_noise}")
    print(f"  Upsample mode: {args.upsample_mode}")
    print()

    # Process each test volume
    num_success = 0
    num_failed = 0
    num_skipped = 0
    start_time = time.time()
    processing_times = []

    print("Processing test volumes...")
    for img_path in tqdm(test_paths, desc="Generating LR stacks"):
        # Extract patient ID from filename
        patient_id = Path(img_path).stem.replace(".nii", "")
        patient_dir = output_dir / patient_id

        # Skip if exists and not overwrite
        if patient_dir.exists() and not args.overwrite:
            num_skipped += 1
            continue

        patient_dir.mkdir(parents=True, exist_ok=True)

        try:
            vol_start_time = time.time()

            # Load and preprocess HR volume
            data_dict = {"image": img_path}
            hr_volume = transform(data_dict)["image"]  # (C, D, H, W)

            # Add batch dimension
            hr_batch = hr_volume.unsqueeze(0).to(args.device)  # (1, C, D, H, W)

            # Generate LR stacks
            with torch.no_grad():
                lr_stacks, true_lr_stacks, hr_aug, resolutions, thicknesses, orientation_mask = generator.generate_paired_data(
                    hr_batch, return_resolution=True, return_intermediate=True
                )

            # Save outputs
            save_patient_data(
                lr_stacks=lr_stacks,
                true_lr_stacks = true_lr_stacks,
                hr_groundtruth=hr_aug,
                resolutions = resolutions,
                thicknesses = thicknesses,
                patient_dir=patient_dir,
                patient_id=patient_id,
                save_mode=args.save_mode,
                atlas_res=args.atlas_res,
                source_path=img_path,
            )

            vol_time = time.time() - vol_start_time
            processing_times.append(vol_time)
            num_success += 1

        except Exception as e:
            num_failed += 1
            print(f"\nERROR processing {patient_id}: {e}")
            # Log to error file
            with open(output_dir / "errors.log", 'a') as f:
                f.write(f"{patient_id}\t{img_path}\t{str(e)}\n")

    total_time = time.time() - start_time

    # Print summary
    print("\n" + "="*60)
    print("TEST DATA GENERATION COMPLETE")
    print("="*60)
    print(f"Total test images: {len(test_paths)}")
    print(f"Successfully processed: {num_success}")
    print(f"Skipped (already exist): {num_skipped}")
    print(f"Failed: {num_failed}")
    print(f"Output directory: {output_dir}")

    if processing_times:
        avg_time = np.mean(processing_times)
        print(f"Average processing time: {avg_time:.2f}s per volume")

    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")

    if num_failed > 0:
        print(f"\nSee {output_dir / 'errors.log'} for details on failures")

    print("\nExample output structure:")
    if num_success > 0:
        example_patient = output_dir / Path(test_paths[0]).stem
        if example_patient.exists():
            print(f"  {example_patient}/")
            for item in sorted(example_patient.iterdir()):
                print(f"    {item.name}")

    print("="*60)


if __name__ == "__main__":
    main()
