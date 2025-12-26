"""
U-HVED Inference Script for Orthogonal Stack Super-Resolution

This script performs super-resolution on MRI volumes using the U-HVED model
with orthogonal low-resolution stacks (axial, coronal, sagittal).

Two modes are supported:
1. Generate orthogonal LR stacks from a single HR volume (--input)
2. Use pre-existing orthogonal LR stacks (--input_stacks)

Usage:

    # Mode 1: Generate stacks from HR volume (single file)
    python test.py \
        --input /path/to/hr_volume.nii.gz \
        --output /path/to/output_sr.nii.gz \
        --model ./models/uhved_orthogonal_best.pth \
        --device cuda

    # Mode 1: Batch inference on directory
    python test.py \
        --input /path/to/hr_volumes_dir \
        --output /path/to/output_dir \
        --model ./models/uhved_orthogonal_best.pth \
        --device cuda \
        --use_sliding_window \
        --patch_size 128 128 128

    # Mode 2: Use pre-existing 3 orthogonal LR stacks
    python test.py \
        --input_stacks lr_axial.nii.gz lr_coronal.nii.gz lr_sagittal.nii.gz \
        --output /path/to/output_sr.nii.gz \
        --model ./models/uhved_orthogonal_best.pth \
        --device cuda

    # Mode 2: With sliding window (for large volumes)
    python test.py \
        --input_stacks lr_axial.nii.gz lr_coronal.nii.gz lr_sagittal.nii.gz \
        --output /path/to/output_sr.nii.gz \
        --model ./models/uhved_orthogonal_best.pth \
        --use_sliding_window \
        --patch_size 96 96 96 \
        --overlap 0.75 \
        --device cuda

License: Apache 2.0
"""

import os
import argparse
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
)

from src import UHVED
from src.data import HRLRDataGenerator
from src.utils import (
    pad_to_multiple_of_32,
    unpad_volume,
    sliding_window_inference,
)


def get_resolution_from_affine(affine: np.ndarray) -> np.ndarray:
    """
    Extract voxel resolution [x,y,z] in mm from NIfTI affine matrix.

    Args:
        affine: 4x4 affine transformation matrix

    Returns:
        Array of [res_x, res_y, res_z] in mm
    """
    res_x = np.linalg.norm(affine[:3, 0])
    res_y = np.linalg.norm(affine[:3, 1])
    res_z = np.linalg.norm(affine[:3, 2])
    return np.array([res_x, res_y, res_z])


def is_anisotropic(resolution: np.ndarray, threshold: float = 0.1) -> bool:
    """
    Check if resolution is anisotropic (non-cubic voxels).

    Args:
        resolution: Array of [res_x, res_y, res_z] in mm
        threshold: Maximum allowed difference (mm) to consider isotropic

    Returns:
        True if anisotropic, False if isotropic
    """
    res_range = resolution.max() - resolution.min()
    return res_range > threshold


def create_isotropic_affine(target_res: list, shape: tuple, original_affine: np.ndarray) -> np.ndarray:
    """
    Create affine matrix for isotropic space, preserving orientation from original.

    Args:
        target_res: Isotropic resolution [x,y,z] in mm
        shape: Shape of isotropic volume (D,H,W)
        original_affine: Original affine to preserve orientation

    Returns:
        4x4 affine matrix for isotropic space
    """
    # Extract rotation/orientation from original affine (normalized)
    rotation = original_affine[:3, :3]
    u = rotation[:, 0] / np.linalg.norm(rotation[:, 0])
    v = rotation[:, 1] / np.linalg.norm(rotation[:, 1])
    w = rotation[:, 2] / np.linalg.norm(rotation[:, 2])

    # Create new affine with isotropic scaling
    new_affine = np.eye(4)
    new_affine[:3, 0] = u * target_res[0]
    new_affine[:3, 1] = v * target_res[1]
    new_affine[:3, 2] = w * target_res[2]
    new_affine[:3, 3] = original_affine[:3, 3]  # Preserve translation

    return new_affine


def load_uhved_from_checkpoint(checkpoint_path, device="cuda"):
    """
    Load U-HVED model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        device: Device to load model on

    Returns:
        Tuple of (model, checkpoint_data)
    """
    print(f"Loading checkpoint: {checkpoint_path}")
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

    print(f"Model configuration:")
    print(f"  - num_orientations: {num_orientations}")
    print(f"  - base_channels: {base_channels}")
    print(f"  - num_scales: {num_scales}")
    print(f"  - reconstruct_orientations: {reconstruct_orientations}")
    print(f"  - use_prior: {use_prior}")
    print(f"  - use_encoder_outputs_as_skip: {use_encoder_outputs_as_skip}")
    print(f"  - decoder_upsample_mode: {decoder_upsample_mode}")
    print(f"  - final_activation: {final_activation}")

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
    # Old checkpoints use 'modality_decoders', new code uses 'orientation_decoders'
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        # Replace modality_decoders with orientation_decoders
        new_key = key.replace('modality_decoders', 'orientation_decoders')
        new_state_dict[new_key] = value

    # Load weights
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    return model, checkpoint


def create_inference_transforms(target_res=[1.0, 1.0, 1.0]):
    """
    Create MONAI preprocessing transforms for inference.

    Args:
        target_res: Target resolution [x, y, z] in mm

    Returns:
        MONAI Compose transform
    """
    return Compose([
        LoadImaged(keys=["image"], image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS", labels=None),
        Spacingd(keys=["image"], pixdim=target_res, mode="bilinear"),
    ])


def load_orthogonal_stacks_from_files(stack_paths, target_res=[1.0, 1.0, 1.0]):
    """
    Load 3 pre-existing orthogonal LR stacks from files.

    IMPORTANT: Stack order must match training configuration:
        - Stack 0 (Axial): High-res in D/depth axis, low-res in H,W
        - Stack 1 (Coronal): High-res in H/height axis, low-res in D,W
        - Stack 2 (Sagittal): High-res in W/width axis, low-res in D,H

    Args:
        stack_paths: List of 3 file paths [axial, coronal, sagittal] - ORDER MATTERS!
                    Can contain None for missing orientations.
        target_res: Target resolution [x, y, z] in mm

    Returns:
        Tuple of (lr_stacks_tensors, metadata_dict)

        metadata_dict contains:
        - 'affine_original': Original affine from first stack
        - 'affine_isotropic': Affine for isotropic resampled space
        - 'resolution_original': Original resolution [x,y,z] in mm
        - 'shape_isotropic': Shape after resampling (D,H,W)
        - 'is_anisotropic': Boolean flag
    """
    print("  Loading pre-existing orthogonal LR stacks...")
    print("  ⚠️  Order: [Axial, Coronal, Sagittal]")

    # Initialize metadata dictionary
    metadata = {
        'affine_original': None,
        'affine_isotropic': None,
        'resolution_original': None,
        'shape_isotropic': None,
        'is_anisotropic': False,
    }

    # BEFORE transforms: Load first valid stack to get original metadata
    for stack_path in stack_paths:
        if stack_path is not None:
            try:
                original_img = nib.load(stack_path)
                metadata['affine_original'] = original_img.affine.copy()
                metadata['resolution_original'] = get_resolution_from_affine(original_img.affine)
                metadata['is_anisotropic'] = is_anisotropic(metadata['resolution_original'])

                if metadata['is_anisotropic']:
                    print(f"  ⚠️  Detected anisotropic resolution: {metadata['resolution_original']} mm")
                    print(f"      Resampling to isotropic: {target_res} mm")
                else:
                    print(f"  ℹ️  Input resolution: {metadata['resolution_original']} mm (already isotropic)")
            except Exception as e:
                print(f"  Warning: Could not load metadata from {stack_path}: {e}")
                # Fallback to default
                metadata['affine_original'] = np.diag([target_res[0], target_res[1], target_res[2], 1.0])
                metadata['resolution_original'] = np.array(target_res)
            break

    # If no valid stacks found, use default
    if metadata['affine_original'] is None:
        metadata['affine_original'] = np.diag([target_res[0], target_res[1], target_res[2], 1.0])
        metadata['resolution_original'] = np.array(target_res)

    transforms = create_inference_transforms(target_res)
    lr_stacks_tensors = [None, None, None]  # Initialize with placeholders
    reference_shape = None

    orientation_mapping = [
        ("Axial", "Stack 0", "High-res in D axis"),
        ("Coronal", "Stack 1", "High-res in H axis"),
        ("Sagittal", "Stack 2", "High-res in W axis"),
    ]

    for i, stack_path in enumerate(stack_paths):
        orientation, stack_num, description = orientation_mapping[i]

        if stack_path is None:
            # Missing orientation - create dummy stack later
            print(f"    - {stack_num} ({orientation}): [MISSING - will use dummy stack]")
            # Keep as None, will create dummy after we know the shape
            continue

        print(f"    - {stack_num} ({orientation}): {stack_path}")
        print(f"      └─ {description}")

        # Load and preprocess with dictionary-based transforms
        data_dict = {"image": stack_path}
        data = transforms(data_dict)

        # Extract volume (dictionary-based transforms return dict)
        volume = data["image"]

        # Convert to numpy if tensor
        if isinstance(volume, torch.Tensor):
            volume_np = volume.cpu().numpy()
        else:
            volume_np = np.array(volume)

        # Remove channel dimension for processing (C, D, H, W) -> (D, H, W)
        if volume_np.ndim == 4 and volume_np.shape[0] == 1:
            volume_np = volume_np[0]

        # Store reference shape for creating dummy stacks
        if reference_shape is None:
            reference_shape = volume_np.shape

        # Normalize to [0, 1]
        volume_np = (volume_np - volume_np.min()) / (volume_np.max() - volume_np.min() + 1e-8)

        # Add to list as tensor with channel dimension
        lr_stacks_tensors[i] = torch.from_numpy(volume_np).float().unsqueeze(0)

    # Create dummy stacks for missing orientations
    if reference_shape is None:
        raise ValueError("No valid stacks provided - at least one stack is required!")

    for i, stack in enumerate(lr_stacks_tensors):
        if stack is None:
            # Create zero-filled dummy stack with same shape as reference
            dummy = torch.zeros((1,) + reference_shape, dtype=torch.float32)
            lr_stacks_tensors[i] = dummy
            print(f"    Created dummy stack for {orientation_mapping[i][0]}: shape {dummy.shape}")

    # AFTER resampling: Create isotropic affine and finalize metadata
    metadata['shape_isotropic'] = lr_stacks_tensors[0].squeeze().shape  # (D,H,W)
    metadata['affine_isotropic'] = create_isotropic_affine(
        target_res,
        metadata['shape_isotropic'],
        metadata['affine_original']
    )

    print(f"    Final stack shapes: {[s.shape for s in lr_stacks_tensors]}")
    return lr_stacks_tensors, metadata


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
        # (D, H, W) -> (1, 1, D, H, W)
        hr_volume = hr_volume.unsqueeze(0).unsqueeze(0)
    elif hr_volume.ndim == 4:
        # (C, D, H, W) -> (1, C, D, H, W)
        hr_volume = hr_volume.unsqueeze(0)

    # Generate orthogonal stacks (simulates acquisition)
    # This uses the same degradation pipeline as training
    lr_stacks_list, _, orientation_mask = generator.generate_paired_data(hr_volume, return_resolution=False)

    # lr_stacks_list is a list of 3 tensors, each with shape (B, C, D, H, W)
    # Remove batch dimension and return as list
    lr_stacks_tensors = [stack.squeeze(0) for stack in lr_stacks_list]

    return lr_stacks_tensors


def predict_single_volume(
    model,
    input_path,
    output_path,
    generator,
    target_res=[1.0, 1.0, 1.0],
    device="cuda",
    use_sliding_window=False,
    patch_size=(128, 128, 128),
    overlap=0.5,
    input_stack_paths=None,
    orientation_mask=None,
    save_reconstructions=False,
    reconstruction_dir=None,
):
    """
    Run U-HVED inference on a single volume.

    Args:
        model: Trained U-HVED model
        input_path: Path to input NIfTI file (if generating stacks from HR)
        output_path: Path to save output
        generator: HRLRDataGenerator for creating orthogonal stacks
        target_res: Target resolution in mm [x, y, z]
        device: 'cuda' or 'cpu'
        use_sliding_window: Use sliding window inference
        patch_size: Patch size for sliding window (D, H, W)
        overlap: Overlap ratio for sliding window
        input_stack_paths: Optional list of 3 pre-existing stack paths [axial, coronal, sagittal]
        orientation_mask: Optional binary mask [1/0, 1/0, 1/0] indicating which orientations are present
        save_reconstructions: Whether to save reconstructed orientations
        reconstruction_dir: Directory to save reconstructed orientations
    """
    if input_stack_paths:
        print(f"\nProcessing pre-existing orthogonal stacks:")
        for i, path in enumerate(input_stack_paths):
            print(f"  {['Axial', 'Coronal', 'Sagittal'][i]}: {path}")
    else:
        print(f"\nProcessing: {input_path}")

    # Two modes: load pre-existing stacks OR generate from HR volume
    if input_stack_paths:
        # Mode 1: Load 3 pre-existing orthogonal LR stacks WITH METADATA
        lr_stacks, metadata = load_orthogonal_stacks_from_files(input_stack_paths, target_res)
        # Use isotropic affine for output (matches resampled data)
        affine = metadata['affine_isotropic']

    else:
        # Mode 2: Generate orthogonal stacks from single HR volume
        # Load and preprocess with MONAI
        print("  Loading and preprocessing...")
        transforms = create_inference_transforms(target_res)
        data_dict = {"image": input_path}
        data = transforms(data_dict)

        # Extract volume (dictionary-based transforms return dict)
        volume = data["image"]

        # Convert to numpy if tensor
        if isinstance(volume, torch.Tensor):
            volume_np = volume.cpu().numpy()
        else:
            volume_np = np.array(volume)

        # Remove channel dimension for processing (C, D, H, W) -> (D, H, W)
        if volume_np.ndim == 4 and volume_np.shape[0] == 1:
            volume_np = volume_np[0]

        print(f"  Input shape: {volume_np.shape}")

        # Get affine for saving
        # Load affine from original file
        try:
            original_img = nib.load(input_path)
            affine = original_img.affine.copy()
        except:
            # Fallback to identity
            affine = np.diag([target_res[0], target_res[1], target_res[2], 1.0])

        # Normalize to [0, 1]
        volume_np = (volume_np - volume_np.min()) / (volume_np.max() - volume_np.min() + 1e-8)

        # Generate orthogonal LR stacks
        print("  Generating orthogonal LR stacks...")
        lr_stacks = generate_orthogonal_stacks(volume_np, generator)

    # Pad to multiple of 32 if needed
    original_shape = lr_stacks[0].squeeze().shape
    lr_stacks_padded = []

    for stack in lr_stacks:
        stack_np = stack.squeeze().cpu().numpy()
        padded, pad_before, orig_shape = pad_to_multiple_of_32(stack_np)
        lr_stacks_padded.append(
            torch.from_numpy(padded).float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        )

    # Move to device
    lr_stacks_padded = [stack.to(device) for stack in lr_stacks_padded]

    # Create orientation mask tensor
    if orientation_mask is not None:
        # Convert to boolean tensor
        orientation_mask_tensor = torch.tensor(orientation_mask, dtype=torch.bool, device=device).unsqueeze(0)  # (1, 3)
        present_orientations = [i for i, m in enumerate(orientation_mask) if m == 1]
        print(f"  Using orientation mask: {orientation_mask}")
        print(f"  Present orientations: {[['Axial', 'Coronal', 'Sagittal'][i] for i in present_orientations]}")
    else:
        # All orientations present
        orientation_mask_tensor = None
        print(f"  Using all 3 orientations")

    # Run inference
    model.eval()
    with torch.no_grad():
        if use_sliding_window:
            print(f"  Running sliding window inference (patch: {patch_size}, overlap: {overlap})...")
            # Note: sliding_window_inference doesn't support orientation_mask yet
            # TODO: Add orientation_mask support to sliding_window_inference
            sr_output = sliding_window_inference(
                model=model,
                orientations=lr_stacks_padded,
                patch_size=patch_size,
                overlap=overlap,
                batch_size=1,
                device=device,
                blend_mode="gaussian",
                progress=True,
            )
            orientation_outputs = []  # Not available with sliding window
        else:
            print("  Running standard inference...")
            outputs = model(lr_stacks_padded, orientation_mask=orientation_mask_tensor)

            # Extract outputs
            if isinstance(outputs, dict):
                sr_output = outputs['sr_output']
                orientation_outputs = outputs.get('orientation_outputs', [])
            else:
                sr_output = outputs
                orientation_outputs = []

    # Convert SR output back to numpy
    sr_output = sr_output.squeeze().cpu().numpy()  # (D, H, W)
    print(f"  SR output shape before unpad: {sr_output.shape}")

    # Unpad to original shape
    sr_output = unpad_volume(sr_output, pad_before, orig_shape)
    print(f"  SR output shape after unpad: {sr_output.shape}")

    # Denormalize (keep in 0-1 range, scale by original max)
    sr_output = np.clip(sr_output, 0, 1)

    # Save SR output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_nii = nib.Nifti1Image(sr_output, affine)
    nib.save(out_nii, output_path)

    # Log output information
    output_res = get_resolution_from_affine(affine)
    print(f"  ✓ SR output saved to: {output_path}")
    print(f"    Output shape: {sr_output.shape}")
    print(f"    Output resolution: [{output_res[0]:.2f}, {output_res[1]:.2f}, {output_res[2]:.2f}] mm")
    print(f"    Output range: [{sr_output.min():.4f}, {sr_output.max():.4f}]")

    # Save reconstructed orientations if requested
    if save_reconstructions and len(orientation_outputs) > 0:
        print(f"\n  Saving reconstructed orientations...")

        # Determine output directory
        if reconstruction_dir is None:
            reconstruction_dir = os.path.dirname(output_path) or "."
        os.makedirs(reconstruction_dir, exist_ok=True)

        # Get base filename
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        if base_name.endswith('.nii'):
            base_name = base_name[:-4]  # Remove .nii from .nii.gz

        orientation_names = ['axial', 'coronal', 'sagittal']
        for i, recon in enumerate(orientation_outputs):
            # Convert to numpy and unpad
            recon_np = recon.squeeze().cpu().numpy()
            recon_np = unpad_volume(recon_np, pad_before, orig_shape)
            recon_np = np.clip(recon_np, 0, 1)

            # Save
            recon_path = os.path.join(reconstruction_dir, f"{base_name}_recon_{orientation_names[i]}.nii.gz")
            recon_nii = nib.Nifti1Image(recon_np, affine)
            nib.save(recon_nii, recon_path)
            print(f"    - {orientation_names[i]}: {recon_path}")
    elif save_reconstructions and len(orientation_outputs) == 0:
        print(f"  Note: Model was not trained with orientation reconstruction, skipping...")



def predict_batch(
    input_paths,
    output_paths,
    model_path,
    target_res=[1.0, 1.0, 1.0],
    output_shape=(128, 128, 128),
    device="cuda",
    use_sliding_window=False,
    patch_size=(128, 128, 128),
    overlap=0.5,
    input_stack_paths=None,
    orientation_mask=None,
    save_reconstructions=False,
    reconstruction_dir=None,
):
    """Process multiple volumes in batch."""
    print("=" * 80)
    print("U-HVED Inference - Orthogonal Stack Super-Resolution")
    print("=" * 80)

    # Load model
    model, checkpoint = load_uhved_from_checkpoint(model_path, device=device)

    # Create data generator for orthogonal stacks (only needed if generating stacks)
    generator = None
    if not input_stack_paths:
        print("\nInitializing data generator for orthogonal stacks...")
        generator = HRLRDataGenerator(
            atlas_res=target_res,
            target_res=target_res,
            output_shape=list(output_shape),
            randomise_res=False,  # Fixed resolution for inference
            prob_motion=0.0,      # No augmentation for inference
            prob_spike=0.0,
            prob_aliasing=0.0,
            prob_bias_field=0.0,
            prob_noise=0.0,
            apply_intensity_aug=False,
            clip_to_unit_range=True,
            upsample_mode=upsample_mode,
        )

    print(f"\nInference settings:")
    print(f"  Mode: {'Pre-existing stacks' if input_stack_paths else 'Generate from HR volume'}")
    print(f"  Device: {device}")
    print(f"  Target resolution: {target_res} mm")
    if not input_stack_paths:
        print(f"  Output shape: {output_shape}")
    print(f"  Sliding window: {use_sliding_window}")
    if use_sliding_window:
        print(f"  Patch size: {patch_size}")
        print(f"  Overlap: {overlap}")

    if input_stack_paths:
        print(f"\nProcessing 1 set of stacks...\n")
    else:
        print(f"\nProcessing {len(input_paths)} volumes...\n")

    # Process volume(s)
    if input_stack_paths:
        # Single case: process one set of 3 stacks
        print(f"[1/1]")
        try:
            predict_single_volume(
                model=model,
                input_path=None,
                output_path=output_paths[0] if isinstance(output_paths, list) else output_paths,
                generator=generator,
                target_res=target_res,
                device=device,
                use_sliding_window=use_sliding_window,
                patch_size=patch_size,
                overlap=overlap,
                input_stack_paths=input_stack_paths,
                orientation_mask=orientation_mask,
                save_reconstructions=save_reconstructions,
                reconstruction_dir=reconstruction_dir,
            )
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        # Multiple volumes: generate stacks from each
        for idx, (input_path, output_path) in enumerate(zip(input_paths, output_paths)):
            print(f"[{idx + 1}/{len(input_paths)}]")
            try:
                predict_single_volume(
                    model=model,
                    input_path=input_path,
                    output_path=output_path,
                    generator=generator,
                    target_res=target_res,
                    device=device,
                    use_sliding_window=use_sliding_window,
                    patch_size=patch_size,
                    overlap=overlap,
                    input_stack_paths=None,
                    orientation_mask=orientation_mask,
                    save_reconstructions=save_reconstructions,
                    reconstruction_dir=reconstruction_dir,
                )
            except Exception as e:
                print(f"  ✗ ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    print("\n" + "=" * 80)
    print("Inference complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-HVED Inference with Orthogonal Stacks")

    # Input/output arguments
    parser.add_argument("--input", type=str, required=False,
                       help="Input image file or directory (single HR volume)")
    parser.add_argument("--input_stacks", type=str, nargs='+', default=None,
                       help="Orthogonal LR stack files (1-3 stacks). Provide in order: axial, coronal, sagittal. "
                            "If fewer than 3 stacks, you MUST also specify --orientation_mask to indicate which orientations are present. "
                            "Example: For axial+coronal only, use '--input_stacks axial.nii.gz coronal.nii.gz --orientation_mask 1 1 0'")
    parser.add_argument("--output", type=str, required=True,
                       help="Output image file or directory")

    # Model arguments
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model checkpoint (.pth file)")

    # Preprocessing arguments
    parser.add_argument("--target_res", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                       help="Target resolution in mm (e.g., 1.0 1.0 1.0)")
    parser.add_argument("--output_shape", type=int, nargs=3, default=[128, 128, 128],
                       help="Output volume shape (e.g., 128 128 128)")
    parser.add_argument("--upsample_mode", type=str, default="nearest",
                       choices=["nearest", "trilinear", "nearest-exact"],
                       help="Interpolation mode for FFT upsample recovery (default: nearest)")

    # Inference arguments
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device: cuda or cpu")
    parser.add_argument("--use_sliding_window", action="store_true",
                       help="Use sliding window inference (for large volumes)")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128],
                       help="Patch size for sliding window (e.g., 128 128 128)")
    parser.add_argument("--overlap", type=float, default=0.5,
                       help="Overlap ratio for sliding window (0.0-1.0)")

    # Orientation handling arguments
    parser.add_argument("--orientation_mask", type=int, nargs=3, default=None,
                       help="Binary mask indicating which orientations are present (e.g., 1 1 0 for axial+coronal only). "
                            "Use this to handle missing modalities. Default: all present (1 1 1)")
    parser.add_argument("--save_reconstructions", action="store_true",
                       help="Save reconstructed orientation outputs (if model was trained with orientation reconstruction)")
    parser.add_argument("--reconstruction_dir", type=str, default=None,
                       help="Directory to save reconstructed orientations (default: same as output directory)")

    args = parser.parse_args()

    # Validate input arguments
    if args.input_stacks and args.input:
        raise ValueError("Cannot specify both --input and --input_stacks. Choose one mode.")

    if not args.input_stacks and not args.input:
        raise ValueError("Must specify either --input (for HR volume) or --input_stacks (for orthogonal LR stacks)")

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Mode 1: Pre-existing orthogonal stacks
    if args.input_stacks:
        # Validate number of stacks
        num_stacks = len(args.input_stacks)
        if num_stacks < 1 or num_stacks > 3:
            raise ValueError(f"Expected 1-3 input stacks, got {num_stacks}")

        # Validate that all stack files exist
        for i, stack_path in enumerate(args.input_stacks):
            if not Path(stack_path).exists():
                raise ValueError(f"Stack {i+1} not found: {stack_path}")

        # Handle partial stacks (fewer than 3)
        if num_stacks < 3:
            if args.orientation_mask is None:
                raise ValueError(
                    f"When providing fewer than 3 stacks ({num_stacks} provided), "
                    "you MUST specify --orientation_mask to indicate which orientations are present.\n"
                    "Example: For axial+coronal, use: --orientation_mask 1 1 0"
                )

            # Verify orientation_mask matches number of stacks
            num_present = sum(args.orientation_mask)
            if num_present != num_stacks:
                raise ValueError(
                    f"Orientation mask indicates {num_present} present orientations, "
                    f"but {num_stacks} stacks were provided. These must match!"
                )

            print(f"\n⚠️  Processing with {num_stacks}/3 orientations:")
            present_names = [name for name, present in zip(['Axial', 'Coronal', 'Sagittal'], args.orientation_mask) if present]
            print(f"   Present: {', '.join(present_names)}")

            # Create full list with None placeholders for missing stacks
            input_stack_paths = [None, None, None]
            stack_idx = 0
            for i, present in enumerate(args.orientation_mask):
                if present:
                    input_stack_paths[i] = args.input_stacks[stack_idx]
                    stack_idx += 1
        else:
            # All 3 stacks provided
            input_stack_paths = args.input_stacks
            # If orientation_mask not specified, default to all present
            if args.orientation_mask is None:
                args.orientation_mask = [1, 1, 1]

        input_paths = None
        output_paths = args.output

    # Mode 2: Generate stacks from HR volume(s)
    else:
        input_path = Path(args.input)
        output_path = Path(args.output)
        input_stack_paths = None

        if input_path.is_file():
            # Single file
            input_paths = [str(input_path)]
            output_paths = [str(output_path)]
        elif input_path.is_dir():
            # Directory
            input_paths = sorted(
                [str(p) for p in input_path.glob("*.nii.gz")]
                + [str(p) for p in input_path.glob("*.nii")]
            )

            if len(input_paths) == 0:
                raise ValueError(f"No .nii or .nii.gz files found in {input_path}")

            # Create output directory
            output_path.mkdir(parents=True, exist_ok=True)

            # Generate output paths
            output_paths = [
                str(output_path / (Path(ip).stem.replace(".nii", "") + "_sr.nii.gz"))
                for ip in input_paths
            ]
        else:
            raise ValueError(f"Input path does not exist: {input_path}")

    # Run inference
    predict_batch(
        input_paths=input_paths,
        output_paths=output_paths,
        model_path=args.model,
        target_res=args.target_res,
        output_shape=tuple(args.output_shape),
        device=args.device,
        use_sliding_window=args.use_sliding_window,
        patch_size=tuple(args.patch_size),
        overlap=args.overlap,
        input_stack_paths=input_stack_paths,
        orientation_mask=args.orientation_mask,
        save_reconstructions=args.save_reconstructions,
        reconstruction_dir=args.reconstruction_dir,
    )