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
    LoadImage,
    EnsureChannelFirst,
    Orientation,
    Spacing,
)

from src import UHVED
from src.data import HRLRDataGenerator
from src.utils import (
    pad_to_multiple_of_32,
    unpad_volume,
    sliding_window_inference,
)


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
    num_modalities = model_config.get('num_modalities', 3)
    base_channels = model_config.get('base_channels', 32)
    num_scales = model_config.get('num_scales', 4)

    print(f"Model configuration:")
    print(f"  - Modalities: {num_modalities}")
    print(f"  - Base channels: {base_channels}")
    print(f"  - Scales: {num_scales}")

    # Create model
    model = UHVED(
        num_modalities=num_modalities,
        in_channels=1,
        out_channels=1,
        base_channels=base_channels,
        num_scales=num_scales,
        share_encoder=False,
        share_decoder=False,
        reconstruct_modalities=False,  # Disable for inference
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
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
        LoadImage(image_only=False),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        Spacing(pixdim=target_res, mode="bilinear"),
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
        target_res: Target resolution [x, y, z] in mm

    Returns:
        Tuple of (lr_stacks_tensors, affine)
    """
    print("  Loading pre-existing orthogonal LR stacks...")
    print("  ⚠️  Order must be: [Axial, Coronal, Sagittal]")

    transforms = create_inference_transforms(target_res)
    lr_stacks_tensors = []
    affine = None

    orientation_mapping = [
        ("Axial", "Stack 0", "High-res in D axis"),
        ("Coronal", "Stack 1", "High-res in H axis"),
        ("Sagittal", "Stack 2", "High-res in W axis"),
    ]

    for i, stack_path in enumerate(stack_paths):
        orientation, stack_num, description = orientation_mapping[i]
        print(f"    - {stack_num} ({orientation}): {stack_path}")
        print(f"      └─ {description}")

        # Load and preprocess
        data = transforms(stack_path)

        # Extract volume and metadata
        if isinstance(data, tuple):
            volume, meta = data[0], data[1] if len(data) > 1 else {}
        else:
            volume = data
            meta = {}

        # Convert to numpy if tensor
        if isinstance(volume, torch.Tensor):
            volume_np = volume.cpu().numpy()
        else:
            volume_np = np.array(volume)

        # Remove channel dimension for processing (C, D, H, W) -> (D, H, W)
        if volume_np.ndim == 4 and volume_np.shape[0] == 1:
            volume_np = volume_np[0]

        # Normalize to [0, 1]
        volume_np = (volume_np - volume_np.min()) / (volume_np.max() - volume_np.min() + 1e-8)

        # Add to list as tensor with channel dimension
        lr_stacks_tensors.append(torch.from_numpy(volume_np).float().unsqueeze(0))

        # Get affine from first stack
        if affine is None:
            if hasattr(meta, 'get') and 'affine' in meta:
                affine = meta['affine']
                if isinstance(affine, torch.Tensor):
                    affine = affine.cpu().numpy()
            else:
                # Try loading from original file
                try:
                    original_img = nib.load(stack_path)
                    affine = original_img.affine.copy()
                except:
                    affine = np.diag([target_res[0], target_res[1], target_res[2], 1.0])

    print(f"    Stack shapes: {[s.shape for s in lr_stacks_tensors]}")
    return lr_stacks_tensors, affine


def generate_orthogonal_stacks(hr_volume, generator):
    """
    Generate 3 orthogonal LR stacks from HR volume.

    Args:
        hr_volume: High-resolution volume (D, H, W)
        generator: HRLRDataGenerator instance

    Returns:
        List of 3 LR stacks (each as tensor)
    """
    # Generate orthogonal stacks using the data generator
    # The generator expects numpy array input
    if isinstance(hr_volume, torch.Tensor):
        hr_volume_np = hr_volume.cpu().numpy()
    else:
        hr_volume_np = np.array(hr_volume)

    # Remove channel dimension if present
    if hr_volume_np.ndim == 4 and hr_volume_np.shape[0] == 1:
        hr_volume_np = hr_volume_np[0]

    # Generate orthogonal stacks (simulates acquisition)
    # This uses the same degradation pipeline as training
    lr_stacks, _ = generator.generate_orthogonal_stacks(hr_volume_np)

    # Convert to tensors with channel dimension
    lr_stacks_tensors = [
        torch.from_numpy(stack).float().unsqueeze(0)  # Add channel dim
        for stack in lr_stacks
    ]

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
    """
    if input_stack_paths:
        print(f"\nProcessing pre-existing orthogonal stacks:")
        for i, path in enumerate(input_stack_paths):
            print(f"  {['Axial', 'Coronal', 'Sagittal'][i]}: {path}")
    else:
        print(f"\nProcessing: {input_path}")

    # Two modes: load pre-existing stacks OR generate from HR volume
    if input_stack_paths:
        # Mode 1: Load 3 pre-existing orthogonal LR stacks
        lr_stacks, affine = load_orthogonal_stacks_from_files(input_stack_paths, target_res)

    else:
        # Mode 2: Generate orthogonal stacks from single HR volume
        # Load and preprocess with MONAI
        print("  Loading and preprocessing...")
        transforms = create_inference_transforms(target_res)
        data = transforms(input_path)

        # Extract volume and metadata
        if isinstance(data, tuple):
            volume, meta = data[0], data[1] if len(data) > 1 else {}
        else:
            volume = data
            meta = {}

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
        affine = None
        if hasattr(meta, 'get') and 'affine' in meta:
            affine = meta['affine']
            if isinstance(affine, torch.Tensor):
                affine = affine.cpu().numpy()

        if affine is None:
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

    print(f"  Padded shape: {lr_stacks_padded[0].shape}")

    # Move to device
    lr_stacks_padded = [stack.to(device) for stack in lr_stacks_padded]

    # Run inference
    model.eval()
    with torch.no_grad():
        if use_sliding_window:
            print(f"  Running sliding window inference (patch: {patch_size}, overlap: {overlap})...")
            sr_output = sliding_window_inference(
                model=model,
                modalities=lr_stacks_padded,
                patch_size=patch_size,
                overlap=overlap,
                batch_size=1,
                device=device,
                blend_mode="gaussian",
                progress=True,
            )
        else:
            print("  Running standard inference...")
            outputs = model(lr_stacks_padded)
            sr_output = outputs['sr_output'] if isinstance(outputs, dict) else outputs

    # Convert back to numpy
    sr_output = sr_output.squeeze().cpu().numpy()  # (D, H, W)
    print(f"  Output shape before unpad: {sr_output.shape}")

    # Unpad to original shape
    sr_output = unpad_volume(sr_output, pad_before, orig_shape)
    print(f"  Output shape after unpad: {sr_output.shape}")

    # Denormalize (keep in 0-1 range, scale by original max)
    sr_output = np.clip(sr_output, 0, 1)

    # Save output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_nii = nib.Nifti1Image(sr_output, affine)
    nib.save(out_nii, output_path)
    print(f"  ✓ Saved to: {output_path}")
    print(f"  Output range: [{sr_output.min():.4f}, {sr_output.max():.4f}]")


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
    parser.add_argument("--input_stacks", type=str, nargs=3, default=None,
                       help="Three orthogonal LR stack files (axial coronal sagittal)")
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

    # Inference arguments
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device: cuda or cpu")
    parser.add_argument("--use_sliding_window", action="store_true",
                       help="Use sliding window inference (for large volumes)")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128],
                       help="Patch size for sliding window (e.g., 128 128 128)")
    parser.add_argument("--overlap", type=float, default=0.5,
                       help="Overlap ratio for sliding window (0.0-1.0)")

    args = parser.parse_args()

    # Validate input arguments
    if args.input_stacks and args.input:
        raise ValueError("Cannot specify both --input and --input_stacks. Choose one mode.")

    if not args.input_stacks and not args.input:
        raise ValueError("Must specify either --input (for HR volume) or --input_stacks (for 3 orthogonal LR stacks)")

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Mode 1: Pre-existing orthogonal stacks
    if args.input_stacks:
        # Validate that all 3 stack files exist
        for i, stack_path in enumerate(args.input_stacks):
            if not Path(stack_path).exists():
                raise ValueError(f"Stack {i+1} not found: {stack_path}")

        input_paths = None
        output_paths = args.output
        input_stack_paths = args.input_stacks

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
    )
