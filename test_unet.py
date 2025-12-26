#!/usr/bin/env python3
"""
Test script for U-Net Super-Resolution Model

This script performs inference on single volumes or batches using a trained U-Net model.
It can either generate LR stacks from HR volumes or load pre-existing LR stacks.

Usage examples:
    # Single volume with stack generation
    python test_unet.py --input /path/to/hr_volume.nii.gz \
                        --output /path/to/output_sr.nii.gz \
                        --model checkpoints_unet/unet_best.pth

    # Single volume with pre-existing stacks
    python test_unet.py --input_stacks axial.nii.gz coronal.nii.gz sagittal.nii.gz \
                        --output output_sr.nii.gz \
                        --model checkpoints_unet/unet_best.pth

    # Batch processing with sliding window
    python test_unet.py --input /path/to/hr_volumes/ \
                        --output /path/to/sr_outputs/ \
                        --model checkpoints_unet/unet_best.pth \
                        --use_sliding_window \
                        --patch_size 96 96 96 \
                        --overlap 0.5
"""

import os
import argparse
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd
from monai.inferers import sliding_window_inference

from src.unet import CustomUNet3D
from src.data import HRLRDataGenerator
from src.utils import pad_to_multiple_of_32, unpad_volume


def load_unet_from_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """
    Load U-Net model from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        model: Loaded U-Net model in eval mode
        checkpoint: Full checkpoint dict for metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model configuration
    model_config = checkpoint.get('model_config', {})
    num_input_channels = model_config.get('num_input_channels', 3)
    nb_features = model_config.get('nb_features', 24)
    nb_levels = model_config.get('nb_levels', 5)
    output_shape = model_config.get('output_shape', [128, 128, 128])
    final_activation = model_config.get('final_activation', 'sigmoid')

    # Instantiate model with saved config
    model = CustomUNet3D(
        nb_features=nb_features,
        input_shape=(num_input_channels, *output_shape),
        nb_levels=nb_levels,
        nb_labels=1,
        final_pred_activation=final_activation,
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded U-Net model from {checkpoint_path}:")
    print(f"  Input channels: {num_input_channels}")
    print(f"  Base features: {nb_features}")
    print(f"  Levels: {nb_levels}")
    print(f"  Output shape: {output_shape}")
    print(f"  Final activation: {final_activation}")
    if 'epoch' in checkpoint:
        print(f"  Checkpoint epoch: {checkpoint['epoch']}")

    return model, checkpoint


def concatenate_stacks(lr_stacks: List[torch.Tensor]) -> torch.Tensor:
    """
    Concatenate LR stacks channel-wise for U-Net input.

    Args:
        lr_stacks: List of 2-3 tensors, each (B, 1, D, H, W)

    Returns:
        Concatenated tensor (B, N, D, H, W) where N=2 or 3
    """
    return torch.cat(lr_stacks, dim=1)


def create_inference_transforms(target_res: List[float] = [1.0, 1.0, 1.0]):
    """
    Create MONAI transforms for loading and preprocessing volumes.

    Args:
        target_res: Target resolution in mm [x, y, z]

    Returns:
        MONAI Compose transform chain
    """
    return Compose([
        LoadImaged(keys=["image"], image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS", labels=None),
        Spacingd(keys=["image"], pixdim=target_res, mode="bilinear"),
    ])


def load_and_generate_stacks(
    input_path: str,
    generator: HRLRDataGenerator,
    transforms,
    device: str = "cuda"
) -> Tuple[List[torch.Tensor], np.ndarray, np.ndarray, np.ndarray]:
    """
    Load HR volume and generate LR stacks.

    Args:
        input_path: Path to HR volume NIfTI file
        generator: HRLRDataGenerator instance for creating LR stacks
        transforms: MONAI transform chain
        device: Device to put tensors on

    Returns:
        lr_stacks: List of 3 LR stacks (padded, on device)
        affine: Affine matrix for saving outputs
        original_shape: Shape before padding
        pad_before: Padding amounts applied
    """
    # Load and preprocess
    data_dict = {"image": input_path}
    transformed = transforms(data_dict)
    hr_volume = transformed["image"]
    affine = transformed["image_meta_dict"]["affine"]

    # Ensure correct dimensions
    if hr_volume.ndim == 3:
        hr_volume = hr_volume.unsqueeze(0)  # Add channel
    hr_volume = hr_volume.unsqueeze(0)  # Add batch

    # Normalize to [0, 1]
    hr_min = hr_volume.min()
    hr_max = hr_volume.max()
    hr_volume = (hr_volume - hr_min) / (hr_max - hr_min + 1e-8)

    # Generate LR stacks
    lr_stacks_list, _, _, _, _ = generator.generate_paired_data(
        hr_volume, return_resolution=True
    )

    # Convert to numpy and pad
    lr_stacks_padded = []
    original_shape = None
    pad_before = None

    for stack in lr_stacks_list:
        stack_np = stack.squeeze(0).squeeze(0).cpu().numpy()

        if original_shape is None:
            original_shape = np.array(stack_np.shape)

        # Pad to multiple of 32
        padded, pb, _ = pad_to_multiple_of_32(stack_np)
        if pad_before is None:
            pad_before = pb

        # Convert back to tensor
        lr_stacks_padded.append(
            torch.from_numpy(padded).float().unsqueeze(0).unsqueeze(0).to(device)
        )

    return lr_stacks_padded, affine, original_shape, pad_before


def load_existing_stacks(
    stack_paths: List[str],
    transforms,
    device: str = "cuda"
) -> Tuple[List[torch.Tensor], np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pre-existing LR stack files.

    Args:
        stack_paths: List of 2-3 paths to LR stack NIfTI files
        transforms: MONAI transform chain
        device: Device to put tensors on

    Returns:
        lr_stacks: List of LR stacks (padded, on device)
        affine: Affine matrix from first stack
        original_shape: Shape before padding
        pad_before: Padding amounts applied
    """
    lr_stacks_padded = []
    affine = None
    original_shape = None
    pad_before = None

    for stack_path in stack_paths:
        # Load stack
        data_dict = {"image": stack_path}
        transformed = transforms(data_dict)
        stack = transformed["image"]

        if affine is None:
            affine = transformed["image_meta_dict"]["affine"]

        # Normalize
        stack_min = stack.min()
        stack_max = stack.max()
        stack = (stack - stack_min) / (stack_max - stack_min + 1e-8)

        # To numpy
        stack_np = stack.squeeze().cpu().numpy()

        if original_shape is None:
            original_shape = np.array(stack_np.shape)

        # Pad
        padded, pb, _ = pad_to_multiple_of_32(stack_np)
        if pad_before is None:
            pad_before = pb

        # To tensor
        lr_stacks_padded.append(
            torch.from_numpy(padded).float().unsqueeze(0).unsqueeze(0).to(device)
        )

    return lr_stacks_padded, affine, original_shape, pad_before


def predict_single_volume(
    model: torch.nn.Module,
    input_path: str,
    output_path: str,
    generator: Optional[HRLRDataGenerator] = None,
    target_res: List[float] = [1.0, 1.0, 1.0],
    device: str = "cuda",
    use_sliding_window: bool = False,
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    overlap: float = 0.5,
    input_stack_paths: Optional[List[str]] = None,
) -> dict:
    """
    Run inference on a single volume.

    Args:
        model: Trained U-Net model
        input_path: Path to input HR volume (if generating stacks)
        output_path: Path to save SR output
        generator: Data generator for creating LR stacks
        target_res: Target resolution in mm
        device: Inference device
        use_sliding_window: Use sliding window for large volumes
        patch_size: Patch size for sliding window
        overlap: Overlap ratio for sliding window (0.0 to 1.0)
        input_stack_paths: Pre-existing LR stack paths (alternative to generation)

    Returns:
        Dictionary with inference metadata
    """
    transforms = create_inference_transforms(target_res)

    # Load or generate LR stacks
    if input_stack_paths is not None:
        lr_stacks_padded, affine, original_shape, pad_before = load_existing_stacks(
            input_stack_paths, transforms, device
        )
    else:
        lr_stacks_padded, affine, original_shape, pad_before = load_and_generate_stacks(
            input_path, generator, transforms, device
        )

    # Concatenate stacks for U-Net input
    lr_concat = concatenate_stacks(lr_stacks_padded)  # (1, N, D, H, W)

    # Inference
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

    # Post-process
    sr_output = sr_output.squeeze().cpu().numpy()  # (D, H, W)
    sr_output = unpad_volume(sr_output, pad_before, original_shape)
    sr_output = np.clip(sr_output, 0, 1)

    # Save output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_nii = nib.Nifti1Image(sr_output, affine)
    nib.save(out_nii, output_path)

    return {
        'output_path': output_path,
        'output_shape': sr_output.shape,
        'value_range': (float(sr_output.min()), float(sr_output.max())),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test U-Net Super-Resolution Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input/Output
    parser.add_argument("--input", type=str,
                       help="Input HR volume file or directory")
    parser.add_argument("--input_stacks", type=str, nargs='+',
                       help="Pre-existing LR stack files (2-3 files)")
    parser.add_argument("--output", type=str, required=True,
                       help="Output file or directory for SR volumes")

    # Model
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained U-Net checkpoint (.pth)")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device for inference")

    # Data generation (if using --input)
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

    # Inference options
    parser.add_argument("--use_sliding_window", action="store_true",
                       help="Use sliding window inference for large volumes")
    parser.add_argument("--patch_size", type=int, nargs=3,
                       default=[128, 128, 128],
                       help="Patch size for sliding window (x y z)")
    parser.add_argument("--overlap", type=float, default=0.5,
                       help="Overlap ratio for sliding window (0.0-1.0)")

    args = parser.parse_args()

    # Validation
    if args.input is None and args.input_stacks is None:
        raise ValueError("Must provide either --input or --input_stacks")

    # Device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"

    # Load model
    print("\nLoading model...")
    model, checkpoint = load_unet_from_checkpoint(args.model, device)

    # Create data generator if needed
    generator = None
    if args.input is not None:
        print("\nInitializing data generator...")
        generator = HRLRDataGenerator(
            atlas_res=args.target_res,
            target_res=[1.0, 1.0, 1.0],
            output_shape=args.output_shape,
            max_res_aniso=args.max_res_aniso,
            randomise_res=False,  # Deterministic for testing
            apply_intensity_aug=False,
            orientation_dropout_prob=0.0,
            upsample_mode=args.upsample_mode,
        )

    # Process
    if args.input_stacks or (args.input and os.path.isfile(args.input)):
        # Single volume
        print("\nProcessing single volume...")
        result = predict_single_volume(
            model=model,
            input_path=args.input,
            output_path=args.output,
            generator=generator,
            target_res=args.target_res,
            device=device,
            use_sliding_window=args.use_sliding_window,
            patch_size=tuple(args.patch_size),
            overlap=args.overlap,
            input_stack_paths=args.input_stacks,
        )
        print(f"\nSaved SR output: {result['output_path']}")
        print(f"  Output shape: {result['output_shape']}")
        print(f"  Value range: [{result['value_range'][0]:.4f}, {result['value_range'][1]:.4f}]")
    else:
        # Batch processing
        input_dir = args.input
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)

        input_files = sorted(Path(input_dir).glob("*.nii.gz"))
        print(f"\nProcessing {len(input_files)} volumes from {input_dir}")

        for input_file in tqdm(input_files, desc="Processing volumes"):
            output_file = os.path.join(
                output_dir,
                input_file.stem.replace('.nii', '') + '_sr.nii.gz'
            )

            try:
                predict_single_volume(
                    model=model,
                    input_path=str(input_file),
                    output_path=output_file,
                    generator=generator,
                    target_res=args.target_res,
                    device=device,
                    use_sliding_window=args.use_sliding_window,
                    patch_size=tuple(args.patch_size),
                    overlap=args.overlap,
                )
            except Exception as e:
                print(f"\nError processing {input_file.name}: {e}")
                continue

        print(f"\nProcessed {len(input_files)} volumes → {output_dir}")


if __name__ == "__main__":
    main()
