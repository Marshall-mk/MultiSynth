"""
Inference script for U-Net Super-Resolution with Multi-Stack Input

This script performs inference using trained U-Net models (SegResNet or SwinUNETR)
with orthogonal low-resolution stacks (axial, coronal, sagittal) as input.
"""

import os
import argparse
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import gc

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
)

from monai.networks.nets import SwinUNETR, SegResNet
from src.utils import (
    pad_to_multiple_of_32,
    unpad_volume,
)


def cuda_cleanup():
    """Best-effort GPU memory cleanup between cases."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


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


def parse_stack_selection(use_stacks: str):
    """
    Parse stack selection string to list of indices.

    Args:
        use_stacks: "all", "012", "01", "02", "12", "0", "1", "2"

    Returns:
        List of stack indices, e.g., [0, 1, 2], [0, 1], or [0]
    """
    if use_stacks.lower() == "all":
        return [0, 1, 2]

    # Parse digit string
    try:
        indices = [int(c) for c in use_stacks if c.isdigit()]
        if not indices or len(indices) > 3:
            raise ValueError(f"Invalid stack selection: {use_stacks}")
        if any(i not in [0, 1, 2] for i in indices):
            raise ValueError(f"Stack indices must be 0, 1, or 2")
        return sorted(indices)
    except Exception as e:
        raise ValueError(
            f"Invalid --use_stacks format: {use_stacks}. "
            f"Use 'all', '012', '01', '02', '12', '0', '1', or '2'"
        )


def get_swinunetr(
    in_channels: int = 1,
    out_channels: int = 1,
    feature_size: int = 48,
    use_checkpoint: bool = False,
    spatial_dims: int = 3,
):
    """Create SwinUNETR model."""
    return SwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
        spatial_dims=spatial_dims,
    )


def get_segresnet(
    in_channels: int = 1,
    out_channels: int = 1,
    init_filters: int = 32,
    blocks_down: tuple = (1, 2, 2, 4),
    blocks_up: tuple = (1, 1, 1),
    dropout_prob: float = None,
    spatial_dims: int = 3,
):
    """Create SegResNet model."""
    return SegResNet(
        in_channels=in_channels,
        out_channels=out_channels,
        init_filters=init_filters,
        blocks_down=blocks_down,
        blocks_up=blocks_up,
        dropout_prob=dropout_prob,
        spatial_dims=spatial_dims,
    )


def load_unet_from_checkpoint(checkpoint_path, device="cuda"):
    """
    Load U-Net model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        device: Device to load model on

    Returns:
        Tuple of (model, checkpoint_data, stack_indices)
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model configuration
    model_config = checkpoint.get('model_config', {})
    model_type = checkpoint.get('model_type', 'unet')

    # Get stack selection
    use_stacks = model_config.get('use_stacks', 'all')
    num_input_channels = model_config.get('num_input_channels', 3)

    print(f"Model configuration:")
    print(f"  - model_type: {model_type}")
    print(f"  - use_stacks: {use_stacks}")
    print(f"  - num_input_channels: {num_input_channels}")

    # Parse stack indices
    stack_indices = parse_stack_selection(use_stacks)

    # Validate
    if len(stack_indices) != num_input_channels:
        print(f"  Warning: Stack indices ({len(stack_indices)}) != num_input_channels ({num_input_channels})")
        print(f"           Using num_input_channels from checkpoint")

    # Get model architecture type from checkpoint filename or config
    if 'swinunetr' in checkpoint_path.lower():
        arch_type = 'swinunetr'
    elif 'segresnet' in checkpoint_path.lower():
        arch_type = 'segresnet'
    else:
        # Default to segresnet if not clear from path
        arch_type = 'segresnet'
        print(f"  Note: Model architecture not clear from checkpoint path, defaulting to {arch_type}")

    # Create model
    print(f"Creating {arch_type} model with {num_input_channels} input channels...")

    if arch_type == 'swinunetr':
        model = get_swinunetr(
            in_channels=num_input_channels,
            out_channels=1,
            feature_size=48,
            use_checkpoint=False,
            spatial_dims=3,
        )
    elif arch_type == 'segresnet':
        model = get_segresnet(
            in_channels=num_input_channels,
            out_channels=1,
            init_filters=32,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=None,
            spatial_dims=3,
        )
    else:
        raise ValueError(f"Unknown model architecture: {arch_type}")

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    return model, checkpoint, stack_indices


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

    # Load first valid stack to get original metadata
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
                metadata['affine_original'] = np.diag([target_res[0], target_res[1], target_res[2], 1.0])
                metadata['resolution_original'] = np.array(target_res)
            break

    # If no valid stacks found, use default
    if metadata['affine_original'] is None:
        metadata['affine_original'] = np.diag([target_res[0], target_res[1], target_res[2], 1.0])
        metadata['resolution_original'] = np.array(target_res)

    transforms = create_inference_transforms(target_res)
    lr_stacks_tensors = [None, None, None]
    reference_shape = None

    orientation_mapping = [
        ("Axial", "Stack 0", "High-res in D axis"),
        ("Coronal", "Stack 1", "High-res in H axis"),
        ("Sagittal", "Stack 2", "High-res in W axis"),
    ]

    for i, stack_path in enumerate(stack_paths):
        orientation, stack_num, description = orientation_mapping[i]

        if stack_path is None:
            print(f"    - {stack_num} ({orientation}): [MISSING - will use dummy stack]")
            continue

        print(f"    - {stack_num} ({orientation}): {stack_path}")
        print(f"      └─ {description}")

        # Load and preprocess
        data_dict = {"image": stack_path}
        data = transforms(data_dict)
        volume = data["image"]

        # Convert to numpy
        if isinstance(volume, torch.Tensor):
            volume_np = volume.cpu().numpy()
        else:
            volume_np = np.array(volume)

        # Remove channel dimension (C, D, H, W) -> (D, H, W)
        if volume_np.ndim == 4 and volume_np.shape[0] == 1:
            volume_np = volume_np[0]

        # Store reference shape
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
            dummy = torch.zeros((1,) + reference_shape, dtype=torch.float32)
            lr_stacks_tensors[i] = dummy
            print(f"    Created dummy stack for {orientation_mapping[i][0]}: shape {dummy.shape}")

    # Create isotropic affine
    metadata['shape_isotropic'] = lr_stacks_tensors[0].squeeze().shape
    metadata['affine_isotropic'] = create_isotropic_affine(
        target_res,
        metadata['shape_isotropic'],
        metadata['affine_original']
    )

    print(f"    Final stack shapes: {[s.shape for s in lr_stacks_tensors]}")
    return lr_stacks_tensors, metadata


def concatenate_stacks(lr_stacks_list, stack_indices):
    """
    Concatenate selected LR stacks channel-wise.

    Args:
        lr_stacks_list: List of 3 tensors, each (1, D, H, W)
        stack_indices: Indices of stacks to use, e.g., [0, 1, 2] or [0, 2]

    Returns:
        Concatenated tensor (1, N, D, H, W) where N = len(stack_indices)
    """
    selected_stacks = [lr_stacks_list[i] for i in stack_indices]
    # Each stack is (1, D, H, W), concatenate along channel dim
    return torch.cat(selected_stacks, dim=0).unsqueeze(0)  # (1, N, D, H, W)


def predict_single_volume(
    model,
    output_path,
    device="cuda",
    input_stack_paths=None,
    target_res=[1.0, 1.0, 1.0],
    stack_indices=[0, 1, 2],
):
    """
    Run U-Net inference on a single volume.

    Args:
        model: Trained U-Net model
        output_path: Path to save output
        device: 'cuda' or 'cpu'
        input_stack_paths: List of 3 pre-existing stack paths [axial, coronal, sagittal]
        target_res: Target resolution [x, y, z] in mm
        stack_indices: Indices of stacks to use (e.g., [0, 1, 2])
    """
    for i, path in enumerate(input_stack_paths):
        print(f"  {['Axial', 'Coronal', 'Sagittal'][i]}: {path}")

    # Load orthogonal stacks
    lr_stacks, metadata = load_orthogonal_stacks_from_files(input_stack_paths, target_res)
    affine = metadata['affine_isotropic']

    # Pad to multiple of 32
    original_shape = lr_stacks[0].squeeze().shape
    lr_stacks_padded = []

    for stack in lr_stacks:
        stack_np = stack.squeeze().cpu().numpy()
        padded, pad_before, orig_shape = pad_to_multiple_of_32(stack_np)
        lr_stacks_padded.append(
            torch.from_numpy(padded).float().unsqueeze(0)  # (1, D, H, W)
        )

    # Concatenate selected stacks
    print(f"  Using stacks: {stack_indices} ({[['Axial', 'Coronal', 'Sagittal'][i] for i in stack_indices]})")
    lr_input = concatenate_stacks(lr_stacks_padded, stack_indices)  # (1, N, D, H, W)
    lr_input = lr_input.to(device)

    print(f"  Input shape: {lr_input.shape}")

    # Run inference
    try:
        model.eval()
        with torch.no_grad():
            print("  Running inference...")
            sr_output = model(lr_input)

        # Convert to numpy
        sr_output = sr_output.squeeze().cpu().numpy()  # (D, H, W)
        print(f"  SR output shape before unpad: {sr_output.shape}")

        # Unpad to original shape
        sr_output = unpad_volume(sr_output, pad_before, orig_shape)
        print(f"  SR output shape after unpad: {sr_output.shape}")

        # Clip to [0, 1]
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

    finally:
        try:
            del lr_stacks, lr_stacks_padded, lr_input
            if 'sr_output' in locals():
                del sr_output
        except Exception:
            pass
        cuda_cleanup()


def predict_batch(
    output_paths,
    model_path,
    target_res=[1.0, 1.0, 1.0],
    device="cuda",
    input_stack_paths=None,
):
    """Process multiple volumes in batch."""
    print("=" * 80)
    print("U-Net Inference - Orthogonal Stack Super-Resolution")
    print("=" * 80)

    # Load model
    model, checkpoint, stack_indices = load_unet_from_checkpoint(model_path, device=device)

    print(f"\nInference settings:")
    print(f"  Device: {device}")
    print(f"  Target resolution: {target_res} mm")
    print(f"  Using stacks: {stack_indices}")

    if input_stack_paths:
        print(f"\nProcessing 1 set of stacks...\n")

    # Process volume
    if input_stack_paths:
        print(f"[1/1]")
        try:
            predict_single_volume(
                model=model,
                output_path=output_paths[0] if isinstance(output_paths, list) else output_paths,
                target_res=target_res,
                device=device,
                input_stack_paths=input_stack_paths,
                stack_indices=stack_indices,
            )
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Inference complete!")
    print("=" * 80)


def predict_folder(
    input_stacks_root: str,
    output_root: str,
    model_path: str,
    target_res=[1.0, 1.0, 1.0],
    device="cuda",
    pattern_ax="axial_upsampled.nii.gz",
    pattern_cor="coronal_upsampled.nii.gz",
    pattern_sag="sagittal_upsampled.nii.gz",
    output_name="unet_prediction.nii.gz",
    skip_existing=False,
    fail_fast=False,
):
    """
    Batch inference over a directory of subject folders.

    Each subject folder is expected to contain:
      - axial_upsampled.nii.gz
      - coronal_upsampled.nii.gz
      - sagittal_upsampled.nii.gz
    """
    stacks_root = Path(input_stacks_root)
    if not stacks_root.exists() or not stacks_root.is_dir():
        raise ValueError(f"--input_stacks_root is not a directory: {stacks_root}")

    out_root = Path(output_root) if output_root else stacks_root
    out_root.mkdir(parents=True, exist_ok=True)

    # Load model ONCE
    model, checkpoint, stack_indices = load_unet_from_checkpoint(model_path, device=device)

    subject_dirs = sorted([p for p in stacks_root.iterdir() if p.is_dir()])
    print(f"\nFound {len(subject_dirs)} subject folders in: {stacks_root}\n")

    for i, subj_dir in enumerate(tqdm(subject_dirs, desc="Processing subjects"), start=1):
        subj_id = subj_dir.name

        ax = subj_dir / pattern_ax
        cor = subj_dir / pattern_cor
        sag = subj_dir / pattern_sag

        # Build stack paths
        stack_paths = [
            str(ax) if ax.exists() else None,
            str(cor) if cor.exists() else None,
            str(sag) if sag.exists() else None,
        ]

        # Check if required stacks exist
        missing_required = []
        for idx in stack_indices:
            if stack_paths[idx] is None:
                missing_required.append(idx)

        # Decide output path
        out_path = out_root / subj_id / output_name if output_root else subj_dir / output_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if skip_existing and out_path.exists():
            print(f"\n[{i}/{len(subject_dirs)}] {subj_id} -> skipping (exists): {out_path}")
            continue

        if missing_required:
            msg = f"\n[{i}/{len(subject_dirs)}] {subj_id} -> skipping (missing required stacks: {missing_required})"
            if fail_fast:
                raise FileNotFoundError(msg)
            print(msg)
            continue

        print(f"\n[{i}/{len(subject_dirs)}] {subj_id}")
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"  GPU mem (before): allocated={allocated:.2f}GB reserved={reserved:.2f}GB")

            predict_single_volume(
                model=model,
                output_path=str(out_path),
                device=device,
                input_stack_paths=stack_paths,
                target_res=target_res,
                stack_indices=stack_indices,
            )

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"  GPU mem (after):  allocated={allocated:.2f}GB reserved={reserved:.2f}GB")

        except Exception as e:
            print(f"  ✗ ERROR on {subj_id}: {e}")
            import traceback
            traceback.print_exc()
            if fail_fast:
                raise

    print("\nAll subjects done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-Net Inference with Orthogonal Stacks")

    # Input/output arguments
    parser.add_argument("--input_stacks", type=str, nargs='+', default=None,
                       help="Orthogonal LR stack files (1-3 stacks) in order: axial, coronal, sagittal")
    parser.add_argument("--output", type=str, required=False,
                       help="Output image file or directory")

    # Model arguments
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model checkpoint (.pth file)")

    # Preprocessing arguments
    parser.add_argument("--target_res", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                       help="Target resolution in mm (e.g., 1.0 1.0 1.0)")

    # Inference arguments
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device: cuda or cpu")

    # Folder mode arguments
    parser.add_argument("--input_stacks_root", type=str, default=None,
                       help="Root dir containing subject subfolders of orthogonal stacks")
    parser.add_argument("--output_root", type=str, default=None,
                       help="Where to save outputs for folder mode. If omitted, saves into each subject folder")
    parser.add_argument("--pattern_ax", type=str, default="axial_upsampled.nii.gz")
    parser.add_argument("--pattern_cor", type=str, default="coronal_upsampled.nii.gz")
    parser.add_argument("--pattern_sag", type=str, default="sagittal_upsampled.nii.gz")
    parser.add_argument("--output_name", type=str, default="unet_prediction.nii.gz")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")

    args = parser.parse_args()

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Mode 0: Folder mode (batch over subject directories)
    if getattr(args, "input_stacks_root", None):
        predict_folder(
            input_stacks_root=args.input_stacks_root,
            output_root=getattr(args, "output_root", None),
            model_path=args.model,
            target_res=args.target_res,
            device=args.device,
            pattern_ax=getattr(args, "pattern_ax", "axial_upsampled.nii.gz"),
            pattern_cor=getattr(args, "pattern_cor", "coronal_upsampled.nii.gz"),
            pattern_sag=getattr(args, "pattern_sag", "sagittal_upsampled.nii.gz"),
            output_name=getattr(args, "output_name", "unet_prediction.nii.gz"),
            skip_existing=getattr(args, "skip_existing", False),
            fail_fast=getattr(args, "fail_fast", False),
        )
        raise SystemExit(0)

    # Mode 1: Pre-existing orthogonal stacks (single case)
    if args.input_stacks:
        num_stacks = len(args.input_stacks)
        if not (1 <= num_stacks <= 3):
            raise ValueError(f"Expected 1-3 input stacks, got {num_stacks}")

        # Validate stack files exist
        for i, stack_path in enumerate(args.input_stacks):
            if not Path(stack_path).exists():
                raise ValueError(f"Provided stack {i+1} not found: {stack_path}")

        # Build full [ax, cor, sag] list
        # Assume provided stacks match [axial, coronal, sagittal] order
        if num_stacks == 3:
            input_stack_paths = list(args.input_stacks)
        else:
            # For partial stacks, fill remaining with None
            input_stack_paths = list(args.input_stacks) + [None] * (3 - num_stacks)

        print(f"\n📦 Stack mode: using {num_stacks} stack(s)")

        # Run inference (single case)
        predict_batch(
            output_paths=args.output,
            model_path=args.model,
            target_res=args.target_res,
            device=args.device,
            input_stack_paths=input_stack_paths,
        )
        raise SystemExit(0)

    # If no valid mode was chosen
    raise ValueError(
        "No valid inference mode selected. Use one of:\n"
        "  - --input_stacks_root <dir>\n"
        "  - --input_stacks <1-3 paths>\n"
    )
