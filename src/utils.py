"""
Utility functions for the U-HVED architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import glob
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple, Union
from monai.losses import SSIMLoss as MonaiSSIM


class PixelShuffle3d(nn.Module):
    """
    3D pixel shuffle operation for sub-pixel convolution upsampling.

    Rearranges elements in a tensor of shape (B, C*r^3, D, H, W) to (B, C, D*r, H*r, W*r)
    where r is the upscale factor.

    This is the 3D equivalent of PyTorch's nn.PixelShuffle for 2D data.

    Args:
        upscale_factor (int): Factor to increase spatial resolution by

    Example:
        >>> ps = PixelShuffle3d(upscale_factor=2)
        >>> input = torch.randn(1, 8, 4, 4, 4)  # (B, C*r^3, D, H, W) where r=2, C=1
        >>> output = ps(input)  # (1, 1, 8, 8, 8) - (B, C, D*r, H*r, W*r)
    """

    def __init__(self, upscale_factor: int):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of 3D pixel shuffle.

        Args:
            x: Input tensor of shape (B, C*r^3, D, H, W)

        Returns:
            Output tensor of shape (B, C, D*r, H*r, W*r)
        """
        batch_size, in_channels, in_depth, in_height, in_width = x.size()
        r = self.upscale_factor

        # Calculate output channels
        out_channels = in_channels // (r ** 3)

        if in_channels != out_channels * (r ** 3):
            raise ValueError(
                f"Input channels ({in_channels}) must be divisible by "
                f"upscale_factor^3 ({r}^3 = {r**3})"
            )

        # Reshape: (B, C*r^3, D, H, W) -> (B, C, r, r, r, D, H, W)
        x = x.view(batch_size, out_channels, r, r, r, in_depth, in_height, in_width)

        # Permute: (B, C, r, r, r, D, H, W) -> (B, C, D, r, H, r, W, r)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        # Reshape: (B, C, D, r, H, r, W, r) -> (B, C, D*r, H*r, W*r)
        x = x.view(batch_size, out_channels, in_depth * r, in_height * r, in_width * r)

        return x

    def extra_repr(self) -> str:
        return f'upscale_factor={self.upscale_factor}'


def pixel_shuffle_3d(x: torch.Tensor, upscale_factor: int) -> torch.Tensor:
    """
    Functional interface for 3D pixel shuffle.

    Args:
        x: Input tensor of shape (B, C*r^3, D, H, W)
        upscale_factor: Factor to increase spatial resolution by

    Returns:
        Output tensor of shape (B, C, D*r, H*r, W*r)
    """
    ps = PixelShuffle3d(upscale_factor)
    return ps(x)


# =============================================================================
# Padding Utilities for Inference
# =============================================================================


def pad_to_multiple_of_32(
    volume: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pad volume to make all dimensions multiples of 32 (centered padding).

    This is used for inference with UNet models that require input dimensions
    to be divisible by 2^n_levels (e.g., 32 for 5 levels).

    Args:
        volume: Input volume array (D, H, W)

    Returns:
        Tuple of (padded volume, padding before, original shape)

    Example:
        >>> volume = np.random.rand(100, 120, 90)
        >>> padded, pad_before, orig_shape = pad_to_multiple_of_32(volume)
        >>> print(padded.shape)  # (128, 128, 96) - next multiples of 32
        >>> # Later unpad: volume = padded[pad_before[0]:pad_before[0]+orig_shape[0], ...]
    """
    shape = np.array(volume.shape)
    # Calculate target shape (next multiple of 32)
    target_shape = (np.ceil(shape / 32.0) * 32).astype("int")

    # Calculate padding (centered)
    padding = target_shape - shape
    pad_before = np.floor(padding / 2).astype("int")
    pad_after = padding - pad_before

    # Pad the volume
    padded = np.pad(
        volume,
        [(pad_before[i], pad_after[i]) for i in range(3)],
        mode="constant",
        constant_values=0,
    )

    return padded, pad_before, shape


def unpad_volume(
    volume: np.ndarray, pad_before: np.ndarray, original_shape: np.ndarray
) -> np.ndarray:
    """
    Remove padding from volume that was added by pad_to_multiple_of_32.

    Args:
        volume: Padded volume array
        pad_before: Padding amounts before (from pad_to_multiple_of_32)
        original_shape: Original shape before padding (from pad_to_multiple_of_32)

    Returns:
        Unpadded volume with original shape

    Example:
        >>> padded, pad_before, orig_shape = pad_to_multiple_of_32(volume)
        >>> # ... process padded volume ...
        >>> result = unpad_volume(processed, pad_before, orig_shape)
    """
    return volume[
        pad_before[0] : pad_before[0] + original_shape[0],
        pad_before[1] : pad_before[1] + original_shape[1],
        pad_before[2] : pad_before[2] + original_shape[2],
    ]


# =============================================================================
# Model Statistics and Memory Profiling
# =============================================================================


def get_model_statistics(
    model: nn.Module,
    input_shapes: Optional[List[Tuple]] = None,
    device: str = "cuda",
    base_channels: int = 32,
    num_scales: int = 4
):
    """
    Calculate model parameters and memory usage.

    Args:
        model: PyTorch model
        input_shapes: Optional list of input tensor shapes for memory estimation
                     For U-HVED: [(B, C, D, H, W), (B, C, D, H, W), (B, C, D, H, W)]
        device: Device to use for memory estimation
        base_channels: Base channel count for U-HVED architecture (default 32)
        num_scales: Number of hierarchical scales in U-HVED (default 4)

    Returns:
        Dictionary with model statistics
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    # Model memory (parameters + buffers)
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
    model_memory_mb = (param_memory + buffer_memory) / (1024 ** 2)

    stats = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'model_memory_mb': model_memory_mb,
    }

    # Estimate input data memory if shapes provided
    if input_shapes:
        # Input data memory (always float32 due to explicit .float() conversion)
        total_input_memory = 0
        for shape in input_shapes:
            # Data is float32 (4 bytes per element)
            # Note: Training code explicitly converts inputs to float32 via .float()
            # even when using mixed precision (train.py lines 495-496, 608-609).
            # Mixed precision only affects internal model computations, not input data.
            numel = 1
            for dim in shape:
                numel *= dim
            total_input_memory += numel * 4  # 4 bytes for float32

        stats['input_memory_mb'] = total_input_memory / (1024 ** 2)

        # Estimate activation memory (forward pass feature maps)
        # Assumes U-HVED architecture with 3 orientations
        # Note: This is an approximation. Actual memory may vary based on:
        # - Gradient checkpointing (would reduce activation memory)
        # - Mixed precision (would use 2 bytes for internal activations instead of 4)
        # - Optimizer state (not included here - adds ~2x model memory for Adam)
        activation_memory = estimate_uhved_activation_memory(
            input_shapes=input_shapes,
            base_channels=base_channels,
            num_scales=num_scales,
            bytes_per_element=4  # float32 activations
        )
        stats['activation_memory_mb'] = activation_memory / (1024 ** 2)

    return stats


def estimate_uhved_activation_memory(
    input_shapes: List[Tuple],
    base_channels: int = 32,
    num_scales: int = 4,
    bytes_per_element: int = 4  # float32
):
    """
    Estimate activation memory for U-HVED model.

    U-HVED has:
    - 3 independent encoders (one per orientation: axial, coronal, sagittal)
    - Multi-scale latent fusion
    - 1 decoder for SR output
    - Optional orientation reconstruction decoders

    Args:
        input_shapes: List of input tensor shapes [(B, C, D, H, W), ...]
        base_channels: Base channel count (default 32)
        num_scales: Number of hierarchical scales (default 4)
        bytes_per_element: Bytes per element (4 for float32, 2 for float16)

    Returns:
        Estimated activation memory in bytes
    """
    if not input_shapes or len(input_shapes) == 0:
        return 0

    # Assume all inputs have same shape (orthogonal stacks)
    B, C, D, H, W = input_shapes[0]

    total_activation_memory = 0

    # 1. ENCODER ACTIVATIONS (3 encoders, one per orientation)
    # Each encoder has num_scales levels with downsampling by 2 at each level
    num_orientations = len(input_shapes)  # Typically 3 (axial, coronal, sagittal)

    for orientation_idx in range(num_orientations):
        spatial_dims = [D, H, W]

        for scale in range(num_scales):
            # Channel count doubles at each scale
            channels = base_channels * (2 ** scale)

            # Spatial dimensions halve at each scale (downsampling by 2)
            current_dims = [max(1, dim // (2 ** scale)) for dim in spatial_dims]

            # Conv block activations (typically 2 conv layers per block)
            # Each conv: input + output activations
            numel = B * channels * current_dims[0] * current_dims[1] * current_dims[2]
            total_activation_memory += numel * 2  # 2 conv layers per block

    # 2. LATENT SPACE ACTIVATIONS (Fusion of 3 encoder outputs)
    # At the bottleneck: B x (base_channels * 2^(num_scales-1)) x (D/2^(num_scales-1)) x (H/2^(num_scales-1)) x (W/2^(num_scales-1))
    bottleneck_channels = base_channels * (2 ** (num_scales - 1))
    bottleneck_dims = [max(1, dim // (2 ** (num_scales - 1))) for dim in [D, H, W]]

    # Fusion concatenates 3 orientation features: 3 x bottleneck_channels
    # Then reduces back to bottleneck_channels via conv
    fusion_numel = B * (bottleneck_channels * num_orientations) * bottleneck_dims[0] * bottleneck_dims[1] * bottleneck_dims[2]
    total_activation_memory += fusion_numel

    # Posterior (mu + logvar for VAE)
    posterior_numel = B * bottleneck_channels * bottleneck_dims[0] * bottleneck_dims[1] * bottleneck_dims[2]
    total_activation_memory += posterior_numel * 2  # mu and logvar

    # 3. DECODER ACTIVATIONS (SR decoder, upsampling path)
    spatial_dims = bottleneck_dims
    for scale in range(num_scales - 1, -1, -1):
        # Channels halve as we go up (opposite of encoder)
        channels = base_channels * (2 ** scale)

        # Spatial dimensions double at each level (upsampling by 2)
        current_dims = [max(1, dim * (2 ** (num_scales - 1 - scale))) for dim in bottleneck_dims]

        # Skip connections from encoders (concatenated from all 3 orientations)
        skip_channels = channels * num_orientations

        # Decoder block: upsampled features + skip connections + conv output
        # Upsample: B x channels x current_dims
        # After concat with skip: B x (channels + skip_channels) x current_dims
        # After conv: B x channels x current_dims
        numel_upsample = B * channels * current_dims[0] * current_dims[1] * current_dims[2]
        numel_concat = B * (channels + skip_channels) * current_dims[0] * current_dims[1] * current_dims[2]
        numel_conv = B * channels * current_dims[0] * current_dims[1] * current_dims[2]

        total_activation_memory += numel_upsample + numel_concat + numel_conv

    # 4. ORIENTATION RECONSTRUCTION DECODERS (if enabled)
    # Typically 3 smaller decoders to reconstruct input orientations
    # Approximate as ~30% of main decoder memory
    orientation_decoder_memory = total_activation_memory * 0.15  # Conservative estimate
    total_activation_memory += orientation_decoder_memory

    # Convert to bytes
    total_activation_memory_bytes = int(total_activation_memory * bytes_per_element)

    return total_activation_memory_bytes


def print_model_statistics(
    model: nn.Module,
    input_shapes: Optional[List[Tuple]] = None,
    device: str = "cuda",
    base_channels: int = 32,
    num_scales: int = 4
):
    """
    Print formatted model statistics.

    Args:
        model: PyTorch model
        input_shapes: Optional list of input tensor shapes
        device: Device to use for memory estimation
        base_channels: Base channel count for U-HVED architecture (default 32)
        num_scales: Number of hierarchical scales in U-HVED (default 4)
    """
    stats = get_model_statistics(model, input_shapes, device, base_channels, num_scales)

    print("=" * 80)
    print("MODEL STATISTICS")
    print("=" * 80)
    print(f"Total parameters:        {stats['total_params']:>15,}")
    print(f"Trainable parameters:    {stats['trainable_params']:>15,}")
    print(f"Non-trainable parameters:{stats['non_trainable_params']:>15,}")
    print(f"Model memory:            {stats['model_memory_mb']:>15.2f} MB (params + buffers)")

    if 'input_memory_mb' in stats:
        print(f"Input data memory:       {stats['input_memory_mb']:>15.2f} MB (per batch, float32)")

        if 'activation_memory_mb' in stats:
            print(f"Activation memory (est): {stats['activation_memory_mb']:>15.2f} MB (forward pass)")
            total_memory = stats['model_memory_mb'] + stats['input_memory_mb'] + stats['activation_memory_mb']
            print(f"Total GPU memory (est):  {total_memory:>15.2f} MB")
            print(f"  Note: Excludes gradients (~{stats['activation_memory_mb']:.2f} MB) and optimizer state (~{stats['model_memory_mb']*2:.2f} MB for Adam)")
        else:
            total_memory = stats['model_memory_mb'] + stats['input_memory_mb']
            print(f"Total GPU memory (est):  {total_memory:>15.2f} MB (model + input only)")

    print("=" * 80)


def get_gpu_memory_stats(device: Optional[torch.device] = None):
    """
    Get current GPU memory usage statistics.

    Args:
        device: CUDA device to query. If None, uses current device.

    Returns:
        Dictionary with memory statistics in MB, or None if not using CUDA
    """
    if not torch.cuda.is_available():
        return None

    if device is None:
        device = torch.cuda.current_device()
    elif isinstance(device, str):
        device = torch.device(device)

    # Get device index
    if device.type != 'cuda':
        return None

    device_idx = device.index if device.index is not None else torch.cuda.current_device()

    # Memory statistics
    allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 2)  # MB
    reserved = torch.cuda.memory_reserved(device_idx) / (1024 ** 2)    # MB
    max_allocated = torch.cuda.max_memory_allocated(device_idx) / (1024 ** 2)  # MB
    max_reserved = torch.cuda.max_memory_reserved(device_idx) / (1024 ** 2)    # MB

    # Get total GPU memory
    total_memory = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 2)  # MB

    return {
        'allocated_mb': allocated,
        'reserved_mb': reserved,
        'max_allocated_mb': max_allocated,
        'max_reserved_mb': max_reserved,
        'total_mb': total_memory,
        'utilization_pct': (allocated / total_memory) * 100 if total_memory > 0 else 0
    }


def print_gpu_memory_stats(device: Optional[torch.device] = None, prefix: str = ""):
    """
    Print formatted GPU memory statistics.

    Args:
        device: CUDA device to query
        prefix: Optional prefix for the output (e.g., "After first batch: ")
    """
    stats = get_gpu_memory_stats(device)

    if stats is None:
        print(f"{prefix}GPU memory tracking not available (not using CUDA)")
        return

    print("=" * 80)
    if prefix:
        print(f"{prefix.upper()}")
    print("GPU MEMORY USAGE")
    print("=" * 80)
    print(f"Allocated:               {stats['allocated_mb']:>15.2f} MB")
    print(f"Reserved (cached):       {stats['reserved_mb']:>15.2f} MB")
    print(f"Max allocated:           {stats['max_allocated_mb']:>15.2f} MB")
    print(f"Max reserved:            {stats['max_reserved_mb']:>15.2f} MB")
    print(f"Total GPU memory:        {stats['total_mb']:>15.2f} MB")
    print(f"Utilization:             {stats['utilization_pct']:>15.1f} %")
    print("=" * 80)


# =============================================================================
# Data Loading Utilities
# =============================================================================


def load_image_paths_from_csv(
    csv_path: Union[str, Path],
    base_dir: Union[str, Path],
    split: str = "train",
    acquisition_types: Optional[List[str]] = ["3D"],
    mri_classifications: Optional[List[str]] = None,
    filter_4d: bool = True,
    log_filtered_path: Optional[Union[str, Path]] = None,
) -> List[Path]:
    """
    Load image paths from a CSV file with filtering by split, acquisition type, and MRI classification.

    Args:
        csv_path: Path to CSV file
        base_dir: Base directory to prepend to relative paths
        split: Data split to filter for ('train', 'val', or 'test')
        acquisition_types: List of acquisition types to include (default: ['3D'])
                          Set to None to include all types, or pass specific list like ['3D', '2D']
        mri_classifications: List of MRI classifications to include (e.g., ['T1', 'T2', 'FLAIR'])
                            Set to None to include all classifications
        filter_4d: If True, filters out 4D images (with time dimension) using 'dimensions' column
        log_filtered_path: Optional path to save list of filtered 4D images (CSV format)

    Returns:
        List of absolute image paths

    CSV Format:
        The CSV should have the following columns:
        - relative_path: Relative path to the image file
        - mr_acquisition_type: Type of MR acquisition ('3D' or '2D')
        - split: Data split ('train', 'val', or 'test')
        - MRI_classification (optional): MRI classification type ('T1', 'T2', 'FLAIR', etc.)
        - dimensions (optional): Image dimensions as tuple string, e.g., "(256, 256, 128)"

    Example CSV:
        relative_path,mr_acquisition_type,split,MRI_classification,dimensions
        images/scan001.nii.gz,3D,train,T1,"(256, 256, 128)"
        images/scan002.nii.gz,2D,train,T2,"(256, 256, 128)"
        images/scan003.nii.gz,3D,val,FLAIR,"(256, 256, 128, 10)"
        images/scan004.nii.gz,3D,test,T1,"(256, 256, 128)"

    Example:
        >>> # Load 3D training images only (filtering out 4D) with logging
        >>> train_paths = load_image_paths_from_csv(
        ...     'data.csv',
        ...     base_dir='/data/mri',
        ...     split='train',
        ...     filter_4d=True,
        ...     log_filtered_path='./model/filtered_4d_train.csv'
        ... )
        >>>
        >>> # Load T1 and T2 training images only
        >>> train_paths = load_image_paths_from_csv(
        ...     'data.csv',
        ...     base_dir='/data/mri',
        ...     split='train',
        ...     mri_classifications=['T1', 'T2']
        ... )
        >>>
        >>> # Load validation images of all types (including 4D)
        >>> val_paths = load_image_paths_from_csv(
        ...     'data.csv',
        ...     base_dir='/data/mri',
        ...     split='val',
        ...     acquisition_types=None,
        ...     filter_4d=False
        ... )
    """
    csv_path = Path(csv_path)
    base_dir = Path(base_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    # Read CSV
    df = pd.read_csv(csv_path)

    # Validate required columns
    required_columns = ["relative_path", "mr_acquisition_type", "split"]
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"CSV missing required columns: {missing_columns}. "
            f"Found columns: {list(df.columns)}"
        )

    # Filter by split
    df_filtered = df[df["split"] == split].copy()

    if len(df_filtered) == 0:
        raise ValueError(
            f"No images found for split '{split}'. "
            f"Available splits: {df['split'].unique().tolist()}"
        )

    # Filter by acquisition type if specified
    # acquisition_types=None means include all types (no filtering)
    # acquisition_types=['3D'] is the default (3D only)
    if acquisition_types is not None:
        df_filtered = df_filtered[
            df_filtered["mr_acquisition_type"].isin(acquisition_types)
        ]

        if len(df_filtered) == 0:
            raise ValueError(
                f"No images found for split '{split}' with acquisition types {acquisition_types}. "
                f"Available acquisition types: {df[df['split'] == split]['mr_acquisition_type'].unique().tolist()}"
            )
    else:
        # Include all acquisition types
        acquisition_types = df_filtered["mr_acquisition_type"].unique().tolist()

    # Filter by MRI classification if specified
    if mri_classifications is not None and "MRI_classification" in df_filtered.columns:
        df_filtered = df_filtered[
            df_filtered["MRI_classification"].isin(mri_classifications)
        ]

        if len(df_filtered) == 0:
            raise ValueError(
                f"No images found for split '{split}' with MRI classifications {mri_classifications}. "
                f"Available MRI classifications: {df[df['split'] == split]['MRI_classification'].unique().tolist()}"
            )

        print(f"Filtered by MRI classifications: {mri_classifications}")
    elif mri_classifications is not None and "MRI_classification" not in df_filtered.columns:
        print(
            "Warning: 'MRI_classification' column not found in CSV. "
            "Cannot filter by MRI classification. Continuing with all MRI types."
        )

    # Filter out 4D images if requested
    if filter_4d and "dimensions" in df_filtered.columns:
        initial_count = len(df_filtered)

        def is_3d_image(dim_str):
            """Check if dimensions string represents a 3D image (not 4D)."""
            if pd.isna(dim_str):
                # If dimensions column is missing for this row, assume it's okay (will be caught later)
                return True

            try:
                # Parse dimension string - could be "(256, 256, 128)" or similar
                dim_str = str(dim_str).strip()
                # Remove parentheses and split by comma
                dim_str = dim_str.replace("(", "").replace(")", "").strip()
                dims = [int(d.strip()) for d in dim_str.split(",") if d.strip()]
                # Return True if exactly 3 dimensions (3D image)
                return len(dims) == 3
            except Exception as e:
                # If parsing fails, log warning and assume it's okay
                print(f"Warning: Could not parse dimensions '{dim_str}': {e}")
                return True

        # Identify 4D images before filtering
        is_3d_mask = df_filtered["dimensions"].apply(is_3d_image)
        filtered_4d_images = df_filtered[~is_3d_mask].copy()

        # Apply filter
        df_filtered = df_filtered[is_3d_mask].copy()

        filtered_count = len(filtered_4d_images)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} 4D images (with time dimension)")

            # Save filtered images to log file if requested
            if log_filtered_path is not None:
                log_path = Path(log_filtered_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)

                # Add reason column
                filtered_4d_images["filter_reason"] = "4D image (time dimension)"

                # Save to CSV
                filtered_4d_images.to_csv(log_path, index=False)
                print(f"Saved list of filtered 4D images to: {log_path}")

        if len(df_filtered) == 0:
            raise ValueError(
                f"No 3D images found for split '{split}' after filtering out 4D images. "
                f"All {initial_count} images had 4 dimensions."
            )
    elif filter_4d and "dimensions" not in df_filtered.columns:
        print(
            "Warning: 'dimensions' column not found in CSV. Cannot filter 4D images. "
            "Consider adding a 'dimensions' column to your CSV for better filtering."
        )

    # Convert relative paths to absolute paths
    image_paths = []
    missing_files = []
    total_in_csv = len(df_filtered)

    for rel_path in df_filtered["relative_path"]:
        abs_path = base_dir / rel_path
        if not abs_path.exists():
            missing_files.append(abs_path)
        else:
            image_paths.append(abs_path)

    # Report missing files
    if len(missing_files) > 0:
        print(f"Warning: {len(missing_files)}/{total_in_csv} files not found on disk")
        # Show first few missing files
        max_show = 5
        for missing_path in missing_files[:max_show]:
            print(f"  - Missing: {missing_path}")
        if len(missing_files) > max_show:
            print(f"  ... and {len(missing_files) - max_show} more")

    if len(image_paths) == 0:
        raise ValueError(
            f"No valid image files found for split '{split}'. "
            f"Check that files exist in {base_dir}"
        )

    # Build summary message
    summary = f"✓ Loaded {len(image_paths)}/{total_in_csv} {split} images"
    summary += f" (acquisition types: {acquisition_types}"
    if mri_classifications is not None:
        summary += f", MRI classifications: {mri_classifications}"
    summary += ")"
    print(summary)

    return image_paths


def get_image_paths(
    image_dir=None,
    csv_file=None,
    base_dir=None,
    split="train",
    model_dir=None,
    mri_classifications=None,
    acquisition_types=["3D"],
    filter_4d=True,
):
    """
    Get image paths from either directory or CSV file.

    Args:
        image_dir: Directory containing images (mutually exclusive with csv_file)
        csv_file: CSV file with image metadata (mutually exclusive with image_dir)
        base_dir: Base directory for relative paths in CSV (required if csv_file is provided)
        split: Data split for CSV ('train', 'val', or 'test')
        model_dir: Model directory for saving filtered images log (optional)
        mri_classifications: List of MRI classifications to include (e.g., ['T1', 'T2', 'FLAIR'])
                           Only applicable when using csv_file
        acquisition_types: List of acquisition types to include (default: ['3D'])
                          Set to None to include all types. Only applicable when using csv_file
        filter_4d: If True, filters out 4D images (default: True)
                  Only applicable when using csv_file

    Returns:
        List of image paths
    """
    if csv_file is not None:
        if base_dir is None:
            raise ValueError("--base_dir is required when using --csv_file")

        # Set up log path for filtered 4D images if model_dir is provided
        log_filtered_path = None
        if model_dir is not None and filter_4d:
            log_filtered_path = os.path.join(
                model_dir, f"filtered_4d_images_{split}.csv"
            )

        return load_image_paths_from_csv(
            csv_file,
            base_dir,
            split=split,
            acquisition_types=acquisition_types,
            mri_classifications=mri_classifications,
            filter_4d=filter_4d,
            log_filtered_path=log_filtered_path,
        )
    elif image_dir is not None:
        # Get all .nii.gz files from directory
        image_dir = Path(image_dir)
        return sorted([str(p) for p in image_dir.glob("*.nii.gz")])
    else:
        raise ValueError("Either --hr_image_dir or --csv_file must be provided")


# =============================================================================
# Model Checkpoint Management
# =============================================================================

def find_latest_checkpoint(model_dir, model_type="uhved"):
    """Find the latest checkpoint in the model directory."""
    checkpoints = glob.glob(os.path.join(model_dir, f"{model_type}_*_epoch_*.pth"))
    if not checkpoints:
        return None

    # Sort by epoch number
    def get_epoch(path):
        try:
            return int(path.split('epoch_')[-1].split('.pth')[0])
        except:
            return -1

    checkpoints.sort(key=get_epoch)
    return checkpoints[-1] if checkpoints else None


def save_model_checkpoint(filepath, model, optimizer, epoch, loss, val_loss=None,
                          model_type="uhved", model_config=None, scheduler_state_dict=None,
                          val_metrics=None, training_config=None):
    """
    Save model checkpoint with complete training state.

    Args:
        filepath: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Training loss
        val_loss: Validation loss (optional)
        model_type: Type of model (default: "uhved")
        model_config: Model architecture configuration
        scheduler_state_dict: Learning rate scheduler state (optional)
        val_metrics: Dictionary of validation metrics (optional)
        training_config: Full training configuration for reproducibility (optional)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'val_loss': val_loss,
        'model_type': model_type,
        'model_config': model_config or {},
    }

    if scheduler_state_dict:
        checkpoint['scheduler_state_dict'] = scheduler_state_dict

    if val_metrics:
        checkpoint['val_metrics'] = val_metrics

    if training_config:
        checkpoint['training_config'] = training_config

    torch.save(checkpoint, filepath)

def save_training_config(model_dir, args, n_train_samples, n_val_samples, training_stage):
    """Save training configuration to JSON."""
    import json

    config = {
        # Training stage and architecture
        'training_stage': training_stage,
        'model_architecture': 'uhved',
        'num_orientations': 3,  # Fixed: axial, coronal, sagittal

        # Dataset info
        'n_train_samples': n_train_samples,
        'n_val_samples': n_val_samples,

        # Training parameters
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'seed': args.seed,

        # Model architecture
        'base_channels': args.base_channels,
        'num_scales': args.num_scales,
        'final_activation': args.final_activation,
        'use_prior': args.use_prior,
        'use_encoder_outputs_as_skip': args.use_encoder_outputs_as_skip,
        'decoder_upsample_mode': args.decoder_upsample_mode,

        # Output configuration
        'output_shape': args.output_shape,
        'atlas_res': args.atlas_res,

        # Resolution parameters
        'min_resolution': args.min_resolution,
        'max_res_aniso': args.max_res_aniso,
        'randomise_res': not args.no_randomise_res,

        # Loss configuration
        'recon_loss_type': args.recon_loss_type,
        'recon_weight': args.recon_weight,
        'kl_weight': args.kl_weight,
        'perceptual_weight': args.perceptual_weight,
        'ssim_weight': args.ssim_weight,
        'orientation_weight': args.orientation_weight,
        'use_perceptual': args.use_perceptual,
        'use_ssim': args.use_ssim,
        'perceptual_backend': args.perceptual_backend,

        # Artifact probabilities
        'prob_motion': args.prob_motion,
        'prob_spike': args.prob_spike,
        'prob_aliasing': args.prob_aliasing,
        'prob_bias_field': args.prob_bias_field,
        'prob_noise': args.prob_noise,
        'apply_intensity_aug': not args.no_intensity_aug,
        'orientation_dropout_prob': args.orientation_dropout_prob,
        'min_orientations': args.min_orientations,
        'drop_orientations': args.drop_orientations,

        # Training optimization
        'mixed_precision': args.mixed_precision,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'max_grad_norm': args.max_grad_norm,

        # Checkpointing
        'save_interval': args.save_interval,
        'val_interval': args.val_interval,

        # Validation configuration
        'use_sliding_window_val': args.use_sliding_window_val,
        'val_patch_size': args.val_patch_size,
        'val_overlap': args.val_overlap,
    }

    config_path = os.path.join(model_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Training configuration saved to: {config_path}")

# =============================================================================
# Evaluation Metrics
# =============================================================================

def calculate_metrics(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0):
    """
    Calculate evaluation metrics.

    Args:
        pred: Predicted images (B, C, D, H, W)
        target: Target images (B, C, D, H, W)
        max_val: Maximum pixel value for PSNR calculation (default: 1.0)

    Returns:
        Dictionary of metrics
    """
    with torch.no_grad():
        # MAE (L1)
        mae = torch.abs(pred - target).mean().item()

        # MSE (L2)
        mse = ((pred - target) ** 2).mean().item()

        # RMSE
        rmse = torch.sqrt(torch.tensor(mse)).item()

        # PSNR (Peak Signal-to-Noise Ratio)
        if mse > 0:
            psnr = 10 * torch.log10(torch.tensor(max_val**2 / mse)).item()
        else:
            psnr = float("inf")

        # R² (Coefficient of Determination)
        target_mean = target.mean()
        ss_tot = ((target - target_mean) ** 2).sum().item()
        ss_res = ((target - pred) ** 2).sum().item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # SSIM (Structural Similarity Index Measure)
        # MONAI's SSIMLoss returns 1 - SSIM, so we need to invert it
        # SSIM ranges from -1 to 1, where 1 is perfect similarity
        ssim_loss_fn = MonaiSSIM(spatial_dims=3, data_range=max_val)
        ssim_loss = ssim_loss_fn(pred, target).item()
        ssim = 1 - ssim_loss  # Convert from loss to similarity

        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "psnr": psnr,
            "r2": r2,
            "ssim": ssim,
        }


def calculate_lpips_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    backend: str = 'monai',
    feature_layers: List[str] = None,
    device: str = 'cuda'
) -> float:
    """
    Calculate 3D-LPIPS (Learned Perceptual Image Patch Similarity) for volumetric data.

    Uses proper 3D medical imaging networks instead of 2D slice-by-slice processing.
    Leverages PerceptualLoss3D infrastructure with MedicalNet, MONAI, or Models Genesis.

    Args:
        pred: Predicted volume (B, C, D, H, W)
        target: Target volume (B, C, D, H, W)
        backend: 3D network backend ('medicalnet', 'monai', 'models_genesis')
        feature_layers: Layers to extract features from
        device: Device for computation

    Returns:
        3D-LPIPS score (lower is better, 0 = identical)

    References:
        - Zhang et al. "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"
        - Adapted for 3D medical imaging volumes
    """
    try:
        from src.losses import PerceptualLoss3D
    except ImportError:
        raise ImportError("PerceptualLoss3D not found in src.losses")

    # Initialize 3D perceptual loss model (cached after first call)
    cache_key = f'lpips_3d_{backend}'
    if not hasattr(calculate_lpips_3d, cache_key):
        if feature_layers is None:
            feature_layers = ['layer1', 'layer2', 'layer3', 'layer4']

        perceptual_loss = PerceptualLoss3D(
            backend=backend,
            model_depth=18,  # ResNet-18 for efficiency
            feature_layers=feature_layers,
            weights=[1.0] * len(feature_layers),
            pretrained=False,  # We use random init for metric, not pre-trained
            normalize_input=True,
            freeze_backbone=False  # Allow gradient flow for feature extraction
        ).to(device)
        perceptual_loss.eval()
        setattr(calculate_lpips_3d, cache_key, perceptual_loss)

    perceptual_loss = getattr(calculate_lpips_3d, cache_key)

    with torch.no_grad():
        # Move to device
        pred = pred.to(device)
        target = target.to(device)

        # Extract features from both volumes
        pred_features = perceptual_loss.extract_features(pred)
        target_features = perceptual_loss.extract_features(target)

        # Compute L2 distance in feature space (LPIPS formula)
        lpips_score = 0.0
        num_layers = 0

        for layer_name in feature_layers:
            pred_feat = None
            target_feat = None

            # Find matching features
            for name, feat in pred_features.items():
                if layer_name in name:
                    pred_feat = feat
                    break

            for name, feat in target_features.items():
                if layer_name in name:
                    target_feat = feat
                    break

            if pred_feat is not None and target_feat is not None:
                # Spatial L2 normalization (LPIPS style)
                diff = (pred_feat - target_feat) ** 2
                # Average across spatial dimensions and channels
                layer_dist = diff.mean(dim=[1, 2, 3, 4]).mean()
                lpips_score += layer_dist.item()
                num_layers += 1

        # Average across layers
        if num_layers > 0:
            lpips_score /= num_layers

    return lpips_score


def calculate_rlpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    backend: str = 'monai',
    adversarial_eps: float = 0.03,
    device: str = 'cuda'
) -> float:
    """
    Calculate R-LPIPS (Robust LPIPS) using adversarially trained features.

    R-LPIPS was proposed to be robust against adversarial perturbations by using
    features from adversarially trained networks instead of standard pre-trained ones.

    Reference:
        Ghazanfari et al. "R-LPIPS: An Adversarially Robust Perceptual Similarity Metric"
        arXiv:2307.15157 (2023)
        https://arxiv.org/abs/2307.15157

    Args:
        pred: Predicted volume (B, C, D, H, W)
        target: Target volume (B, C, D, H, W)
        backend: 3D network backend ('medicalnet', 'monai', 'models_genesis')
        adversarial_eps: Epsilon for adversarial robustness (default: 0.03)
        device: Device for computation

    Returns:
        R-LPIPS score (robust perceptual distance)

    Note:
        This implementation uses a simulated robust feature extractor.
        For true R-LPIPS, networks should be adversarially trained (e.g., using PGD).
        In medical imaging, adversarially trained 3D networks are rare, so this
        provides an approximation using ensemble-based robustness.
    """
    try:
        from src.losses import PerceptualLoss3D
    except ImportError:
        raise ImportError("PerceptualLoss3D not found in src.losses")

    # Initialize robust perceptual loss model (cached after first call)
    cache_key = f'rlpips_3d_{backend}'
    if not hasattr(calculate_rlpips, cache_key):
        feature_layers = ['layer1', 'layer2', 'layer3', 'layer4']

        perceptual_loss = PerceptualLoss3D(
            backend=backend,
            model_depth=18,
            feature_layers=feature_layers,
            weights=[1.0] * len(feature_layers),
            pretrained=False,
            normalize_input=True,
            freeze_backbone=False
        ).to(device)
        perceptual_loss.eval()
        setattr(calculate_rlpips, cache_key, perceptual_loss)

    perceptual_loss = getattr(calculate_rlpips, cache_key)

    with torch.no_grad():
        # Move to device
        pred = pred.to(device)
        target = target.to(device)

        # R-LPIPS: Ensemble-based robustness approximation
        # Since we don't have access to adversarially trained 3D medical networks,
        # we use an ensemble of perturbations to simulate robustness
        # This is inspired by the E-LPIPS (Ensemble LPIPS) approach

        # Base features
        pred_features_base = perceptual_loss.extract_features(pred)
        target_features_base = perceptual_loss.extract_features(target)

        # Add small random perturbations to simulate adversarial robustness
        num_perturbations = 3
        all_scores = []

        for i in range(num_perturbations):
            # Random perturbation within epsilon
            if i == 0:
                # Use original (no perturbation)
                pred_pert = pred
                target_pert = target
            else:
                # Add small Gaussian noise (simulating adversarial robustness)
                noise_scale = adversarial_eps * (i / num_perturbations)
                pred_noise = torch.randn_like(pred) * noise_scale
                target_noise = torch.randn_like(target) * noise_scale

                pred_pert = torch.clamp(pred + pred_noise, 0, 1)
                target_pert = torch.clamp(target + target_noise, 0, 1)

            # Extract features with perturbation
            pred_features = perceptual_loss.extract_features(pred_pert)
            target_features = perceptual_loss.extract_features(target_pert)

            # Compute perceptual distance
            score = 0.0
            num_layers = 0

            for layer_name in perceptual_loss.feature_layers:
                pred_feat = None
                target_feat = None

                for name, feat in pred_features.items():
                    if layer_name in name:
                        pred_feat = feat
                        break

                for name, feat in target_features.items():
                    if layer_name in name:
                        target_feat = feat
                        break

                if pred_feat is not None and target_feat is not None:
                    diff = (pred_feat - target_feat) ** 2
                    layer_dist = diff.mean(dim=[1, 2, 3, 4]).mean()
                    score += layer_dist.item()
                    num_layers += 1

            if num_layers > 0:
                score /= num_layers

            all_scores.append(score)

        # R-LPIPS: Robust estimate using median of ensemble
        # (median is more robust to outliers than mean)
        rlpips_score = float(np.median(all_scores))

    return rlpips_score


def calculate_metrics_with_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
    compute_lpips: bool = True,
    lpips_backend: str = 'monai',
    device: str = 'cuda'
):
    """
    Calculate comprehensive evaluation metrics including 3D-LPIPS and R-LPIPS.

    Args:
        pred: Predicted volumes (B, C, D, H, W)
        target: Target volumes (B, C, D, H, W)
        max_val: Maximum pixel value for PSNR calculation (default: 1.0)
        compute_lpips: Whether to compute LPIPS metrics
        lpips_backend: 3D network backend ('monai', 'medicalnet', 'models_genesis')
        device: Device for computation

    Returns:
        Dictionary of metrics including standard metrics, 3D-LPIPS, and R-LPIPS

    References:
        - Zhang et al. "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (CVPR 2018)
        - Ghazanfari et al. "R-LPIPS: An Adversarially Robust Perceptual Similarity Metric" (arXiv 2023)
    """
    # Get standard metrics
    metrics = calculate_metrics(pred, target, max_val=max_val)

    # Add 3D-LPIPS and R-LPIPS metrics if requested
    if compute_lpips:
        try:
            with torch.no_grad():
                # 3D-LPIPS using proper 3D encoder
                lpips_3d = calculate_lpips_3d(
                    pred, target, backend=lpips_backend, device=device
                )
                metrics['lpips_3d'] = lpips_3d

                # R-LPIPS (Robust LPIPS)
                rlpips = calculate_rlpips(
                    pred, target, backend=lpips_backend, device=device
                )
                metrics['rlpips'] = rlpips

        except ImportError as e:
            import warnings
            warnings.warn(f"3D-LPIPS calculation skipped: {str(e)}")
        except Exception as e:
            import warnings
            warnings.warn(f"Error computing 3D-LPIPS: {str(e)}")

    return metrics


# =============================================================================
# Sliding Window Inference
# =============================================================================

def sliding_window_inference(
    model: nn.Module,
    orientations: list,
    patch_size: tuple = (128, 128, 128),
    overlap: float = 0.5,
    batch_size: int = 1,
    device: str = "cuda",
    blend_mode: str = "gaussian",
    progress: bool = False,
    orientation_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Perform sliding window inference for U-HVED with multiple input orientations.

    Args:
        model: U-HVED model for inference (should be in eval mode)
        orientations: List of 3 input volumes (orthogonal stacks), each (1, C, D, H, W) or (C, D, H, W)
        patch_size: Size of patches for inference (D, H, W)
        overlap: Overlap ratio between adjacent patches [0, 1)
        batch_size: Number of patches to process simultaneously (typically 1 for U-HVED)
        device: Device for inference ('cuda' or 'cpu')
        blend_mode: Method for blending overlapping patches ('gaussian' or 'constant')
        progress: If True, show progress bar
        orientation_mask: Optional boolean tensor (B, 3) indicating which orientations are present

    Returns:
        Predicted SR volume of same shape as input orientations (1, C, D, H, W) or (C, D, H, W)

    Example:
        >>> model.eval()
        >>> lr_axial = torch.randn(1, 1, 128, 128, 128).cuda()
        >>> lr_coronal = torch.randn(1, 1, 128, 128, 128).cuda()
        >>> lr_sagittal = torch.randn(1, 1, 128, 128, 128).cuda()
        >>> orientations = [lr_axial, lr_coronal, lr_sagittal]
        >>> sr_output = sliding_window_inference(
        ...     model=model,
        ...     orientations=orientations,
        ...     patch_size=(128, 128, 128),
        ...     overlap=0.5,
        ...     device="cuda"
        ... )
    """
    from monai.inferers import sliding_window_inference as monai_sliding_window

    # Ensure model is in eval mode
    model.eval()

    # Validate inputs
    if len(orientations) != 3:
        raise ValueError(f"U-HVED requires exactly 3 input orientations, got {len(orientations)}")

    # Handle input shapes and normalize
    normalized_orientations = []
    squeeze_output = False

    for i, volume in enumerate(orientations):
        if volume.ndim == 4:
            volume = volume.unsqueeze(0)  # Add batch dimension
            if i == 0:  # Only set flag once
                squeeze_output = True

        batch, channels, d, h, w = volume.shape
        if batch != 1:
            raise ValueError(f"Batch size must be 1, got {batch}")

        # Move to device
        volume = volume.to(device)
        normalized_orientations.append(volume)

    # Create a wrapper predictor that takes a single stacked input
    # and splits it for the U-HVED model
    def uhved_predictor(stacked_input):
        """Predictor that splits stacked input into orientations for U-HVED."""
        # Split the stacked input back into 3 orientations
        batch_size = stacked_input.shape[0]
        channels = stacked_input.shape[1] // 3

        mod1 = stacked_input[:, :channels]
        mod2 = stacked_input[:, channels:channels*2]
        mod3 = stacked_input[:, channels*2:]

        # Create list of orientations for each batch item
        batch_orientations = []
        for b in range(batch_size):
            batch_orientations.append([
                mod1[b:b+1],
                mod2[b:b+1],
                mod3[b:b+1]
            ])

        # Process each item in batch
        outputs = []
        for mods in batch_orientations:
            with torch.no_grad():
                result = model(mods, orientation_mask=orientation_mask)
                if isinstance(result, dict):
                    outputs.append(result['sr_output'])
                else:
                    outputs.append(result)

        return torch.cat(outputs, dim=0)

    # Stack orientations along channel dimension for sliding window
    # This allows MONAI to handle the patching uniformly
    stacked_input = torch.cat(normalized_orientations, dim=1)  # (1, 3*C, D, H, W)

    # Use MONAI's sliding window inference
    with torch.no_grad():
        output = monai_sliding_window(
            inputs=stacked_input,
            roi_size=patch_size,
            sw_batch_size=batch_size,
            predictor=uhved_predictor,
            overlap=overlap,
            mode=blend_mode,
            progress=progress,
        )

    # Remove batch dimension if input didn't have it
    if squeeze_output:
        output = output.squeeze(0)

    return output


def predict_full_volume(
    model: nn.Module,
    input_path: str,
    output_path: str,
    patch_size: tuple = (128, 128, 64),
    overlap: float = 0.5,
    batch_size: int = 4,
    device: str = "cuda",
    target_spacing: Optional[list] = None,
) -> None:
    """
    Predict on a full NIfTI volume using sliding window inference and save result.

    High-level wrapper that loads an input NIfTI file, applies padding if needed,
    runs sliding window inference, and saves the prediction as a NIfTI file with
    the same metadata (affine, header) as the input.

    Args:
        model: Trained PyTorch model (will be set to eval mode)
        input_path: Path to input NIfTI file (.nii or .nii.gz)
        output_path: Path to save output NIfTI file (.nii or .nii.gz)
        patch_size: Size of patches for inference (D, H, W)
        overlap: Overlap ratio between patches [0, 1)
        batch_size: Number of patches to process simultaneously
        device: Device for inference ('cuda' or 'cpu')
        target_spacing: Optional target voxel spacing [x, y, z] in mm
            If provided, input will be resampled before inference

    Example:
        >>> from src.utils import load_unet3d_from_checkpoint, predict_full_volume
        >>> model, _ = load_unet3d_from_checkpoint("model.pth", device="cuda")
        >>>
        >>> predict_full_volume(
        ...     model=model,
        ...     input_path="input_lr.nii.gz",
        ...     output_path="output_sr.nii.gz",
        ...     patch_size=(128, 128, 64),
        ...     overlap=0.5,
        ...     batch_size=4,
        ...     device="cuda"
        ... )
    """
    import nibabel as nib
    from monai.transforms import (
        LoadImage,
        EnsureChannelFirst,
        Orientation,
        Spacing,
        Compose,
    )

    print(f"Loading input volume: {input_path}")

    # Build preprocessing pipeline
    transforms = [
        LoadImage(image_only=False),  # Keep metadata
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
    ]

    if target_spacing is not None:
        transforms.append(Spacing(pixdim=target_spacing, mode="bilinear"))

    transform = Compose(transforms)

    # Load and preprocess
    img_obj = transform(input_path)
    volume = img_obj[0]  # Get tensor
    metadata = img_obj[1]  # Get metadata

    # Ensure shape is (C, D, H, W)
    if volume.ndim == 3:
        volume = volume.unsqueeze(0)

    print(f"Input volume shape: {volume.shape}")

    # Normalize to [0, 1] for consistency with training
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

    # Pad to ensure divisibility by patch size if needed
    original_shape = volume.shape[1:]  # (D, H, W)
    padded_volume, pad_before, orig_shape = pad_to_multiple_of_32(volume.squeeze(0).numpy())
    padded_volume = torch.from_numpy(padded_volume).unsqueeze(0).float()

    print(f"Padded volume shape: {padded_volume.shape}")

    # Run sliding window inference using MONAI
    print(f"Running sliding window inference...")
    prediction = sliding_window_inference(
        model=model,
        volume=padded_volume,
        patch_size=patch_size,
        overlap=overlap,
        batch_size=batch_size,
        device=device,
        blend_mode="gaussian",
        progress=True,
    )

    # Remove padding
    prediction = prediction.squeeze(0).cpu().numpy()  # (C, D, H, W) -> (D, H, W)
    if prediction.ndim == 4:
        prediction = prediction.squeeze(0)  # Remove channel dim
    prediction = unpad_volume(prediction, pad_before, orig_shape)

    print(f"Output volume shape: {prediction.shape}")

    # Save as NIfTI with original metadata
    affine = metadata.get("affine", np.eye(4))
    output_nii = nib.Nifti1Image(prediction, affine=affine)

    # Copy header metadata if available
    if "original_affine" in metadata:
        output_nii.header["pixdim"] = metadata.get("pixdim", output_nii.header["pixdim"])

    nib.save(output_nii, output_path)
    print(f"Saved prediction to: {output_path}")