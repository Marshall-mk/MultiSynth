"""
NIfTI Slice Extractor with Bounding Box Visualization
Extracts specified slices and crops from NIfTI files for paper figures.
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path


def extract_slice(nifti_data, axis='sagittal', slice_idx=None):
    """
    Extract a 2D slice from 3D NIfTI data.
    
    Args:
        nifti_data: 3D numpy array
        axis: 'sagittal', 'coronal', or 'axial'
        slice_idx: slice index (default: middle slice)
    
    Returns:
        2D numpy array
    """
    if slice_idx is None:
        slice_idx = nifti_data.shape[{'sagittal': 0, 'coronal': 1, 'axial': 2}[axis]] // 2
    
    if axis == 'sagittal':
        return nifti_data[slice_idx, :, :]
    elif axis == 'coronal':
        return nifti_data[:, slice_idx, :]
    elif axis == 'axial':
        return nifti_data[:, :, slice_idx]
    else:
        raise ValueError("axis must be 'sagittal', 'coronal', or 'axial'")


def crop_region(slice_2d, bbox):
    """
    Crop a region from a 2D slice.
    
    Args:
        slice_2d: 2D numpy array
        bbox: tuple (x, y, width, height) in display coordinates
    
    Returns:
        Cropped 2D numpy array
    """
    x, y, w, h = bbox
    # In display coordinates: x=horizontal, y=vertical
    # In array indexing: [rows, cols] = [y, x]
    return slice_2d.T[y:y+h, x:x+w]


def save_slice_with_bbox(slice_2d, bbox, output_path, dpi=150, cmap='gray'):
    """
    Save slice with bounding box overlay.
    
    Args:
        slice_2d: 2D numpy array
        bbox: tuple (x, y, width, height) or None
        output_path: output file path
        dpi: resolution (default: 150)
        cmap: colormap (default: 'gray')
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(slice_2d.T, cmap=cmap, origin='lower')
    
    if bbox is not None:
        x, y, w, h = bbox
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_slice(slice_2d, output_path, dpi=150, cmap='gray', transpose=True):
    """
    Save slice without bounding box.
    
    Args:
        slice_2d: 2D numpy array
        output_path: output file path
        dpi: resolution (default: 150)
        cmap: colormap (default: 'gray')
        transpose: whether to transpose the slice (default: True)
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    img_data = slice_2d.T if transpose else slice_2d
    ax.imshow(img_data, cmap=cmap, origin='lower')
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()


def process_nifti(nifti_path, output_dir, slice_idx, bbox, axis='sagittal', prefix=''):
    """
    Process a single NIfTI file: extract slice, crop, and save visualizations.
    
    Args:
        nifti_path: path to NIfTI file
        output_dir: output directory
        slice_idx: slice index to extract
        bbox: bounding box (x, y, width, height) for crop
        axis: 'sagittal', 'coronal', or 'axial'
        prefix: filename prefix
    """
    # Load NIfTI
    img = nib.load(nifti_path)
    data = img.get_fdata()
    
    # Extract slice
    slice_2d = extract_slice(data, axis=axis, slice_idx=slice_idx)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save slice with bounding box
    save_slice_with_bbox(
        slice_2d, 
        bbox, 
        output_dir / f'{prefix}_slice_with_bbox.png'
    )
    
    # Save cropped region (already transposed by crop_region)
    cropped = crop_region(slice_2d, bbox)
    save_slice(
        cropped,
        output_dir / f'{prefix}_crop.png',
        transpose=False
    )
    
    print(f"Processed: {nifti_path.name}")
    print(f"  - Saved slice with bbox: {prefix}_slice_with_bbox.png")
    print(f"  - Saved crop: {prefix}_crop.png")


def process_nifti_all_axes(nifti_path, output_dir, slice_indices, bboxes, prefix=''):
    """
    Process a single NIfTI file across all three axes.
    
    Args:
        nifti_path: path to NIfTI file
        output_dir: output directory
        slice_indices: dict with keys 'sagittal', 'coronal', 'axial' and slice index values
        bboxes: dict with keys 'sagittal', 'coronal', 'axial' and bbox tuples (x, y, w, h)
        prefix: filename prefix
    """
    img = nib.load(nifti_path)
    data = img.get_fdata()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for axis in ['sagittal', 'coronal', 'axial']:
        slice_idx = slice_indices.get(axis)
        bbox = bboxes.get(axis)
        
        # Extract slice
        slice_2d = extract_slice(data, axis=axis, slice_idx=slice_idx)
        
        # Save slice with bounding box
        save_slice_with_bbox(
            slice_2d,
            bbox,
            output_dir / f'{prefix}_{axis}_slice_with_bbox.png'
        )
        
        # Save cropped region
        if bbox is not None:
            cropped = crop_region(slice_2d, bbox)
            save_slice(
                cropped,
                output_dir / f'{prefix}_{axis}_crop.png',
                transpose=False
            )
    
    print(f"Processed {nifti_path.name}: sagittal, coronal, axial")


def batch_process(input_dir, output_dir, slice_idx, bbox, axis='sagittal', pattern='*.nii.gz'):
    """
    Process all NIfTI files in a directory.
    
    Args:
        input_dir: directory containing NIfTI files
        output_dir: output directory
        slice_idx: slice index to extract
        bbox: bounding box (x, y, width, height)
        axis: 'sagittal', 'coronal', or 'axial'
        pattern: file pattern to match
    """
    input_dir = Path(input_dir)
    nifti_files = sorted(input_dir.glob(pattern))
    
    if not nifti_files:
        print(f"No files matching '{pattern}' found in {input_dir}")
        return
    
    for nifti_file in nifti_files:
        prefix = nifti_file.stem.replace('.nii', '')
        process_nifti(nifti_file, output_dir, slice_idx, bbox, axis, prefix)
    
    print(f"\nProcessed {len(nifti_files)} files")


def batch_process_all_axes(input_dir, output_dir, slice_indices, bboxes, pattern='*.nii.gz'):
    """
    Process all NIfTI files across all three axes.
    
    Args:
        input_dir: directory containing NIfTI files
        output_dir: output directory
        slice_indices: dict {'sagittal': idx, 'coronal': idx, 'axial': idx}
        bboxes: dict {'sagittal': (x,y,w,h), 'coronal': (x,y,w,h), 'axial': (x,y,w,h)}
        pattern: file pattern to match
    """
    input_dir = Path(input_dir)
    nifti_files = sorted(input_dir.glob(pattern))
    
    if not nifti_files:
        print(f"No files matching '{pattern}' found in {input_dir}")
        return
    
    for nifti_file in nifti_files:
        prefix = nifti_file.stem.replace('.nii', '')
        process_nifti_all_axes(nifti_file, output_dir, slice_indices, bboxes, prefix)
    
    print(f"\nProcessed {len(nifti_files)} files across all axes")


# Example usage
if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "/Users/cleancoder/Downloads/IXI049-HH-1358-T2"
    OUTPUT_DIR = "all_axes_output"
    
    # Define slice indices for each axis
    SLICE_INDICES = {
        'sagittal': 120,
        'coronal': 140,
        'axial': 100
    }
    
    # Define bounding boxes for each axis
    BBOXES = {
        'sagittal': (59, 82, 93, 52),
        'coronal': (50, 80, 100, 120),
        'axial': (60, 70, 90, 110)
    }
    
    # Process all files across all axes
    batch_process_all_axes(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        slice_indices=SLICE_INDICES,
        bboxes=BBOXES,
        pattern='*.nii.gz'
    )
