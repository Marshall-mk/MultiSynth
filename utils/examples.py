"""
Example usage and interactive tools for NIfTI visualization
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from pathlib import Path
from nifti_visualizer import extract_slice, process_nifti, batch_process, process_nifti_all_axes, batch_process_all_axes


def find_slice_interactive(nifti_path, axis='sagittal'):
    """
    Interactively browse through slices to find the right one.
    Use left/right arrow keys to navigate, close window when done.
    """
    img = nib.load(nifti_path)
    data = img.get_fdata()
    
    axis_idx = {'sagittal': 0, 'coronal': 1, 'axial': 2}[axis]
    n_slices = data.shape[axis_idx]
    
    current_slice = [n_slices // 2]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def update_slice():
        ax.clear()
        slice_2d = extract_slice(data, axis=axis, slice_idx=current_slice[0])
        ax.imshow(slice_2d.T, cmap='gray', origin='lower')
        ax.set_title(f'{axis.capitalize()} Slice: {current_slice[0]}/{n_slices-1}')
        ax.axis('off')
        fig.canvas.draw()
    
    def on_key(event):
        if event.key == 'right':
            current_slice[0] = min(current_slice[0] + 1, n_slices - 1)
            update_slice()
        elif event.key == 'left':
            current_slice[0] = max(current_slice[0] - 1, 0)
            update_slice()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    update_slice()
    plt.show()
    
    print(f"Selected slice: {current_slice[0]}")
    return current_slice[0]


def find_bbox_interactive(nifti_path, slice_idx, axis='sagittal'):
    """
    Interactively select bounding box on a slice.
    Click and drag to select region, close window when done.
    """
    img = nib.load(nifti_path)
    data = img.get_fdata()
    slice_2d = extract_slice(data, axis=axis, slice_idx=slice_idx)
    
    bbox = [None]
    
    def on_select(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        
        bbox[0] = (x, y, w, h)
        print(f"Selected bbox: {bbox[0]}")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(slice_2d.T, cmap='gray', origin='lower')
    ax.set_title(f'Click and drag to select region - {axis.capitalize()} slice {slice_idx}')
    
    selector = RectangleSelector(
        ax, on_select,
        useblit=True,
        button=[1],
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=True
    )
    
    plt.show()
    
    if bbox[0] is not None:
        print(f"Bounding box: {bbox[0]}")
        return bbox[0]
    else:
        print("No bounding box selected")
        return None


def find_parameters_all_axes(nifti_path):
    """
    Interactively find slice indices and bounding boxes for all three axes.
    Returns: (slice_indices dict, bboxes dict)
    """
    slice_indices = {}
    bboxes = {}
    
    for axis in ['sagittal', 'coronal', 'axial']:
        print(f"\n=== Finding parameters for {axis.upper()} axis ===")
        
        # Find slice
        slice_idx = find_slice_interactive(nifti_path, axis=axis)
        slice_indices[axis] = slice_idx
        
        # Find bbox
        bbox = find_bbox_interactive(nifti_path, slice_idx=slice_idx, axis=axis)
        bboxes[axis] = bbox
    
    print("\n=== Final Parameters ===")
    print(f"Slice indices: {slice_indices}")
    print(f"Bounding boxes: {bboxes}")
    
    return slice_indices, bboxes


def example_single_file_all_axes():
    """Process a single file across all axes"""
    slice_indices = {
        'sagittal': 120,
        'coronal': 140,
        'axial': 100
    }
    
    bboxes = {
        'sagittal': (59, 82, 93, 52),
        'coronal': (50, 80, 100, 120),
        'axial': (60, 70, 90, 110)
    }
    
    process_nifti_all_axes(
        nifti_path="example.nii.gz",
        output_dir="output",
        slice_indices=slice_indices,
        bboxes=bboxes,
        prefix='brain'
    )


def example_batch_all_axes():
    """Process all files in a directory across all axes"""
    slice_indices = {
        'sagittal': 120,
        'coronal': 140,
        'axial': 100
    }
    
    bboxes = {
        'sagittal': (59, 82, 93, 52),
        'coronal': (50, 80, 100, 120),
        'axial': (60, 70, 90, 110)
    }
    
    batch_process_all_axes(
        input_dir="nifti_files",
        output_dir="output",
        slice_indices=slice_indices,
        bboxes=bboxes,
        pattern='*.nii.gz'
    )


if __name__ == "__main__":
    # Example 1: Find parameters for all axes interactively
    # slice_indices, bboxes = find_parameters_all_axes("/Users/cleancoder/Downloads/IXI049-HH-1358-T2/axial_upsampled.nii.gz")
    
    # Example 2: Process single file across all axes
    # example_single_file_all_axes()
    
    # Example 3: Batch process all files across all axes
    example_batch_all_axes()
    
    print("\nUncomment the examples you want to run")
