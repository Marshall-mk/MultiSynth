#!/usr/bin/env python3
"""
Minimal pipeline for MRI preprocessing: resample → register → crop overlap.
"""
import argparse
import os
import tempfile
from pathlib import Path

import ants
import nibabel as nib
import numpy as np
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, Spacing, Orientation
import warnings
warnings.filterwarnings("ignore")

# ========== RESAMPLING ==========

def resample(path: str, resolution: list, orientation: str = "RAS", interp: str = "trilinear"):
    """Resample NIfTI to target resolution and orientation."""
    transform = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes=orientation),
        Spacing(pixdim=resolution, mode=interp),
    ])
    data = transform(path)
    affine = data.affine.cpu().numpy() if hasattr(data, 'affine') else nib.load(path).affine
    return data.squeeze(0).cpu().numpy(), affine, nib.load(path).header


def save_nifti(data: np.ndarray, affine: np.ndarray, header, path: str):
    """Save array as NIfTI."""
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine, header), path)


# ========== REGISTRATION ==========

def register_to_fixed(fixed_path: str, moving_path: str, transform_type: str = "Rigid", metric: str = "gc") -> str:
    """Register moving image to fixed using ANTs. Returns path to registered image."""
    fixed = ants.image_read(fixed_path)
    moving = ants.image_read(moving_path)
    
    result = ants.registration(fixed=fixed, moving=moving, type_of_transform=transform_type, aff_metric=metric)
    
    out_path = moving_path.replace('.nii', '_registered.nii')
    result['warpedmovout'].to_filename(out_path)
    return out_path


# ========== CROPPING ==========

def get_nonzero_bbox(data: np.ndarray):
    """Get bounding box of non-zero region."""
    nz = np.where(data != 0)
    return [(nz[i].min(), nz[i].max() + 1) for i in range(3)]


def find_overlap_bbox(images: list):
    """Find intersection bounding box across all images."""
    bboxes = [get_nonzero_bbox(nib.load(p).get_fdata()) for p in images]
    overlap = []
    for dim in range(3):
        start = max(bb[dim][0] for bb in bboxes)
        end = min(bb[dim][1] for bb in bboxes)
        if start >= end:
            raise ValueError(f"No overlap in dimension {dim}")
        overlap.append((start, end))
    return overlap


def crop_to_bbox(path: str, bbox: list, out_path: str):
    """Crop image to bounding box and save."""
    img = nib.load(path)
    data = img.get_fdata()[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
    
    # Update affine origin
    affine = img.affine.copy()
    offset = np.array([bbox[0][0], bbox[1][0], bbox[2][0], 0])
    affine[:, 3] += affine @ offset
    
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine, img.header), out_path)


# ========== PIPELINE ==========

def run_pipeline(
    inputs: list,
    output_dir: str,
    hr_input: str = None,
    do_resample: bool = True,
    do_register: bool = True,
    do_crop: bool = True,
    save_registered: bool = False,
    resolution: list = None,
    orientation: str = "RAS",
    interp: str = "trilinear",
    fixed_idx: int = 0,
    transform_type: str = "Rigid",
    metric: str = "gc",
    verbose: bool = False,
):
    """Run preprocessing pipeline with selectable stages."""
    resolution = resolution or [1.0, 1.0, 1.0]
    os.makedirs(output_dir, exist_ok=True)
    
    # Track current working paths
    current = list(inputs)
    temp_dir = tempfile.mkdtemp()
    hr_current = None
    
    # Stage 1: Resample
    if do_resample:
        if verbose:
            print("=== Resampling LR stacks ===")
        resampled = []
        for i, path in enumerate(current):
            if verbose:
                print(f"  {Path(path).name}")
            data, affine, header = resample(path, resolution, orientation, interp)
            out = os.path.join(temp_dir, f"resampled_{i}.nii.gz")
            save_nifti(data, affine, header, out)
            resampled.append(out)
        current = resampled
        
        # Resample HR if provided
        if hr_input:
            if verbose:
                print(f"=== Resampling HR scan ===")
                print(f"  {Path(hr_input).name}")
            data, affine, header = resample(hr_input, resolution, orientation, interp)
            hr_current = os.path.join(temp_dir, "resampled_hr.nii.gz")
            save_nifti(data, affine, header, hr_current)
    elif hr_input:
        hr_current = hr_input
    
    # Stage 2: Register
    if do_register:
        if verbose:
            print(f"=== Registering LR stacks to image {fixed_idx} ===")
        fixed_path = current[fixed_idx]
        registered = []
        for i, path in enumerate(current):
            if i == fixed_idx:
                registered.append(path)
            else:
                if verbose:
                    print(f"  {Path(path).name} → fixed")
                reg_path = register_to_fixed(fixed_path, path, transform_type, metric)
                registered.append(reg_path)
        current = registered
        
        # Register HR to the same fixed LR stack
        if hr_current:
            if verbose:
                print(f"=== Registering HR scan to fixed LR ===")
                print(f"  {Path(hr_current).name} → fixed")
            hr_current = register_to_fixed(fixed_path, hr_current, transform_type, metric)
        
        # Save registered outputs if requested
        if save_registered:
            if verbose:
                print("=== Saving registered LR outputs ===")
            for orig, curr in zip(inputs, current):
                stem = Path(orig).stem.replace('.nii', '')
                out_path = os.path.join(output_dir, f"{stem}.nii.gz")
                img = nib.load(curr)
                nib.save(img, out_path)
                if verbose:
                    print(f"  Saved: {out_path}")
            
            # Save registered HR
            if hr_current:
                stem = Path(hr_input).stem.replace('.nii', '')
                out_path = os.path.join(output_dir, f"{stem}.nii.gz")
                img = nib.load(hr_current)
                nib.save(img, out_path)
                if verbose:
                    print(f"  Saved: {out_path}")
    
    # Stage 3: Crop to overlap
    if do_crop:
        if verbose:
            print("=== Cropping to overlap ===")
        bbox = find_overlap_bbox(current)
        if verbose:
            print(f"  Overlap bbox: {bbox}")
        
        # Determine suffix based on whether we also saved registered
        suffix = "_cropped" if save_registered else ""
        
        if verbose:
            print("=== Saving cropped LR outputs ===")
        for orig, curr in zip(inputs, current):
            stem = Path(orig).stem.replace('.nii', '')
            out_path = os.path.join(output_dir, f"{stem}{suffix}.nii.gz")
            crop_to_bbox(curr, bbox, out_path)
            if verbose:
                print(f"  Saved: {out_path}")
        
        # Crop HR using same bbox
        if hr_current:
            if verbose:
                print("=== Saving cropped HR output ===")
            stem = Path(hr_input).stem.replace('.nii', '')
            out_path = os.path.join(output_dir, f"{stem}{suffix}.nii.gz")
            crop_to_bbox(hr_current, bbox, out_path)
            if verbose:
                print(f"  Saved: {out_path}")
    
    # If no crop and no save_registered, just save current state
    elif not save_registered:
        if verbose:
            print("=== Saving outputs ===")
        for orig, curr in zip(inputs, current):
            stem = Path(orig).stem.replace('.nii', '')
            out_path = os.path.join(output_dir, f"{stem}.nii.gz")
            img = nib.load(curr)
            nib.save(img, out_path)
            if verbose:
                print(f"  Saved: {out_path}")
        
        if hr_current:
            stem = Path(hr_input).stem.replace('.nii', '')
            out_path = os.path.join(output_dir, f"{stem}.nii.gz")
            img = nib.load(hr_current)
            nib.save(img, out_path)
            if verbose:
                print(f"  Saved: {out_path}")
    
    n_processed = len(inputs) + (1 if hr_input else 0)
    print(f"✓ Processed {n_processed} images → {output_dir}")


# ========== CLI ==========

def main():
    parser = argparse.ArgumentParser(
        description="MRI preprocessing: resample → register → crop overlap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python pipeline.py -i a.nii.gz b.nii.gz c.nii.gz -o out/

  # With HR ground truth scan
  python pipeline.py -i lr1.nii.gz lr2.nii.gz lr3.nii.gz --hr hr.nii.gz -o out/

  # Save both registered and cropped (LR + HR)
  python pipeline.py -i lr1.nii.gz lr2.nii.gz --hr hr.nii.gz -o out/ --save-registered

  # Only resample
  python pipeline.py -i a.nii.gz b.nii.gz -o out/ --no-register --no-crop

  # Custom resolution, metric, and affine registration
  python pipeline.py -i a.nii.gz b.nii.gz -o out/ -r 0.5 0.5 0.5 --metric meansquares --transform Affine
        """
    )
    parser.add_argument("-i", "--inputs", nargs="+", required=True, help="Input LR NIfTI files")
    parser.add_argument("--hr", help="High-resolution ground truth scan (optional)")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-r", "--resolution", nargs=3, type=float, default=[1.0, 1.0, 1.0], help="Target resolution (default: 1 1 1)")
    parser.add_argument("--orientation", default="RAS", help="Target orientation (default: RAS)")
    parser.add_argument("--interp", choices=["trilinear", "bilinear", "nearest"], default="trilinear", help="Interpolation mode (default: trilinear)")
    parser.add_argument("--fixed", type=int, default=0, help="Fixed image index for registration (default: 0)")
    parser.add_argument("--transform", choices=["Rigid", "Affine"], default="Rigid", help="Registration type (default: Rigid)")
    parser.add_argument("--metric", choices=["mattes", "meansquares", "gc"], default="gc", help="Registration metric (default: gc)")
    parser.add_argument("--no-resample", action="store_true", help="Skip resampling")
    parser.add_argument("--no-register", action="store_true", help="Skip registration")
    parser.add_argument("--no-crop", action="store_true", help="Skip overlap cropping")
    parser.add_argument("--save-registered", action="store_true", help="Save registered outputs (cropped will have _cropped suffix)")
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Validate inputs
    for p in args.inputs:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Not found: {p}")
    if args.hr and not os.path.exists(args.hr):
        raise FileNotFoundError(f"HR scan not found: {args.hr}")
    
    run_pipeline(
        inputs=args.inputs,
        output_dir=args.output,
        hr_input=args.hr,
        do_resample=not args.no_resample,
        do_register=not args.no_register,
        do_crop=not args.no_crop,
        save_registered=args.save_registered,
        resolution=args.resolution,
        orientation=args.orientation,
        interp=args.interp,
        fixed_idx=args.fixed,
        transform_type=args.transform,
        metric=args.metric,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()