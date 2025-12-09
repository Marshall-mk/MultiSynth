#!/usr/bin/env python3
"""
Example demonstrating the three orthogonal LR stack generation.

This shows how the modified data pipeline creates three low-resolution stacks
from a single high-resolution volume, each with high resolution in one orientation.
"""

import torch
from src.data import HRLRDataGenerator

def main():
    print("="*80)
    print("Orthogonal LR Stack Generation Example")
    print("="*80)

    # Create the data generator
    generator = HRLRDataGenerator(
        atlas_res=[1.0, 1.0, 1.0],  # HR resolution
        target_res=[1.0, 1.0, 1.0],  # Target resolution
        output_shape=[128, 128, 128],  # Output shape
        min_resolution=[1.0, 1.0, 1.0],  # Min (best) resolution
        max_res_aniso=[9.0, 9.0, 9.0],  # Max (worst) resolution for anisotropic axes
        randomise_res=True,  # Randomize low-res axes
        prob_bias_field=0.5,
        prob_noise=0.8,
        prob_motion=0.2,
        apply_intensity_aug=True,
    )

    # Create a synthetic high-resolution volume
    batch_size = 2
    hr_volume = torch.randn(batch_size, 1, 128, 128, 128)  # (B, C, D, H, W)

    print(f"\nInput HR Volume Shape: {hr_volume.shape}")
    print(f"  Batch: {hr_volume.shape[0]}")
    print(f"  Channels: {hr_volume.shape[1]}")
    print(f"  Depth (D): {hr_volume.shape[2]}")
    print(f"  Height (H): {hr_volume.shape[3]}")
    print(f"  Width (W): {hr_volume.shape[4]}")

    # Generate three orthogonal LR stacks
    print("\n" + "-"*80)
    print("Generating three orthogonal LR stacks...")
    print("-"*80)

    lr_stacks, hr_augmented, resolutions, thicknesses = generator.generate_paired_data(
        hr_volume,
        return_resolution=True
    )

    print(f"\nNumber of LR stacks generated: {len(lr_stacks)}")
    print(f"HR augmented shape: {hr_augmented.shape}")

    # Display information about each stack
    stack_names = ["Axial (high-res in D)", "Coronal (high-res in H)", "Sagittal (high-res in W)"]
    axis_names = ["D (Depth)", "H (Height)", "W (Width)"]

    for stack_idx in range(3):
        print(f"\n{'='*80}")
        print(f"Stack {stack_idx}: {stack_names[stack_idx]}")
        print(f"{'='*80}")
        print(f"Shape: {lr_stacks[stack_idx].shape}")

        for batch_idx in range(batch_size):
            print(f"\n  Batch {batch_idx}:")
            print(f"    Resolution (mm):")
            for axis_idx in range(3):
                res_val = resolutions[stack_idx][batch_idx, axis_idx].item()
                thick_val = thicknesses[stack_idx][batch_idx, axis_idx].item()
                is_high_res = (axis_idx == stack_idx)
                marker = "★ HIGH-RES" if is_high_res else "☆ low-res"
                print(f"      {axis_names[axis_idx]}: {res_val:.2f} mm (thickness: {thick_val:.2f} mm) {marker}")

    # Verify orthogonality
    print(f"\n{'='*80}")
    print("Verification: Each stack should have high resolution in different axes")
    print(f"{'='*80}")

    for stack_idx in range(3):
        print(f"\nStack {stack_idx} ({stack_names[stack_idx]}):")
        high_res_axes = []
        for batch_idx in range(batch_size):
            batch_high_res = []
            for axis_idx in range(3):
                res_val = resolutions[stack_idx][batch_idx, axis_idx].item()
                if res_val < 2.0:  # Threshold for "high resolution"
                    batch_high_res.append(axis_names[axis_idx])
            high_res_axes.append(batch_high_res)

        print(f"  High-res axes across batch: {high_res_axes}")

        # Check if the expected axis is consistently high-res
        expected_axis = axis_names[stack_idx]
        all_correct = all(expected_axis in axes for axes in high_res_axes)
        status = "✓ CORRECT" if all_correct else "✗ INCORRECT"
        print(f"  Expected high-res axis: {expected_axis} {status}")

    print(f"\n{'='*80}")
    print("Example: Using with U-HVED Model")
    print(f"{'='*80}")

    print("\nThe three LR stacks can be fed directly to U-HVED:")
    print("```python")
    print("from src import UHVED")
    print("")
    print("# Create model with 3 modalities (one per orientation)")
    print("model = UHVED(")
    print("    num_modalities=3,")
    print("    in_channels=1,")
    print("    out_channels=1,")
    print("    base_channels=32,")
    print("    num_scales=4")
    print(")")
    print("")
    print("# Feed the three orthogonal stacks")
    print("outputs = model(lr_stacks)  # lr_stacks is the list of 3 LR volumes")
    print("sr_output = outputs['sr_output']  # Super-resolved HR volume")
    print("```")

    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"✓ Generated {len(lr_stacks)} orthogonal LR stacks from {batch_size} HR volumes")
    print(f"✓ Stack 0: High resolution in axial (D) direction")
    print(f"✓ Stack 1: High resolution in coronal (H) direction")
    print(f"✓ Stack 2: High resolution in sagittal (W) direction")
    print(f"✓ Each stack has independent degradations (bias, noise, motion, etc.)")
    print(f"✓ Ready to feed into U-HVED model for multi-modal super-resolution")
    print("="*80)


if __name__ == "__main__":
    main()
