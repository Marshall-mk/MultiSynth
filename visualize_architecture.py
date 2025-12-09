#!/usr/bin/env python3
"""
Visualize U-HVED model architectures using torchviz for different configurations.

Usage:
    python visualize_architecture.py --config default
    python visualize_architecture.py --config lite
    python visualize_architecture.py --config sr2x
    python visualize_architecture.py --config sr4x
    python visualize_architecture.py --config all
"""

import torch
from torchviz import make_dot
import argparse
from pathlib import Path

from src import UHVED, UHVEDLite, UHVEDWithUpscale, create_uhved


def create_sample_inputs(num_modalities=3, in_channels=1, height=64, width=64, batch_size=1):
    """
    Create sample input tensors for the model.

    Args:
        num_modalities: Number of input modalities
        in_channels: Number of channels per modality
        height: Input height
        width: Input width
        batch_size: Batch size

    Returns:
        List of input tensors, one per modality
    """
    modalities = []
    for _ in range(num_modalities):
        modality = torch.randn(batch_size, in_channels, height, width)
        modalities.append(modality)
    return modalities


def visualize_uhved_config(config_name, output_dir="architecture_plots"):
    """
    Visualize a specific U-HVED configuration.

    Args:
        config_name: Configuration name (default, lite, sr2x, sr4x, or custom)
        output_dir: Directory to save the visualization
    """
    print(f"\n{'='*60}")
    print(f"Visualizing U-HVED configuration: {config_name}")
    print(f"{'='*60}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Create model based on configuration
    if config_name == "default":
        model = create_uhved("default")
        input_size = (64, 64)
    elif config_name == "lite":
        model = create_uhved("lite")
        input_size = (64, 64)
    elif config_name == "sr2x":
        model = create_uhved("sr2x")
        input_size = (64, 64)
    elif config_name == "sr4x":
        model = create_uhved("sr4x")
        input_size = (64, 64)
    elif config_name == "custom":
        # Custom configuration example
        model = UHVED(
            num_modalities=3,
            in_channels=1,
            out_channels=1,
            base_channels=16,
            num_scales=3,
            share_encoder=True,
            share_decoder=True,
            reconstruct_modalities=False
        )
        input_size = (64, 64)
    else:
        raise ValueError(f"Unknown configuration: {config_name}")

    model.eval()

    # Get model info
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")

    # Create sample inputs
    num_modalities = 3
    in_channels = 1

    print(f"Input shape per modality: (1, {in_channels}, {input_size[0]}, {input_size[1]})")
    print(f"Number of modalities: {num_modalities}")

    modalities = create_sample_inputs(
        num_modalities=num_modalities,
        in_channels=in_channels,
        height=input_size[0],
        width=input_size[1],
        batch_size=1
    )

    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(modalities)

    # Handle different output formats
    if isinstance(outputs, dict):
        # Extract the main super-resolution output
        if 'sr_output' in outputs:
            output_tensor = outputs['sr_output']
            print(f"SR output shape: {output_tensor.shape}")
        elif 'output' in outputs:
            output_tensor = outputs['output']
            print(f"Output shape: {output_tensor.shape}")
        else:
            # Use the first available tensor output
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    output_tensor = value
                    print(f"Using output '{key}' with shape: {output_tensor.shape}")
                    break
    else:
        output_tensor = outputs
        print(f"Output shape: {output_tensor.shape}")

    # Re-run with gradients for visualization
    print("Generating visualization...")
    model.train()  # Enable gradients
    modalities_grad = [m.requires_grad_(True) for m in modalities]
    outputs_grad = model(modalities_grad)

    if isinstance(outputs_grad, dict):
        output_tensor_grad = outputs_grad.get('sr_output',
                                              outputs_grad.get('output',
                                                             list(outputs_grad.values())[0]))
    else:
        output_tensor_grad = outputs_grad

    # Create visualization
    # Use mean to reduce to scalar for visualization
    dot = make_dot(
        output_tensor_grad.mean(),
        params=dict(model.named_parameters()),
        show_attrs=True,
        show_saved=False  # Set to True for more detailed graphs (can be very large)
    )

    # Save visualization
    output_file = output_path / f"uhved_{config_name}"
    print(f"Saving visualization to: {output_file}.png")
    dot.render(str(output_file), format="png", cleanup=True)
    print(f"✓ Visualization saved successfully!")

    # Also save as PDF for better quality
    print(f"Saving high-quality PDF to: {output_file}.pdf")
    dot.render(str(output_file), format="pdf", cleanup=True)
    print(f"✓ PDF saved successfully!")

    return output_file


def visualize_all_configs(output_dir="architecture_plots"):
    """Visualize all standard U-HVED configurations."""
    configs = ["default", "lite", "sr2x", "sr4x"]

    print("\n" + "="*60)
    print("VISUALIZING ALL U-HVED CONFIGURATIONS")
    print("="*60)

    results = {}
    for config in configs:
        try:
            output_file = visualize_uhved_config(config, output_dir)
            results[config] = "✓ Success"
        except Exception as e:
            print(f"✗ Error visualizing {config}: {e}")
            results[config] = f"✗ Failed: {e}"

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for config, status in results.items():
        print(f"{config:15s}: {status}")
    print("\n")


def compare_configurations():
    """Print a comparison table of different configurations."""
    configs = {
        "default": create_uhved("default"),
        "lite": create_uhved("lite"),
        "sr2x": create_uhved("sr2x"),
        "sr4x": create_uhved("sr4x"),
    }

    print("\n" + "="*80)
    print("U-HVED CONFIGURATION COMPARISON")
    print("="*80)
    print(f"{'Config':<15} {'Parameters':<15} {'Scales':<10} {'Base Ch':<10} {'Upscale':<10}")
    print("-"*80)

    for name, model in configs.items():
        num_params = sum(p.numel() for p in model.parameters())

        # Extract model details
        if hasattr(model, 'uhved'):
            base_model = model.uhved
            has_upscale = True
        else:
            base_model = model
            has_upscale = False

        num_scales = base_model.num_scales if hasattr(base_model, 'num_scales') else "N/A"
        base_channels = base_model.base_channels if hasattr(base_model, 'base_channels') else "N/A"
        upscale = "Yes" if has_upscale else "No"

        print(f"{name:<15} {num_params:<15,} {num_scales:<10} {base_channels:<10} {upscale:<10}")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize U-HVED model architectures using torchviz"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=["default", "lite", "sr2x", "sr4x", "custom", "all"],
        help="Model configuration to visualize"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="architecture_plots",
        help="Directory to save visualization plots"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show comparison table of all configurations"
    )
    parser.add_argument(
        "--show-saved",
        action="store_true",
        help="Include saved tensors in visualization (creates larger graphs)"
    )

    args = parser.parse_args()

    # Show comparison table if requested
    if args.compare:
        compare_configurations()

    # Visualize configurations
    if args.config == "all":
        visualize_all_configs(args.output_dir)
    else:
        visualize_uhved_config(args.config, args.output_dir)

    print("\n" + "="*60)
    print("DONE!")
    print(f"Check the '{args.output_dir}' directory for visualization files.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
