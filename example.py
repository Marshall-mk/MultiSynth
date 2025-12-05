"""
Example usage of U-HVED for Super-Resolution

This script demonstrates:
1. How to create and initialize the model
2. How to prepare multi-modal inputs
3. How to run inference
4. How to handle missing modalities
5. How to compute losses for training
"""

import torch
import torch.nn.functional as F
from models import UHVED, UHVEDLoss, create_uhved


def basic_usage():
    """Basic usage example."""
    print("=" * 60)
    print("Basic U-HVED Usage")
    print("=" * 60)

    # Create model
    model = UHVED(
        num_modalities=4,      # Number of input types (e.g., different degradations)
        in_channels=1,         # Channels per modality (1 for grayscale, 3 for RGB)
        out_channels=1,        # Output channels
        base_channels=32,      # Base feature channels
        num_scales=4,          # Number of hierarchical scales
        reconstruct_modalities=True  # Also reconstruct input modalities
    )

    # Create dummy input - 4 modalities, each (batch=2, channels=1, H=64, W=64)
    batch_size = 2
    height, width = 64, 64

    modalities = [
        torch.randn(batch_size, 1, height, width)  # Modality 1: e.g., bicubic downsampled
        for _ in range(4)
    ]

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(modalities)

    print(f"Input shape (per modality): {modalities[0].shape}")
    print(f"SR Output shape: {outputs['sr_output'].shape}")
    print(f"Number of reconstructed modalities: {len(outputs['modality_outputs'])}")
    print(f"Number of posterior distributions: {len(outputs['posteriors'])}")

    # Check posterior dimensions at each scale
    for i, (mu, logvar) in enumerate(outputs['posteriors']):
        print(f"  Scale {i}: mu shape = {mu.shape}, logvar shape = {logvar.shape}")


def missing_modality_example():
    """Example showing how to handle missing modalities."""
    print("\n" + "=" * 60)
    print("Handling Missing Modalities")
    print("=" * 60)

    model = UHVED(num_modalities=4, in_channels=1, out_channels=1)

    batch_size = 2
    height, width = 64, 64

    # Create 4 modalities
    modalities = [torch.randn(batch_size, 1, height, width) for _ in range(4)]

    # Create mask: only modalities 0 and 2 are available
    modality_mask = torch.tensor([True, False, True, False])

    model.eval()
    with torch.no_grad():
        outputs = model(modalities, modality_mask=modality_mask)

    print(f"Available modalities: {modality_mask.tolist()}")
    print(f"SR Output shape: {outputs['sr_output'].shape}")
    print("Model successfully handles missing modalities via Product of Gaussians fusion!")


def training_example():
    """Example showing training loop setup."""
    print("\n" + "=" * 60)
    print("Training Setup Example")
    print("=" * 60)

    # Create model
    model = UHVED(
        num_modalities=4,
        in_channels=3,  # RGB images
        out_channels=3,
        base_channels=32,
        num_scales=4,
        reconstruct_modalities=True
    )

    # Create loss function
    loss_fn = UHVEDLoss(
        recon_loss_type='l1',        # L1 reconstruction loss
        recon_weight=1.0,             # Weight for main SR loss
        kl_weight=0.001,              # Weight for KL divergence
        perceptual_weight=0.1,        # Weight for perceptual loss
        modality_weight=0.5,          # Weight for modality reconstruction
        use_perceptual=False,         # Disable perceptual loss for this example
        kl_annealing=True,            # Gradually increase KL weight
        kl_anneal_steps=10000         # Annealing steps
    )

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Dummy data
    batch_size = 4
    lr_size = 64   # Low-resolution input size
    hr_size = 64   # High-resolution target size (same in this example)

    # Multiple degraded inputs
    modalities = [torch.randn(batch_size, 3, lr_size, lr_size) for _ in range(4)]
    hr_target = torch.randn(batch_size, 3, hr_size, hr_size)

    # Training step
    model.train()

    optimizer.zero_grad()

    # Forward pass
    outputs = model(modalities)

    # Compute loss
    losses = loss_fn(
        sr_output=outputs['sr_output'],
        sr_target=hr_target,
        posteriors=outputs['posteriors'],
        modality_outputs=outputs['modality_outputs'],
        modality_targets=modalities,
        return_components=True
    )

    # Backward pass
    losses['total'].backward()
    optimizer.step()

    print("Loss components:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")


def different_configs_example():
    """Example showing different model configurations."""
    print("\n" + "=" * 60)
    print("Different Model Configurations")
    print("=" * 60)

    configs = ['default', 'lite', 'sr2x', 'sr4x']

    for config_name in configs:
        model = create_uhved(config_name, in_channels=3, out_channels=3)

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())

        print(f"\n{config_name}:")
        print(f"  Parameters: {num_params:,}")
        print(f"  Type: {type(model).__name__}")


def super_resolution_workflow():
    """Complete super-resolution workflow example."""
    print("\n" + "=" * 60)
    print("Complete Super-Resolution Workflow")
    print("=" * 60)

    # Step 1: Define degradation functions
    def create_degraded_modalities(hr_image: torch.Tensor, scale: int = 4):
        """Create multiple degraded versions of an HR image."""
        b, c, h, w = hr_image.shape
        lr_h, lr_w = h // scale, w // scale

        modalities = []

        # Modality 1: Bicubic downsampling
        lr_bicubic = F.interpolate(hr_image, size=(lr_h, lr_w), mode='bicubic', align_corners=False)
        modalities.append(lr_bicubic)

        # Modality 2: Bilinear downsampling
        lr_bilinear = F.interpolate(hr_image, size=(lr_h, lr_w), mode='bilinear', align_corners=False)
        modalities.append(lr_bilinear)

        # Modality 3: Bicubic + noise
        noise = torch.randn_like(lr_bicubic) * 0.05
        lr_noisy = torch.clamp(lr_bicubic + noise, -1, 1)
        modalities.append(lr_noisy)

        # Modality 4: Area downsampling (different frequency characteristics)
        lr_area = F.interpolate(hr_image, size=(lr_h, lr_w), mode='area')
        modalities.append(lr_area)

        return modalities

    # Step 2: Create model
    model = create_uhved('default', in_channels=3, out_channels=3)
    model.eval()

    # Step 3: Create synthetic HR image
    hr_image = torch.randn(1, 3, 256, 256)

    # Step 4: Create degraded inputs
    modalities = create_degraded_modalities(hr_image, scale=4)

    print(f"HR image shape: {hr_image.shape}")
    print(f"LR modality shape: {modalities[0].shape}")
    print(f"Number of modalities: {len(modalities)}")

    # Step 5: Super-resolve
    with torch.no_grad():
        outputs = model(modalities)

    sr_output = outputs['sr_output']
    print(f"SR output shape: {sr_output.shape}")

    # Step 6: Test with partial modalities
    print("\nTesting robustness to missing modalities:")
    for num_available in [4, 3, 2, 1]:
        mask = torch.zeros(4, dtype=torch.bool)
        mask[:num_available] = True

        with torch.no_grad():
            outputs = model(modalities, modality_mask=mask)

        # Compute simple quality metric (MSE with HR target)
        sr = outputs['sr_output']
        # In real scenario, upsample HR target to match SR output size if needed
        mse = F.mse_loss(sr, F.interpolate(hr_image, size=sr.shape[-2:], mode='bicubic', align_corners=False))
        print(f"  {num_available} modalities available - Output MSE: {mse.item():.4f}")


def latent_space_analysis():
    """Example showing latent space properties."""
    print("\n" + "=" * 60)
    print("Latent Space Analysis")
    print("=" * 60)

    model = UHVED(num_modalities=4, in_channels=1, out_channels=1, num_scales=4)
    model.eval()

    # Create two similar images
    img_a = torch.randn(1, 1, 64, 64)
    img_b = img_a + torch.randn_like(img_a) * 0.1  # Slight perturbation

    # Create modalities from each
    modalities_a = [img_a.clone() for _ in range(4)]
    modalities_b = [img_b.clone() for _ in range(4)]

    with torch.no_grad():
        outputs_a = model(modalities_a)
        outputs_b = model(modalities_b)

    # Compare latent representations at each scale
    print("Latent space similarity across scales:")
    for i, ((mu_a, _), (mu_b, _)) in enumerate(zip(outputs_a['posteriors'], outputs_b['posteriors'])):
        # Compute cosine similarity
        mu_a_flat = mu_a.flatten()
        mu_b_flat = mu_b.flatten()
        cosine_sim = F.cosine_similarity(mu_a_flat.unsqueeze(0), mu_b_flat.unsqueeze(0))
        print(f"  Scale {i}: Cosine similarity = {cosine_sim.item():.4f}")


def main():
    """Run all examples."""
    print("\n" + "#" * 60)
    print("# U-HVED Super-Resolution Examples")
    print("#" * 60)

    basic_usage()
    missing_modality_example()
    training_example()
    different_configs_example()
    super_resolution_workflow()
    latent_space_analysis()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
