"""
Test script for 3D U-HVED architecture
"""

import torch
from src.uhved import create_uhved

def test_3d_uhved():
    """Test 3D U-HVED with synthetic data."""
    print("Testing 3D U-HVED architecture...")

    # Create model
    model = create_uhved(
        config='default',
        num_modalities=3,
        in_channels=1,
        out_channels=1,
        base_channels=16,  # Smaller for testing
        num_scales=3,       # Fewer scales for testing
        reconstruct_modalities=False
    )

    # Set to eval mode
    model.eval()

    # Create synthetic 3D data
    batch_size = 2
    depth, height, width = 32, 32, 32

    # Three modalities (e.g., different orientations or degradations)
    modalities = [
        torch.randn(batch_size, 1, depth, height, width),
        torch.randn(batch_size, 1, depth, height, width),
        torch.randn(batch_size, 1, depth, height, width)
    ]

    print(f"Input shape: {modalities[0].shape}")

    # Forward pass
    with torch.no_grad():
        outputs = model(modalities, deterministic=True)

    sr_output = outputs['sr_output']
    posteriors = outputs['posteriors']

    print(f"Output shape: {sr_output.shape}")
    print(f"Number of scales: {len(posteriors)}")

    # Verify shapes
    assert sr_output.shape == (batch_size, 1, depth, height, width), \
        f"Expected output shape {(batch_size, 1, depth, height, width)}, got {sr_output.shape}"

    print("✓ 3D U-HVED test passed!")

    return True


def test_3d_pixelshuffle():
    """Test 3D PixelShuffle utility."""
    print("\nTesting 3D PixelShuffle...")

    from src.utils import PixelShuffle3d

    # Create test input
    batch_size = 2
    in_channels = 8  # 8 = 1 * 2^3 (for upscale factor 2)
    d, h, w = 4, 4, 4

    x = torch.randn(batch_size, in_channels, d, h, w)
    print(f"Input shape: {x.shape}")

    # Apply 3D pixel shuffle
    ps = PixelShuffle3d(upscale_factor=2)
    y = ps(x)

    print(f"Output shape: {y.shape}")

    # Verify shape
    expected_shape = (batch_size, 1, d*2, h*2, w*2)
    assert y.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {y.shape}"

    print("✓ 3D PixelShuffle test passed!")

    return True


def test_3d_upscaler():
    """Test 3D U-HVED with upscaler."""
    print("\nTesting 3D U-HVED with upscaler...")

    # Create model with 2x upscaling
    model = create_uhved(
        config='sr2x',
        num_modalities=3,
        in_channels=1,
        base_channels=16,
        num_scales=3,
        upscale_factor=2
    )

    model.eval()

    # Create synthetic 3D data
    batch_size = 1
    depth, height, width = 16, 16, 16

    modalities = [
        torch.randn(batch_size, 1, depth, height, width),
        torch.randn(batch_size, 1, depth, height, width),
        torch.randn(batch_size, 1, depth, height, width)
    ]

    print(f"Input shape: {modalities[0].shape}")

    # Forward pass
    with torch.no_grad():
        outputs = model(modalities)

    sr_output = outputs['sr_output']

    print(f"Output shape: {sr_output.shape}")

    # Verify upscaled shape
    expected_shape = (batch_size, 1, depth*2, height*2, width*2)
    assert sr_output.shape == expected_shape, \
        f"Expected output shape {expected_shape}, got {sr_output.shape}"

    print("✓ 3D U-HVED with upscaler test passed!")

    return True


if __name__ == '__main__':
    print("=" * 60)
    print("3D U-HVED Architecture Tests")
    print("=" * 60)

    try:
        test_3d_pixelshuffle()
        test_3d_uhved()
        test_3d_upscaler()

        print("\n" + "=" * 60)
        print("All tests passed successfully! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
