"""
Utility functions for the U-HVED architecture.
"""

import torch
import torch.nn as nn


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
