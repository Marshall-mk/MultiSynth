"""
SegResNet-style Decoder for U-HVED

Key changes from original:
- Pre-activation residual blocks (Norm → ReLU → Conv)
- GroupNorm instead of InstanceNorm
- Trilinear upsampling (default) or transposed conv
- Additive skip connections (SegResNet style) instead of concatenation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class SegResBlock(nn.Module):
    """SegResNet-style pre-activation residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_groups: int = 8
    ):
        super().__init__()

        padding = kernel_size // 2
        num_groups_1 = min(num_groups, in_channels)
        num_groups_2 = min(num_groups, out_channels)

        self.norm1 = nn.GroupNorm(num_groups_1, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, 1, padding)

        self.norm2 = nn.GroupNorm(num_groups_2, out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, 1, padding)

        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        return out + identity


class SegResUpsampleBlock(nn.Module):
    """
    Upsampling block with trilinear interpolation (SegResNet default).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        mode: str = 'trilinear',
        num_groups: int = 8
    ):
        super().__init__()

        self.mode = mode
        self.scale_factor = scale_factor

        if mode == 'trilinear':
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=False)
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        elif mode == 'transpose':
            self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = nn.Identity()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        return x


class SegResDecoderBlock(nn.Module):
    """
    Decoder block with upsampling, skip fusion, and SegResNet-style residual blocks.
    
    SegResNet uses additive skip connections (after channel alignment) rather than concat.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        upsample_mode: str = 'trilinear',
        num_groups: int = 8
    ):
        super().__init__()

        # Upsample
        self.upsample = SegResUpsampleBlock(in_channels, out_channels, mode=upsample_mode, num_groups=num_groups)

        # Skip projection (align channels for additive fusion)
        self.skip_proj = nn.Identity() if skip_channels == out_channels else nn.Conv3d(skip_channels, out_channels, 1)

        # Residual blocks
        blocks = [SegResBlock(out_channels, out_channels, num_groups=num_groups) for _ in range(num_blocks)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upsample(x)

        if skip is not None:
            # Match spatial dims if needed
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='trilinear', align_corners=False)
            # Additive fusion (SegResNet style)
            x = x + self.skip_proj(skip)

        x = self.blocks(x)
        return x


class SegResDecoder(nn.Module):
    """
    Multi-scale SegResNet-style decoder for U-HVED.

    Architecture:
        Deepest latent → [Upsample + Skip + SegResBlocks] × (num_scales-1) → Output
    """

    def __init__(
        self,
        out_channels: int = 1,
        init_filters: int = 32,
        num_scales: int = 4,
        blocks_per_scale: Tuple[int, ...] = (1, 1, 1),
        upsample_mode: str = 'trilinear',
        num_groups: int = 8,
        final_activation: str = 'sigmoid'
    ):
        """
        Args:
            out_channels: Output channels
            init_filters: Base filter count (matches encoder)
            num_scales: Number of scales (matches encoder)
            blocks_per_scale: Residual blocks per decoder level
            upsample_mode: 'trilinear' or 'transpose'
            num_groups: Groups for GroupNorm
            final_activation: 'sigmoid', 'tanh', or 'none'
        """
        super().__init__()

        self.num_scales = num_scales

        # Pad blocks_per_scale
        if len(blocks_per_scale) < num_scales - 1:
            blocks_per_scale = blocks_per_scale + (blocks_per_scale[-1],) * (num_scales - 1 - len(blocks_per_scale))

        # Channel dims at each scale (matching encoder)
        self.scale_channels = [init_filters * (2 ** i) for i in range(num_scales)]

        # Decoder blocks (coarse → fine)
        self.decoder_blocks = nn.ModuleList()
        for i in range(num_scales - 1, 0, -1):
            in_ch = self.scale_channels[i]
            skip_ch = self.scale_channels[i - 1]
            out_ch = self.scale_channels[i - 1]
            block_idx = num_scales - 1 - i

            self.decoder_blocks.append(
                SegResDecoderBlock(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    num_blocks=blocks_per_scale[block_idx],
                    upsample_mode=upsample_mode,
                    num_groups=num_groups
                )
            )

        # Final output
        self.final_norm = nn.GroupNorm(min(num_groups, init_filters), init_filters)
        self.final_relu = nn.ReLU(inplace=True)
        self.final_conv = nn.Conv3d(init_filters, out_channels, kernel_size=1)

        if final_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        elif final_activation == 'tanh':
            self.final_activation = nn.Tanh()
        else:
            self.final_activation = nn.Identity()

    def forward(
        self,
        latent_samples: List[torch.Tensor],
        skip_features: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Args:
            latent_samples: Multi-scale latents [scale_0, scale_1, ..., scale_N] (fine to coarse)
            skip_features: Optional encoder features for skip connections

        Returns:
            Decoded output
        """
        # Start from deepest (coarsest) latent
        x = latent_samples[-1]

        for i, block in enumerate(self.decoder_blocks):
            scale_idx = self.num_scales - 2 - i

            # Get skip (from encoder features or latent samples)
            if skip_features is not None and scale_idx < len(skip_features):
                skip = skip_features[scale_idx]
            elif scale_idx < len(latent_samples):
                skip = latent_samples[scale_idx]
            else:
                skip = None

            x = block(x, skip)

        # Final output
        x = self.final_norm(x)
        x = self.final_relu(x)
        x = self.final_conv(x)
        x = self.final_activation(x)

        return x


class MultiOutputSegResDecoder(nn.Module):
    """
    Decoder producing multiple outputs:
    - Main output (e.g., super-resolved image)
    - Orientation reconstructions (for ELBO loss)
    """

    def __init__(
        self,
        num_orientations: int = 4,
        out_channels: int = 1,
        init_filters: int = 32,
        num_scales: int = 4,
        blocks_per_scale: Tuple[int, ...] = (1, 1, 1),
        upsample_mode: str = 'trilinear',
        num_groups: int = 8,
        share_decoder: bool = False,
        final_activation: str = 'sigmoid'
    ):
        super().__init__()

        self.num_orientations = num_orientations
        self.share_decoder = share_decoder

        # Main decoder
        self.main_decoder = SegResDecoder(
            out_channels=out_channels,
            init_filters=init_filters,
            num_scales=num_scales,
            blocks_per_scale=blocks_per_scale,
            upsample_mode=upsample_mode,
            num_groups=num_groups,
            final_activation=final_activation
        )

        # Orientation decoders
        if share_decoder:
            self.orientation_decoder = SegResDecoder(
                out_channels=out_channels,
                init_filters=init_filters,
                num_scales=num_scales,
                blocks_per_scale=blocks_per_scale,
                upsample_mode=upsample_mode,
                num_groups=num_groups,
                final_activation=final_activation
            )
        else:
            self.orientation_decoders = nn.ModuleList([
                SegResDecoder(
                    out_channels=out_channels,
                    init_filters=init_filters,
                    num_scales=num_scales,
                    blocks_per_scale=blocks_per_scale,
                    upsample_mode=upsample_mode,
                    num_groups=num_groups,
                    final_activation=final_activation
                )
                for _ in range(num_orientations)
            ])

    def forward(
        self,
        latent_samples: List[torch.Tensor],
        skip_features: Optional[List[torch.Tensor]] = None,
        reconstruct_orientations: bool = True
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            main_output: Main decoded output
            orientation_outputs: List of reconstructed orientations
        """
        main_output = self.main_decoder(latent_samples, skip_features)

        orientation_outputs = []
        if reconstruct_orientations:
            for i in range(self.num_orientations):
                decoder = self.orientation_decoder if self.share_decoder else self.orientation_decoders[i]
                orientation_outputs.append(decoder(latent_samples, skip_features))

        return main_output, orientation_outputs
