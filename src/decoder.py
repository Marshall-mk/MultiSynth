"""
Convolutional Decoder for U-HVED Super-Resolution

The decoder takes multi-scale latent samples and skip connections to
reconstruct the high-resolution output image.

For super-resolution adaptation:
- Output is a single high-resolution image (or multiple Orientation reconstructions)
- Progressive upsampling with skip connections
- Sub-pixel convolution option for efficient upsampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from .utils import PixelShuffle3d


class ResidualBlock(nn.Module):
    """Residual block with instance normalization for decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: str = 'leakyrelu'
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, 1, padding)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, 1, padding)

        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.norm2 = nn.InstanceNorm3d(out_channels)

        if activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        else:
            self.activation = nn.LeakyReLU(0.2, inplace=True)

        # Skip connection
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity
        out = self.activation(out)

        return out


class UpsampleBlock(nn.Module):
    """
    Upsampling block with multiple strategies:
    - Trilinear interpolation + conv
    - Transposed convolution
    - Sub-pixel convolution (PixelShuffle3d)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        mode: str = 'trilinear'
    ):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            scale_factor: Upsampling factor
            mode: 'trilinear', 'transpose', or 'pixelshuffle'
        """
        super().__init__()

        self.mode = mode
        self.scale_factor = scale_factor

        if mode == 'trilinear':
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=False)
            self.conv = nn.Conv3d(in_channels, out_channels, 3, 1, 1)
        elif mode == 'transpose':
            self.upsample = nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=4, stride=scale_factor, padding=1
            )
            self.conv = nn.Identity()
        elif mode == 'pixelshuffle':
            # PixelShuffle3d requires in_channels to be divisible by scale_factor^3
            self.conv_expand = nn.Conv3d(in_channels, out_channels * (scale_factor ** 3), 3, 1, 1)
            self.upsample = PixelShuffle3d(scale_factor)
            self.conv = nn.Identity()
        else:
            raise ValueError(f"Unknown upsample mode: {mode}")

        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'pixelshuffle':
            x = self.conv_expand(x)
            x = self.upsample(x)
        else:
            x = self.upsample(x)
            x = self.conv(x)

        x = self.norm(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    """
    Single decoder block with upsampling, skip connection fusion, and residual blocks.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        num_residual_blocks: int = 2,
        upsample_mode: str = 'bilinear',
        activation: str = 'leakyrelu'
    ):
        """
        Args:
            in_channels: Channels from previous decoder layer
            skip_channels: Channels from encoder skip connection
            out_channels: Output channels
            num_residual_blocks: Number of residual blocks
            upsample_mode: Upsampling strategy
            activation: Activation function
        """
        super().__init__()

        # Upsampling
        self.upsample = UpsampleBlock(in_channels, in_channels, mode=upsample_mode)

        # Skip connection fusion
        # Concatenate upsampled features with skip connection
        fused_channels = in_channels + skip_channels

        # Residual blocks
        layers = [ResidualBlock(fused_channels, out_channels, activation=activation)]
        for _ in range(num_residual_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, activation=activation))

        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input from previous decoder layer
            skip: Skip connection from encoder (optional)

        Returns:
            Decoded features
        """
        # Upsample
        x = self.upsample(x)

        # Fuse with skip connection
        if skip is not None:
            # Resize skip if needed (should match after upsampling)
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='trilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)

        # Process through residual blocks
        x = self.blocks(x)

        return x


class ConvDecoder(nn.Module):
    """
    Multi-scale convolutional decoder for U-HVED super-resolution.

    Takes multi-scale latent samples and produces a high-resolution output.
    Uses progressive upsampling with skip connections from the encoder.
    """

    def __init__(
        self,
        out_channels: int = 1,
        base_channels: int = 32,
        num_scales: int = 4,
        upsample_mode: str = 'bilinear',
        activation: str = 'leakyrelu',
        final_activation: str = 'tanh'
    ):
        """
        Args:
            out_channels: Number of output channels (1 for grayscale, 3 for RGB)
            base_channels: Base channel count (should match encoder)
            num_scales: Number of decoding scales
            upsample_mode: Upsampling strategy
            activation: Hidden layer activation
            final_activation: Output activation ('tanh', 'sigmoid', or 'none')
        """
        super().__init__()

        self.num_scales = num_scales

        # Channel dimensions at each scale (matching encoder)
        self.scale_channels = [base_channels * (2 ** i) for i in range(num_scales)]

        # Decoder blocks (in reverse order from encoder)
        self.decoder_blocks = nn.ModuleList()

        for i in range(num_scales - 1, 0, -1):
            in_ch = self.scale_channels[i]
            skip_ch = self.scale_channels[i - 1]  # Skip from encoder at previous scale
            out_ch = self.scale_channels[i - 1]

            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    upsample_mode=upsample_mode,
                    activation=activation
                )
            )

        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, 3, 1, 1),
            nn.InstanceNorm3d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base_channels, out_channels, 3, 1, 1)
        )

        # Final activation
        if final_activation == 'tanh':
            self.final_activation = nn.Tanh()
        elif final_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()

    def forward(
        self,
        latent_samples: List[torch.Tensor],
        skip_features: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Decode multi-scale latent samples to output image.

        Args:
            latent_samples: List of latent tensors at each scale (coarse to fine)
            skip_features: Optional list of encoder features for skip connections

        Returns:
            Reconstructed image tensor
        """
        # Start from coarsest scale
        x = latent_samples[-1]  # Deepest latent

        # Progressive decoding
        for i, decoder_block in enumerate(self.decoder_blocks):
            # Get corresponding skip connection (in reverse order)
            scale_idx = self.num_scales - 2 - i

            if skip_features is not None and scale_idx < len(skip_features):
                skip = skip_features[scale_idx]
            else:
                # Use latent sample as skip if no encoder features
                skip = latent_samples[scale_idx] if scale_idx < len(latent_samples) else None

            x = decoder_block(x, skip)

        # Final output
        x = self.final_conv(x)
        x = self.final_activation(x)

        return x


class MultiOutputDecoder(nn.Module):
    """
    Decoder that produces multiple outputs:
    - Super-resolved image
    - Reconstructed orientations (for reconstruction loss)

    This follows the original U-HVED design where there are n+1 decoders
    (n for Orientation reconstruction, 1 for the main task).
    """

    def __init__(
        self,
        num_orientations: int = 4,
        out_channels: int = 1,
        base_channels: int = 32,
        num_scales: int = 4,
        upsample_mode: str = 'bilinear',
        activation: str = 'leakyrelu',
        share_decoder: bool = False,
        final_activation: str = 'sigmoid'
    ):
        """
        Args:
            num_orientations: Number of input orientations to reconstruct
            out_channels: Output channels per Orientation/SR output
            base_channels: Base channel count
            num_scales: Number of decoding scales
            upsample_mode: Upsampling strategy
            activation: Activation function
            share_decoder: If True, share decoder weights across orientations
            final_activation: Final output activation ('tanh', 'sigmoid', or 'none')
        """
        super().__init__()

        self.num_orientations = num_orientations
        self.share_decoder = share_decoder

        # Main super-resolution decoder
        self.sr_decoder = ConvDecoder(
            out_channels=out_channels,
            base_channels=base_channels,
            num_scales=num_scales,
            upsample_mode=upsample_mode,
            activation=activation,
            final_activation=final_activation
        )

        # Orientation reconstruction decoders
        if share_decoder:
            self.orientation_decoder = ConvDecoder(
                out_channels=out_channels,
                base_channels=base_channels,
                num_scales=num_scales,
                upsample_mode=upsample_mode,
                activation=activation,
                final_activation=final_activation
            )
        else:
            self.orientation_decoders = nn.ModuleList([
                ConvDecoder(
                    out_channels=out_channels,
                    base_channels=base_channels,
                    num_scales=num_scales,
                    upsample_mode=upsample_mode,
                    activation=activation,
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
        Decode to super-resolved image and reconstructed orientations.

        Args:
            latent_samples: Multi-scale latent samples
            skip_features: Optional encoder skip features
            reconstruct_orientations: Whether to also reconstruct input orientations

        Returns:
            sr_output: Super-resolved image
            orientation_outputs: List of reconstructed orientations (empty if not requested)
        """
        # Main SR output
        sr_output = self.sr_decoder(latent_samples, skip_features)

        # Orientation reconstructions
        orientation_outputs = []
        if reconstruct_orientations:
            for i in range(self.num_orientations):
                if self.share_decoder:
                    mod_out = self.orientation_decoder(latent_samples, skip_features)
                else:
                    mod_out = self.orientation_decoders[i](latent_samples, skip_features)

                # DEBUG -> Check for NaN in orientation decoder output
                if self.training and (not torch.isfinite(mod_out).all()):
                    print(f"\n{'='*80}")
                    print(f"CRITICAL: Orientation decoder {i} produced NaN/Inf!")
                    print(f"  This indicates weight explosion in orientation_decoders[{i}]")
                    print(f"  SR decoder works fine, but this orientation decoder has diverged")
                    print(f"  Output: min={mod_out.min().item():.4f}, max={mod_out.max().item():.4f}, "
                          f"mean={mod_out.mean().item():.4f}")
                    print(f"  has_nan={torch.isnan(mod_out).any().item()}, has_inf={torch.isinf(mod_out).any().item()}")
                    print(f"{'='*80}\n")

                orientation_outputs.append(mod_out)

        return sr_output, orientation_outputs


class PixelShuffleUpscaler(nn.Module):
    """
    Final upscaling module using PixelShuffle3d for super-resolution.
    Used when the output needs to be at a higher resolution than the decoder output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 4
    ):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            scale_factor: Total upscaling factor (2, 4, or 8)
        """
        super().__init__()

        self.scale_factor = scale_factor

        layers = []

        # Cascaded pixel shuffle for larger scale factors
        remaining_scale = scale_factor
        current_channels = in_channels

        while remaining_scale > 1:
            if remaining_scale >= 2:
                upscale = 2
                remaining_scale //= 2
            else:
                break

            layers.extend([
                nn.Conv3d(current_channels, current_channels * (upscale ** 3), 3, 1, 1),
                PixelShuffle3d(upscale),
                nn.LeakyReLU(0.2, inplace=True)
            ])

        # Final output conv
        layers.append(nn.Conv3d(current_channels, out_channels, 3, 1, 1))

        self.upscaler = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upscaler(x)
