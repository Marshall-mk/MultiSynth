"""
Convolutional Encoder for U-HVED
Each input orientation is encoded independently, producing mu and logvar at multiple scales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


class ResidualBlock(nn.Module):
    """Residual block with instance normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_instance_norm: bool = True,
        activation: str = 'leakyrelu'
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, 1, padding)

        self.norm1 = nn.InstanceNorm3d(out_channels) if use_instance_norm else nn.Identity()
        self.norm2 = nn.InstanceNorm3d(out_channels) if use_instance_norm else nn.Identity()

        if activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        else:
            self.activation = nn.LeakyReLU(0.2, inplace=True)

        # Skip connection with 1x1 conv if dimensions change
        self.skip = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv3d(in_channels, out_channels, 1, stride)

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


class EncoderBlock(nn.Module):
    """
    Encoder block that outputs variational parameters (mu, logvar).
    Output channels are split: first half is mu, second half is logvar.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_residual_blocks: int = 2,
        downsample: bool = True,
        activation: str = 'leakyrelu'
    ):
        super().__init__()

        # First residual block handles channel change and optional downsampling
        stride = 2 if downsample else 1
        layers = [ResidualBlock(in_channels, hidden_channels, stride=stride, activation=activation)]

        # Additional residual blocks
        for _ in range(num_residual_blocks - 1):
            layers.append(ResidualBlock(hidden_channels, hidden_channels, activation=activation))

        self.blocks = nn.Sequential(*layers)

        # Output layer produces 2x hidden_channels for mu and logvar
        self.variational_proj = nn.Conv3d(hidden_channels, hidden_channels * 2, kernel_size=1)

        self.hidden_channels = hidden_channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            features: Hidden features for skip connection
            mu: Mean of variational distribution
            logvar: Log-variance of variational distribution (clamped)
        """
        # DEBUG -> Check input to encoder block for NaN/Inf
        if self.training and (not torch.isfinite(x).all()):
            print(f"WARNING: Non-finite values in INPUT to EncoderBlock!")
            print(f"  Input: min={x.min().item():.4f}, max={x.max().item():.4f}, "
                  f"mean={x.mean().item():.4f}, has_nan={torch.isnan(x).any().item()}, "
                  f"has_inf={torch.isinf(x).any().item()}")

        features = self.blocks(x)

        # DEBUG -> Early detection: Check for NaN/Inf in features before projection
        if self.training and (not torch.isfinite(features).all()):
            print(f"WARNING: Non-finite values detected in encoder features!")
            print(f"  Features: min={features.min().item():.4f}, max={features.max().item():.4f}, "
                  f"mean={features.mean().item():.4f}, has_nan={torch.isnan(features).any().item()}")

        variational_params = self.variational_proj(features)

        # Split into mu and logvar
        mu = variational_params[:, :self.hidden_channels]
        logvar = variational_params[:, self.hidden_channels:]

        # Clamp logvar for numerical stability (as in original implementation)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        return features, mu, logvar


class ConvEncoder(nn.Module):
    """
    Multi-scale convolutional encoder for U-HVED.

    Encodes a single orientation and produces variational parameters (mu, logvar)
    at multiple spatial scales for hierarchical latent representation.

    Architecture:
    - Initial projection layer
    - 4 encoder blocks with progressive downsampling
    - Each block outputs mu and logvar at that scale
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_scales: int = 4,
        activation: str = 'leakyrelu'
    ):
        """
        Args:
            in_channels: Number of input channels per orientation
            base_channels: Base number of channels (doubles at each scale)
            num_scales: Number of encoding scales (default 4)
            activation: Activation function type
        """
        super().__init__()

        self.num_scales = num_scales

        # Initial projection
        self.initial_conv = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)
        self.initial_norm = nn.InstanceNorm3d(base_channels)
        self.initial_act = nn.LeakyReLU(0.2, inplace=True) if activation == 'leakyrelu' else nn.ReLU(inplace=True)

        # Encoder blocks at each scale
        self.encoder_blocks = nn.ModuleList()
        in_ch = base_channels

        for i in range(num_scales):
            out_ch = base_channels * (2 ** i)
            # First block doesn't downsample (already at initial scale)
            downsample = (i > 0)
            self.encoder_blocks.append(
                EncoderBlock(
                    in_channels=in_ch,
                    hidden_channels=out_ch,
                    downsample=downsample,
                    activation=activation
                )
            )
            in_ch = out_ch

        # Store hidden dimensions at each scale for decoder reference
        self.hidden_dims = [base_channels * (2 ** i) for i in range(num_scales)]

    def forward(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Encode input and return variational parameters at each scale.

        Args:
            x: Input tensor of shape (B, C, D, H, W)

        Returns:
            List of dicts, each containing:
                - 'features': Hidden features for skip connections
                - 'mu': Mean at this scale
                - 'logvar': Log-variance at this scale
        """
        # Initial projection
        h = self.initial_conv(x)
        h = self.initial_norm(h)
        h = self.initial_act(h)

        outputs = []
        for encoder_block in self.encoder_blocks:
            features, mu, logvar = encoder_block(h)
            outputs.append({
                'features': features,
                'mu': mu,
                'logvar': logvar
            })
            h = features

        return outputs


class MultiModalEncoder(nn.Module):
    """
    Wrapper that creates independent encoders for multiple input orientations.

    For super-resolution, orientations can be:
    - Different degradation types (blur, noise, downsampling)
    - Different scale factors
    - Different image representations
    """

    def __init__(
        self,
        num_orientations: int = 4,
        in_channels: int = 1,
        base_channels: int = 32,
        num_scales: int = 4,
        share_weights: bool = False,
        activation: str = 'leakyrelu'
    ):
        """
        Args:
            num_orientations: Number of input orientations
            in_channels: Channels per orientation
            base_channels: Base channel count
            num_scales: Number of encoding scales
            share_weights: If True, all orientations share the same encoder
            activation: Activation function
        """
        super().__init__()

        self.num_orientations = num_orientations
        self.share_weights = share_weights
        self.num_scales = num_scales

        if share_weights:
            # Single shared encoder
            self.encoder = ConvEncoder(in_channels, base_channels, num_scales, activation)
            self.hidden_dims = self.encoder.hidden_dims
        else:
            # Independent encoder per orientation
            self.encoders = nn.ModuleList([
                ConvEncoder(in_channels, base_channels, num_scales, activation)
                for _ in range(num_orientations)
            ])
            self.hidden_dims = self.encoders[0].hidden_dims

    def forward(
        self,
        orientations: List[torch.Tensor],
        orientation_mask: Optional[torch.Tensor] = None
    ) -> List[Dict[str, Dict[int, torch.Tensor]]]:
        """
        Encode all orientations.

        Args:
            orientations: List of tensors, one per orientation
            orientation_mask: Boolean tensor indicating which orientations are present
                          - Shape (num_orientations,): same mask for all batch elements
                          - Shape (B, num_orientations): different mask per batch element

        Returns:
            List (per scale) of dicts containing 'mu' and 'logvar' dicts
            mapping orientation index to tensor
        """
        # Initialize output structure
        scale_outputs = [
            {'mu': {}, 'logvar': {}, 'features': {}}
            for _ in range(self.num_scales)
        ]

        # Determine if mask is batched
        mask_is_batched = orientation_mask is not None and orientation_mask.dim() == 2

        for mod_idx, mod_input in enumerate(orientations):
            # For global masks (1D), skip entirely if masked out
            # For batched masks (2D), encode all orientations - fusion will handle per-batch masking
            if orientation_mask is not None and not mask_is_batched:
                if not orientation_mask[mod_idx]:
                    continue

            # Get encoder for this orientation
            if self.share_weights:
                encoder = self.encoder
            else:
                encoder = self.encoders[mod_idx]

            # Encode
            mod_outputs = encoder(mod_input)

            # Store at each scale
            for scale_idx, scale_out in enumerate(mod_outputs):
                scale_outputs[scale_idx]['mu'][mod_idx] = scale_out['mu']
                scale_outputs[scale_idx]['logvar'][mod_idx] = scale_out['logvar']
                scale_outputs[scale_idx]['features'][mod_idx] = scale_out['features']

        return scale_outputs
