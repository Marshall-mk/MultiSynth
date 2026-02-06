"""
SegResNet-style Encoder for U-HVED

Key changes from original:
- Pre-activation residual blocks (Norm → ReLU → Conv)
- GroupNorm instead of InstanceNorm (better for small 3D batches)
- Strided convolutions for downsampling
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional


class SegResBlock(nn.Module):
    """
    SegResNet-style pre-activation residual block.
    
    Structure: GroupNorm → ReLU → Conv → GroupNorm → ReLU → Conv → (+skip)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        num_groups: int = 8
    ):
        super().__init__()

        padding = kernel_size // 2
        
        # Ensure num_groups divides channels
        num_groups_1 = min(num_groups, in_channels)
        num_groups_2 = min(num_groups, out_channels)

        # Pre-activation path
        self.norm1 = nn.GroupNorm(num_groups_1, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)

        self.norm2 = nn.GroupNorm(num_groups_2, out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, 1, padding)

        # Skip connection
        self.skip = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv3d(in_channels, out_channels, 1, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        return out + identity


class SegResEncoderBlock(nn.Module):
    """
    Encoder block with SegResNet-style residual blocks.
    Outputs variational parameters (mu, logvar).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        downsample: bool = True,
        num_groups: int = 8
    ):
        super().__init__()

        # First block handles channel change and optional downsampling
        stride = 2 if downsample else 1
        blocks = [SegResBlock(in_channels, out_channels, stride=stride, num_groups=num_groups)]

        # Additional blocks at same resolution
        for _ in range(num_blocks - 1):
            blocks.append(SegResBlock(out_channels, out_channels, num_groups=num_groups))

        self.blocks = nn.Sequential(*blocks)

        # Variational projection: features → (mu, logvar)
        self.variational_proj = nn.Conv3d(out_channels, out_channels * 2, kernel_size=1)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            features: For skip connections
            mu: Variational mean
            logvar: Variational log-variance (clamped)
        """
        features = self.blocks(x)
        params = self.variational_proj(features)

        mu = params[:, :self.out_channels]
        logvar = torch.clamp(params[:, self.out_channels:], -10.0, 10.0)

        return features, mu, logvar


class SegResEncoder(nn.Module):
    """
    Multi-scale SegResNet-style encoder for U-HVED.

    Architecture:
        Input → InitConv → [SegResBlock × n] per scale with downsampling
        Each scale outputs (features, mu, logvar)
    """

    def __init__(
        self,
        in_channels: int = 1,
        init_filters: int = 32,
        num_scales: int = 4,
        blocks_per_scale: Tuple[int, ...] = (1, 2, 2, 4),
        num_groups: int = 8
    ):
        """
        Args:
            in_channels: Input channels
            init_filters: Initial filter count (doubles each scale)
            num_scales: Number of encoding scales
            blocks_per_scale: Number of residual blocks at each scale
            num_groups: Groups for GroupNorm
        """
        super().__init__()

        self.num_scales = num_scales

        # Pad blocks_per_scale if needed
        if len(blocks_per_scale) < num_scales:
            blocks_per_scale = blocks_per_scale + (blocks_per_scale[-1],) * (num_scales - len(blocks_per_scale))

        # Initial convolution (no downsampling)
        self.init_conv = nn.Conv3d(in_channels, init_filters, kernel_size=3, padding=1)

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        in_ch = init_filters

        for i in range(num_scales):
            out_ch = init_filters * (2 ** i)
            downsample = (i > 0)  # First scale stays at input resolution

            self.encoder_blocks.append(
                SegResEncoderBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    num_blocks=blocks_per_scale[i],
                    downsample=downsample,
                    num_groups=num_groups
                )
            )
            in_ch = out_ch

        self.hidden_dims = [init_filters * (2 ** i) for i in range(num_scales)]

    def forward(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Returns:
            List of dicts per scale with 'features', 'mu', 'logvar'
        """
        h = self.init_conv(x)

        outputs = []
        for block in self.encoder_blocks:
            features, mu, logvar = block(h)
            outputs.append({'features': features, 'mu': mu, 'logvar': logvar})
            h = features

        return outputs


class MultiModalSegResEncoder(nn.Module):
    """
    Multi-orientation encoder using SegResNet-style blocks.
    Each orientation encoded independently, then fused via Product of Gaussians.
    """

    def __init__(
        self,
        num_orientations: int = 4,
        in_channels: int = 1,
        init_filters: int = 32,
        num_scales: int = 4,
        blocks_per_scale: Tuple[int, ...] = (1, 2, 2, 4),
        num_groups: int = 8,
        share_weights: bool = False
    ):
        super().__init__()

        self.num_orientations = num_orientations
        self.share_weights = share_weights
        self.num_scales = num_scales

        if share_weights:
            self.encoder = SegResEncoder(
                in_channels, init_filters, num_scales, blocks_per_scale, num_groups
            )
            self.hidden_dims = self.encoder.hidden_dims
        else:
            self.encoders = nn.ModuleList([
                SegResEncoder(in_channels, init_filters, num_scales, blocks_per_scale, num_groups)
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

        Returns:
            List (per scale) of dicts with 'mu', 'logvar', 'features' 
            each mapping orientation_idx → tensor
        """
        scale_outputs = [
            {'mu': {}, 'logvar': {}, 'features': {}}
            for _ in range(self.num_scales)
        ]

        mask_is_batched = orientation_mask is not None and orientation_mask.dim() == 2

        for idx, tensor in enumerate(orientations):
            # Skip if globally masked out
            if orientation_mask is not None and not mask_is_batched:
                if not orientation_mask[idx]:
                    continue

            encoder = self.encoder if self.share_weights else self.encoders[idx]
            enc_out = encoder(tensor)

            for scale_idx, scale_data in enumerate(enc_out):
                scale_outputs[scale_idx]['mu'][idx] = scale_data['mu']
                scale_outputs[scale_idx]['logvar'][idx] = scale_data['logvar']
                scale_outputs[scale_idx]['features'][idx] = scale_data['features']

        return scale_outputs
