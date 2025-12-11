"""
U-HVED: Hetero-Modal Variational Encoder-Decoder for Super-Resolution

PyTorch implementation adapted from:
- Paper: https://arxiv.org/abs/1907.11150
- Original TensorFlow code: https://github.com/ReubenDo/U-HVED

Key concept:
The network forces the model to learn the same feature representation for
different inputs (orientations) through:
1. Independent encoding of each orientation
2. Product of Gaussians fusion in the latent space
3. Shared decoder for reconstruction

For super-resolution, orientations can represent:
- Different degradation types (blur kernels, noise levels)
- Different scale factors
- Different representations of the same image

The variational framework ensures a consistent latent space across orientations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union

from .encoder import ConvEncoder, MultiModalEncoder
from .decoder import ConvDecoder, MultiOutputDecoder
from .fusion import ProductOfGaussians, GaussianSampler, MultiScaleFusion
from .utils import PixelShuffle3d


class UHVED(nn.Module):
    """
    U-HVED: Hetero-Modal Variational Encoder-Decoder

    A variational autoencoder architecture that:
    1. Encodes multiple input orientations independently
    2. Fuses their latent distributions via Product of Gaussians
    3. Samples from the fused posterior
    4. Decodes to produce super-resolved output

    The key innovation is the Product of Gaussians fusion which:
    - Combines information from all available orientations
    - Handles missing orientations gracefully
    - Forces a shared latent representation across orientations
    """

    def __init__(
        self,
        num_orientations: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        num_scales: int = 4,
        share_encoder: bool = False,
        share_decoder: bool = False,
        use_prior: bool = True,
        activation: str = 'leakyrelu',
        upsample_mode: str = 'trilinear',
        reconstruct_orientations: bool = True
    ):
        """
        Args:
            num_orientations: Number of input orientations
            in_channels: Channels per input orientation
            out_channels: Output channels
            base_channels: Base channel count (doubles at each scale)
            num_scales: Number of hierarchical scales
            share_encoder: Share encoder weights across orientations
            share_decoder: Share decoder weights for orientation reconstruction
            use_prior: Include prior in Product of Gaussians
            activation: Activation function type
            upsample_mode: Upsampling strategy in decoder
            reconstruct_orientations: Whether to reconstruct input orientations
        """
        super().__init__()

        self.num_orientations = num_orientations
        self.num_scales = num_scales
        self.reconstruct_orientations = reconstruct_orientations

        # Multi-modal encoder
        self.encoder = MultiModalEncoder(
            num_orientations=num_orientations,
            in_channels=in_channels,
            base_channels=base_channels,
            num_scales=num_scales,
            share_weights=share_encoder,
            activation=activation
        )

        # Multi-scale fusion
        self.fusion = MultiScaleFusion(
            num_scales=num_scales,
            use_prior=use_prior
        )

        # Decoder(s)
        if reconstruct_orientations:
            self.decoder = MultiOutputDecoder(
                num_orientations=num_orientations,
                out_channels=out_channels,
                base_channels=base_channels,
                num_scales=num_scales,
                upsample_mode=upsample_mode,
                activation=activation,
                share_decoder=share_decoder
            )
        else:
            self.decoder = ConvDecoder(
                out_channels=out_channels,
                base_channels=base_channels,
                num_scales=num_scales,
                upsample_mode=upsample_mode,
                activation=activation
            )

        # Store dimensions for external reference
        self.hidden_dims = self.encoder.hidden_dims

    def encode(
        self,
        orientations: List[torch.Tensor],
        orientation_mask: Optional[torch.Tensor] = None
    ) -> List[Dict[str, Dict[int, torch.Tensor]]]:
        """
        Encode all orientations.

        Args:
            orientations: List of input tensors (one per orientation)
            orientation_mask: Boolean mask for present orientations

        Returns:
            Multi-scale encoder outputs with mu/logvar per orientation
        """
        return self.encoder(orientations, orientation_mask)

    def fuse(
        self,
        encoder_outputs: List[Dict[str, Dict[int, torch.Tensor]]],
        orientation_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Fuse encoded orientations via Product of Gaussians and sample.

        Args:
            encoder_outputs: Output from encoder
            orientation_mask: Boolean mask for present orientations
            deterministic: If True, return mean without sampling

        Returns:
            samples: Multi-scale latent samples
            posteriors: Multi-scale (mu, logvar) for loss computation
        """
        return self.fusion(encoder_outputs, orientation_mask, deterministic)

    def decode(
        self,
        latent_samples: List[torch.Tensor],
        encoder_outputs: Optional[List[Dict[str, Dict[int, torch.Tensor]]]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Decode latent samples to output.

        Args:
            latent_samples: Multi-scale latent samples
            encoder_outputs: Optional encoder outputs for skip connections

        Returns:
            Super-resolved output (and orientation reconstructions if enabled)
        """
        # Extract skip features from encoder
        skip_features = None
        if encoder_outputs is not None:
            # Use features from first available orientation as skip
            skip_features = []
            for scale_data in encoder_outputs:
                features = scale_data.get('features', {})
                if features:
                    # Average features across orientations
                    feat_list = list(features.values())
                    avg_feat = torch.stack(feat_list, dim=0).mean(dim=0)
                    skip_features.append(avg_feat)

        if self.reconstruct_orientations:
            return self.decoder(latent_samples, skip_features, reconstruct_orientations=True)
        else:
            return self.decoder(latent_samples, skip_features)

    def forward(
        self,
        orientations: List[torch.Tensor],
        orientation_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Full forward pass.

        Args:
            orientations: List of input tensors (B, C, D, H, W) per orientation
            orientation_mask: Boolean tensor indicating which orientations are present
            deterministic: If True, use mean instead of sampling

        Returns:
            Dictionary containing:
                - 'sr_output': Super-resolved output
                - 'orientation_outputs': Reconstructed orientations (if enabled)
                - 'posteriors': List of (mu, logvar) at each scale
                - 'latent_samples': Multi-scale latent samples
        """
        # Encode
        encoder_outputs = self.encode(orientations, orientation_mask)

        # Fuse and sample
        latent_samples, posteriors = self.fuse(
            encoder_outputs,
            orientation_mask,
            deterministic or not self.training
        )

        # Decode
        if self.reconstruct_orientations:
            sr_output, orientation_outputs = self.decode(latent_samples, encoder_outputs)
        else:
            sr_output = self.decode(latent_samples, encoder_outputs)
            orientation_outputs = []

        return {
            'sr_output': sr_output,
            'orientation_outputs': orientation_outputs,
            'posteriors': posteriors,
            'latent_samples': latent_samples
        }


class UHVEDLite(nn.Module):
    """
    Lightweight version of U-HVED for faster training/inference.

    Differences from full U-HVED:
    - Fewer scales
    - Shared encoder/decoder
    - No orientation reconstruction branch
    """

    def __init__(
        self,
        num_orientations: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        num_scales: int = 3
    ):
        super().__init__()

        self.num_orientations = num_orientations

        # Shared encoder for all orientations
        self.encoder = ConvEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            num_scales=num_scales
        )

        # Fusion
        self.fusion = MultiScaleFusion(num_scales=num_scales)

        # Single decoder
        self.decoder = ConvDecoder(
            out_channels=out_channels,
            base_channels=base_channels,
            num_scales=num_scales
        )

    def forward(
        self,
        orientations: List[torch.Tensor],
        orientation_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            orientations: List of input tensors
            orientation_mask: Boolean mask for present orientations

        Returns:
            Dictionary with 'sr_output' and 'posteriors'
        """
        # Encode each orientation with shared encoder
        all_outputs = [{'mu': {}, 'logvar': {}, 'features': {}} for _ in range(len(self.encoder.hidden_dims))]

        for mod_idx, mod_input in enumerate(orientations):
            if orientation_mask is not None and not orientation_mask[mod_idx]:
                continue

            mod_outputs = self.encoder(mod_input)

            for scale_idx, scale_out in enumerate(mod_outputs):
                all_outputs[scale_idx]['mu'][mod_idx] = scale_out['mu']
                all_outputs[scale_idx]['logvar'][mod_idx] = scale_out['logvar']
                all_outputs[scale_idx]['features'][mod_idx] = scale_out['features']

        # Fuse
        latent_samples, posteriors = self.fusion(all_outputs, orientation_mask)

        # Decode
        sr_output = self.decoder(latent_samples)

        return {
            'sr_output': sr_output,
            'posteriors': posteriors,
            'orientation_outputs': []
        }


class UHVEDWithUpscale(nn.Module):
    """
    U-HVED with additional upscaling for higher super-resolution factors.

    The base U-HVED maintains spatial resolution through the encoder-decoder.
    This variant adds explicit upscaling at the end for 2x, 4x, or 8x SR.
    """

    def __init__(
        self,
        num_orientations: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        num_scales: int = 4,
        upscale_factor: int = 4
    ):
        """
        Args:
            num_orientations: Number of input orientations
            in_channels: Input channels
            out_channels: Output channels
            base_channels: Base channel count
            num_scales: Number of scales
            upscale_factor: Final upscaling factor (2, 4, or 8)
        """
        super().__init__()

        self.upscale_factor = upscale_factor

        # Base U-HVED
        self.uhved = UHVED(
            num_orientations=num_orientations,
            in_channels=in_channels,
            out_channels=base_channels,  # Output features, not final image
            base_channels=base_channels,
            num_scales=num_scales,
            reconstruct_orientations=False
        )

        # Upscaling module
        self.upscale = self._build_upscaler(base_channels, out_channels, upscale_factor)

    def _build_upscaler(
        self,
        in_channels: int,
        out_channels: int,
        scale: int
    ) -> nn.Module:
        """Build pixel-shuffle 3D upscaler."""
        layers = []
        remaining = scale

        while remaining > 1:
            layers.extend([
                nn.Conv3d(in_channels, in_channels * 8, 3, 1, 1),  # 8 = 2^3 for 3D
                PixelShuffle3d(2),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            remaining //= 2

        layers.append(nn.Conv3d(in_channels, out_channels, 3, 1, 1))
        layers.append(nn.Tanh())

        return nn.Sequential(*layers)

    def forward(
        self,
        orientations: List[torch.Tensor],
        orientation_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with upscaling.

        Args:
            orientations: List of input tensors
            orientation_mask: Boolean mask

        Returns:
            Dictionary with upscaled 'sr_output' and 'posteriors'
        """
        # Get base features
        outputs = self.uhved(orientations, orientation_mask)

        # Upscale
        sr_output = self.upscale(outputs['sr_output'])

        return {
            'sr_output': sr_output,
            'posteriors': outputs['posteriors'],
            'orientation_outputs': outputs['orientation_outputs']
        }


def create_uhved(
    config: str = 'default',
    **kwargs
) -> nn.Module:
    """
    Factory function to create U-HVED models.

    Args:
        config: Configuration preset
            - 'default': Standard U-HVED
            - 'lite': Lightweight version
            - 'sr2x': With 2x upscaling
            - 'sr4x': With 4x upscaling
        **kwargs: Additional arguments passed to model

    Returns:
        U-HVED model
    """
    configs = {
        'default': {
            'class': UHVED,
            'num_orientations': 4,
            'base_channels': 32,
            'num_scales': 4
        },
        'lite': {
            'class': UHVEDLite,
            'num_orientations': 4,
            'base_channels': 16,
            'num_scales': 3
        },
        'sr2x': {
            'class': UHVEDWithUpscale,
            'num_orientations': 4,
            'base_channels': 32,
            'num_scales': 4,
            'upscale_factor': 2
        },
        'sr4x': {
            'class': UHVEDWithUpscale,
            'num_orientations': 4,
            'base_channels': 32,
            'num_scales': 4,
            'upscale_factor': 4
        }
    }

    if config not in configs:
        raise ValueError(f"Unknown config: {config}. Available: {list(configs.keys())}")

    cfg = configs[config].copy()
    model_class = cfg.pop('class')
    cfg.update(kwargs)

    return model_class(**cfg)
