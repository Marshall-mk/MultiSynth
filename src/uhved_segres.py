"""
U-HVED with SegResNet-style Encoder/Decoder

Changes from original:
- Pre-activation residual blocks: GroupNorm → ReLU → Conv (×2) + skip
- GroupNorm instead of InstanceNorm (better for small 3D batches)
- Additive skip connections in decoder (instead of concat)
- Configurable blocks per scale (like SegResNet's blocks_down/blocks_up)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union

from .encoder_segres import SegResEncoder, MultiModalSegResEncoder
from .decoder_segres import SegResDecoder, MultiOutputSegResDecoder
from .fusion import MultiScaleFusion


class UHVEDSegRes(nn.Module):
    """
    U-HVED with SegResNet-style architecture.

    Architecture:
        Encoder: SegResNet-style blocks with strided conv downsampling
        Fusion: Product of Gaussians (unchanged)
        Decoder: SegResNet-style blocks with trilinear upsampling + additive skips
    """

    def __init__(
        self,
        num_orientations: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        init_filters: int = 32,
        num_scales: int = 4,
        blocks_down: Tuple[int, ...] = (1, 2, 2, 4),
        blocks_up: Tuple[int, ...] = (1, 1, 1),
        num_groups: int = 8,
        share_encoder: bool = False,
        share_decoder: bool = False,
        use_prior: bool = True,
        use_encoder_outputs_as_skip: bool = False,
        upsample_mode: str = 'trilinear',
        reconstruct_orientations: bool = True,
        final_activation: str = 'sigmoid'
    ):
        """
        Args:
            num_orientations: Number of input orientations
            in_channels: Channels per orientation
            out_channels: Output channels
            init_filters: Initial filter count (doubles each scale)
            num_scales: Number of hierarchical scales
            blocks_down: Residual blocks per encoder scale (like SegResNet)
            blocks_up: Residual blocks per decoder scale (like SegResNet)
            num_groups: Groups for GroupNorm
            share_encoder: Share encoder across orientations
            share_decoder: Share decoder for orientation reconstructions
            use_prior: Include prior in Product of Gaussians fusion
            use_encoder_outputs_as_skip: Use encoder features as skip connections
            upsample_mode: 'trilinear' (default) or 'transpose'
            reconstruct_orientations: Decode orientation reconstructions
            final_activation: 'sigmoid', 'tanh', or 'none'
        """
        super().__init__()

        self.num_orientations = num_orientations
        self.num_scales = num_scales
        self.reconstruct_orientations = reconstruct_orientations
        self.use_encoder_outputs_as_skip = use_encoder_outputs_as_skip

        # Encoder
        self.encoder = MultiModalSegResEncoder(
            num_orientations=num_orientations,
            in_channels=in_channels,
            init_filters=init_filters,
            num_scales=num_scales,
            blocks_per_scale=blocks_down,
            num_groups=num_groups,
            share_weights=share_encoder
        )

        # Fusion (unchanged from original U-HVED)
        self.fusion = MultiScaleFusion(num_scales=num_scales, use_prior=use_prior)

        # Decoder
        if reconstruct_orientations:
            self.decoder = MultiOutputSegResDecoder(
                num_orientations=num_orientations,
                out_channels=out_channels,
                init_filters=init_filters,
                num_scales=num_scales,
                blocks_per_scale=blocks_up,
                upsample_mode=upsample_mode,
                num_groups=num_groups,
                share_decoder=share_decoder,
                final_activation=final_activation
            )
        else:
            self.decoder = SegResDecoder(
                out_channels=out_channels,
                init_filters=init_filters,
                num_scales=num_scales,
                blocks_per_scale=blocks_up,
                upsample_mode=upsample_mode,
                num_groups=num_groups,
                final_activation=final_activation
            )

        self.hidden_dims = self.encoder.hidden_dims

    def encode(
        self,
        orientations: List[torch.Tensor],
        orientation_mask: Optional[torch.Tensor] = None
    ) -> List[Dict[str, Dict[int, torch.Tensor]]]:
        """Encode all orientations."""
        return self.encoder(orientations, orientation_mask)

    def fuse(
        self,
        encoder_outputs: List[Dict[str, Dict[int, torch.Tensor]]],
        orientation_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Fuse via Product of Gaussians and sample."""
        return self.fusion(encoder_outputs, orientation_mask, deterministic)

    def decode(
        self,
        latent_samples: List[torch.Tensor],
        encoder_outputs: Optional[List[Dict[str, Dict[int, torch.Tensor]]]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Decode latent samples."""
        skip_features = None
        if self.use_encoder_outputs_as_skip and encoder_outputs is not None:
            skip_features = []
            for scale_data in encoder_outputs:
                features = scale_data.get('features', {})
                if features:
                    # Average across orientations
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
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Full forward pass.

        Returns:
            Dict with 'sr_output', 'orientation_outputs', 'posteriors', 'latent_samples'
        """
        # Encode
        encoder_outputs = self.encode(orientations, orientation_mask)

        # Fuse
        latent_samples, posteriors = self.fuse(encoder_outputs, orientation_mask, deterministic)

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


def create_uhved_segres(config: str = 'default', **kwargs) -> nn.Module:
    """
    Factory function for U-HVED SegRes models.

    Configs:
        'default': Standard 4-scale model
        'small': Lighter model for smaller GPUs
        'deep': More blocks per scale
    """
    configs = {
        'default': {
            'init_filters': 32,
            'num_scales': 4,
            'blocks_down': (1, 2, 2, 4),
            'blocks_up': (1, 1, 1),
            'num_groups': 8
        },
        'small': {
            'init_filters': 16,
            'num_scales': 3,
            'blocks_down': (1, 1, 2),
            'blocks_up': (1, 1),
            'num_groups': 8
        },
        'deep': {
            'init_filters': 32,
            'num_scales': 4,
            'blocks_down': (2, 2, 4, 4),
            'blocks_up': (2, 2, 2),
            'num_groups': 8
        }
    }

    if config not in configs:
        raise ValueError(f"Unknown config: {config}. Available: {list(configs.keys())}")

    cfg = configs[config].copy()
    cfg.update(kwargs)

    return UHVEDSegRes(**cfg)


if __name__ == "__main__":
    print("=" * 60)
    print("U-HVED SegRes Shape Analysis")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_uhved_segres('default', num_orientations=3).to(device)
    model.eval()
    print(model)

    # Test input
    batch_size = 2
    spatial_size = (64, 64, 32)
    orientations = [torch.randn(batch_size, 1, *spatial_size).to(device) for _ in range(3)]

    print(f"\nInput: {len(orientations)} orientations, shape {orientations[0].shape}")

    with torch.no_grad():
        outputs = model(orientations)
        print(f"\nOutput shapes:")
        print(f"  sr_output: {outputs['sr_output'].shape}")
        print(f"  orientation_outputs: {len(outputs['orientation_outputs'])} × {outputs['orientation_outputs'][0].shape if outputs['orientation_outputs'] else 'N/A'}")
        print(f"  posteriors: {len(outputs['posteriors'])} scales")
        for i, (mu, lv) in enumerate(outputs['posteriors']):
            print(f"    Scale {i}: mu={mu.shape}, logvar={lv.shape}")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print("=" * 60)
