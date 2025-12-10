# U-HVED for Super-Resolution
# PyTorch implementation of Hetero-Modal Variational Encoder-Decoder
# Adapted from: https://github.com/ReubenDo/U-HVED
# Paper: https://arxiv.org/abs/1907.11150

from .uhved import UHVED, UHVEDWithUpscale, UHVEDLite, create_uhved
from .encoder import ConvEncoder
from .decoder import ConvDecoder
from .fusion import ProductOfGaussians

from .losses import (
    PerceptualLoss3D,
    SSIM3DLoss,
    UHVEDLoss,
    create_uhved_loss
)

__all__ = [
    'UHVED',
    'UHVEDWithUpscale',
    'UHVEDLite',
    'create_uhved',
    'ConvEncoder',
    'ConvDecoder',
    'ProductOfGaussians',
    'PerceptualLoss3D',
    'SSIM3DLoss',
    'UHVEDLoss',
    'create_uhved_loss'
]
