# U-HVED for Super-Resolution
# PyTorch implementation of Hetero-Modal Variational Encoder-Decoder
# Adapted from: https://github.com/ReubenDo/U-HVED
# Paper: https://arxiv.org/abs/1907.11150

from .uhved import UHVED, UHVEDWithUpscale, UHVEDLite, create_uhved
from .uhved_segres import UHVEDSegRes, create_uhved_segres
from .encoder import ConvEncoder
from .encoder_segres import SegResEncoder
from .decoder import ConvDecoder
from .decoder_segres import SegResDecoder
from .fusion import ProductOfGaussians

from .losses import (
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
    'SSIM3DLoss',
    'UHVEDLoss',
    'create_uhved_loss',
    'create_uhved_segres',
    'SegResEncoder',
    'SegResDecoder',
    'UHVEDSegRes'


]
