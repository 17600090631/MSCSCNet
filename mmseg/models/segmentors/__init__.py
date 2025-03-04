# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseSegmentor
from .cascade_encoder_decoder import CascadeEncoderDecoder
from .encoder_decoder import EncoderDecoder
from .seg_tta import SegTTAModel

from .encoder_decoder_local import EncoderDecoder_LOCAL8x8

__all__ = [
    'BaseSegmentor', 'EncoderDecoder', 'CascadeEncoderDecoder', 'SegTTAModel', 'EncoderDecoder_LOCAL8x8'
]
