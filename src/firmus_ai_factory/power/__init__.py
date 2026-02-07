"""Power delivery network models.

This module provides models for the complete power path from utility
connection to GPU voltage regulators, including:
- Transformers
- DC-DC converters
- Voltage regulator modules (VRMs)
- Power quality analysis
"""

from .transformer import (
    TransformerModel,
    TransformerSpecs,
    TRANSFORMER_13_8KV_TO_480V,
    TRANSFORMER_34_5KV_TO_480V
)

from .dc_dc_converter import (
    BuckConverterModel,
    BuckConverterSpecs,
    BUCK_480V_TO_12V,
    BUCK_12V_TO_1V
)

from .voltage_regulator import (
    MultiphaseVRM,
    VRMSpecs,
    VRM_H100_SXM,
    VRM_B200
)

__all__ = [
    # Transformer
    'TransformerModel',
    'TransformerSpecs',
    'TRANSFORMER_13_8KV_TO_480V',
    'TRANSFORMER_34_5KV_TO_480V',
    # DC-DC Converter
    'BuckConverterModel',
    'BuckConverterSpecs',
    'BUCK_480V_TO_12V',
    'BUCK_12V_TO_1V',
    # VRM
    'MultiphaseVRM',
    'VRMSpecs',
    'VRM_H100_SXM',
    'VRM_B200',
]
