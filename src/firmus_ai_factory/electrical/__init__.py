"""Electrical Infrastructure Module

Mathematical models for electrical power distribution and conditioning systems:
- UPS (Uninterruptible Power Supply) systems
- Power transformers
- Power distribution units (PDUs)
- Grid interaction and power quality

Author: daniel.kearney@firmus.co
"""

from firmus_ai_factory.electrical.ups_model import (
    UPSSystem,
    UPSSpecifications,
    BatterySpecifications,
    UPSMode,
    BatteryTechnology,
    EATON_9395XR_1935KVA_SPECS,
    ESS_MODE_EFFICIENCY_CURVE,
    calculate_ups_array_capacity,
)

from firmus_ai_factory.electrical.transformer_model import (
    TransformerModel,
    TransformerSpecifications,
    CoolingType,
    InsulationClass,
    MV_TRANSFORMER_2500KVA_SPECS,
    MV_TRANSFORMER_10MVA_SPECS,
    calculate_transformer_array_capacity,
)

__all__ = [
    # UPS classes and functions
    "UPSSystem",
    "UPSSpecifications",
    "BatterySpecifications",
    "UPSMode",
    "BatteryTechnology",
    "EATON_9395XR_1935KVA_SPECS",
    "ESS_MODE_EFFICIENCY_CURVE",
    "calculate_ups_array_capacity",
    # Transformer classes and functions
    "TransformerModel",
    "TransformerSpecifications",
    "CoolingType",
    "InsulationClass",
    "MV_TRANSFORMER_2500KVA_SPECS",
    "MV_TRANSFORMER_10MVA_SPECS",
    "calculate_transformer_array_capacity",
]
