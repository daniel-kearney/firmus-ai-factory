"""Thermal management models for AI data center cooling.

Provides comprehensive thermal modeling including:
- Immersion cooling (single-phase and two-phase)
- Air cooling for peripheral components (NVLink switches, CPUs, NICs)
- Direct-to-chip liquid cooling for GPU/CPU cold plates

All models support GB300 NVL72 rack-scale analysis with
accurate thermal resistance networks and pPUE calculations.
"""

from firmus_ai_factory.thermal.immersion_cooling import ImmersionCoolingSystem
from firmus_ai_factory.thermal.air_cooling import (
    AirCoolingSystem,
    NVL72PeripheralAirCooling,
)
from firmus_ai_factory.thermal.direct_to_chip import (
    DirectToChipCooling,
    ColdPlateSpec,
)

__all__ = [
    "ImmersionCoolingSystem",
    "AirCoolingSystem",
    "NVL72PeripheralAirCooling",
    "DirectToChipCooling",
    "ColdPlateSpec",
]
