"""
Workload Modeling Module for Firmus AI Factory Digital Twin

This module provides workload profile generation based on real H200 benchmark data
from the firmus-model-evaluation framework. Workload profiles can be used across
all infrastructure simulations: cooling, electrical, economic, and grid integration.

Author: Dr. Daniel Kearney
Date: February 2026
"""

from .profiles import (
    WorkloadProfile,
    PhaseCharacteristics,
    EnergyMetrics,
    ThermalProfile,
    WorkloadPhase,
    ModelTier
)

from .generator import (
    SyntheticWorkloadGenerator,
    ModelBenchmarkData,
    BENCHMARK_MODELS,
    generate_workload
)

from .blackwell_smoothing import (
    BlackwellPowerSmoothingProfile,
    BlackwellPowerSmoother,
    PRESET_PROFILES,
    apply_blackwell_smoothing,
    estimate_ups_stress_reduction,
    estimate_grid_stability_improvement,
)

__all__ = [
    # Profile data structures
    'WorkloadProfile',
    'PhaseCharacteristics',
    'EnergyMetrics',
    'ThermalProfile',
    'WorkloadPhase',
    'ModelTier',
    
    # Generator
    'SyntheticWorkloadGenerator',
    'ModelBenchmarkData',
    'BENCHMARK_MODELS',
    'generate_workload',
    
    # Blackwell Power Smoothing
    'BlackwellPowerSmoothingProfile',
    'BlackwellPowerSmoother',
    'PRESET_PROFILES',
    'apply_blackwell_smoothing',
    'estimate_ups_stress_reduction',
    'estimate_grid_stability_improvement',
]
