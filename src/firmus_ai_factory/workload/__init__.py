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
]
