"""Optimization algorithms for AI factory control.

This module provides model predictive control, workload scheduling,
and multi-objective optimization.
"""

from .mpc import ModelPredictiveController, WorkloadJob
from .factory_optimizer import (
    optimize,
    OptimizerResult,
    RoIPack,
    EnergyPack,
    SensitivityEntry,
)

__all__ = [
    'ModelPredictiveController',
    'WorkloadJob',
    'optimize',
    'OptimizerResult',
    'RoIPack',
    'EnergyPack',
    'SensitivityEntry',
]
