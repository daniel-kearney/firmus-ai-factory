"""Optimization algorithms for AI factory control.

This module provides model predictive control, workload scheduling,
and multi-objective optimization.
"""

from .mpc import ModelPredictiveController, WorkloadJob

__all__ = [
    'ModelPredictiveController',
    'WorkloadJob',
]
