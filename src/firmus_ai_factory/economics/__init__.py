"""Economic analysis for AI factory operations.

This module provides electricity cost modeling, grid revenue calculation,
and total cost of ownership analysis.
"""

from .electricity_tariff import ElectricityTariff

__all__ = [
    'ElectricityTariff',
]
