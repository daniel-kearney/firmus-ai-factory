"""Grid interface and demand response models.

This module provides grid interconnection, demand response,
and frequency regulation capabilities.
"""

from .interconnection import GridInterface, GridSpecs, GRID_US_480V, GRID_EU_400V
from .demand_response import DemandResponseManager, WorkloadDeferralStrategy, DREvent

__all__ = [
    'GridInterface',
    'GridSpecs',
    'GRID_US_480V',
    'GRID_EU_400V',
    'DemandResponseManager',
    'WorkloadDeferralStrategy',
    'DREvent',
]
