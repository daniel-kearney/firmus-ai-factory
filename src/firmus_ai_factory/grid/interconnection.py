"""Grid interconnection models.

This module implements utility interconnection dynamics, phase-locked loops,
and synchronization for grid-following systems.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class GridSpecs:
    """Grid connection specifications."""
    name: str
    f_nominal: float        # Nominal frequency (Hz)
    V_nominal: float        # Nominal voltage (V)
    P_max: float            # Maximum facility power (W)
    droop_coefficient: float = 0.04  # Droop for frequency response (pu)
    deadband: float = 0.036  # Frequency deadband (Hz)


class GridInterface:
    """Grid interconnection and demand response management.
    
    Implements:
    - Frequency regulation (primary response)
    - Voltage regulation
    - Power factor control
    - Demand response coordination
    """
    
    def __init__(self, specs: GridSpecs):
        """Initialize grid interface.
        
        Args:
            specs: Grid connection specifications
        """
        self.specs = specs
        
        # Droop control parameters
        self.K_droop = specs.droop_coefficient * specs.P_max
        
        # Demand response state
        self.DR_commitment = 0.0  # Current DR commitment (W)
        self.DR_baseline = 0.0    # Baseline power for DR calculation
        
        # State tracking
        self.P_current = 0.0
        self.f_grid = specs.f_nominal
        self.V_grid = specs.V_nominal
    
    def frequency_response(self, f_grid: float, P_current: float) -> float:
        """Calculate power adjustment for frequency regulation.
        
        Implements primary frequency response with droop control.
        
        Args:
            f_grid: Measured grid frequency (Hz)
            P_current: Current power consumption (W)
            
        Returns:
            P_adjustment: Power adjustment (W, negative = reduce load)
        """
        # Frequency error
        f_error = f_grid - self.specs.f_nominal
        
        # Apply deadband (NERC standard)
        if abs(f_error) < self.specs.deadband:
            return 0.0
        
        # Droop response: reduce load when frequency is low
        # P_adjustment = -K_droop * (f_error - sign(f_error)*deadband)
        P_adjustment = -self.K_droop * (f_error - np.sign(f_error) * self.specs.deadband)
        
        # Limit to available capacity
        P_new = P_current + P_adjustment
        P_new = np.clip(P_new, 0, self.specs.P_max)
        
        return P_new - P_current
    
    def calculate_power_factor(self, P_active: float, Q_reactive: float) -> float:
        """Calculate power factor.
        
        Args:
            P_active: Active power (W)
            Q_reactive: Reactive power (VAR)
            
        Returns:
            pf: Power factor (0-1)
        """
        S_apparent = np.sqrt(P_active**2 + Q_reactive**2)
        if S_apparent > 0:
            pf = P_active / S_apparent
        else:
            pf = 1.0
        return pf
    
    def update_state(self, P_current: float, f_grid: float, V_grid: float):
        """Update grid interface state.
        
        Args:
            P_current: Current power consumption (W)
            f_grid: Grid frequency (Hz)
            V_grid: Grid voltage (V)
        """
        self.P_current = P_current
        self.f_grid = f_grid
        self.V_grid = V_grid


# Standard grid connection specifications

# US Grid (60 Hz)
GRID_US_480V = GridSpecs(
    name="US Grid 480V",
    f_nominal=60.0,
    V_nominal=480.0,
    P_max=10e6,  # 10 MW
    droop_coefficient=0.04,
    deadband=0.036
)

# EU Grid (50 Hz)
GRID_EU_400V = GridSpecs(
    name="EU Grid 400V",
    f_nominal=50.0,
    V_nominal=400.0,
    P_max=10e6,  # 10 MW
    droop_coefficient=0.04,
    deadband=0.030
)
