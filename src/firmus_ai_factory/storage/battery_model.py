"""Battery models for energy storage systems.

This module implements lithium-ion battery electrochemical models
with thermal effects and degradation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class BatterySpecs:
    """Battery specifications."""
    name: str
    C_nom: float            # Nominal capacity (Ah)
    V_nom: float            # Nominal voltage (V)
    R_series: float         # Series resistance (Ω)
    SOC_min: float = 0.2    # Minimum SOC
    SOC_max: float = 0.9    # Maximum SOC


class LithiumIonBattery:
    """Lithium-ion battery model with thermal effects.
    
    Models battery dynamics including:
    - State-of-charge evolution
    - Open-circuit voltage vs SOC
    - Internal resistance
    - Thermal behavior
    - Capacity fade and degradation
    """
    
    def __init__(self, specs: BatterySpecs, SOC_init: float = 0.5):
        """Initialize battery model.
        
        Args:
            specs: Battery specifications
            SOC_init: Initial state-of-charge (0-1)
        """
        self.specs = specs
        self.SOC = SOC_init
        self.T_cell = 25.0  # Initial temperature (°C)
        
        # Degradation state
        self.capacity_fade = 0.0  # Fractional capacity loss
        self.resistance_growth = 0.0  # Fractional resistance increase
        self.cycles = 0
    
    def open_circuit_voltage(self, SOC: float) -> float:
        """Calculate OCV as function of SOC.
        
        Uses empirical polynomial fit for Li-ion chemistry.
        
        Args:
            SOC: State-of-charge (0-1)
            
        Returns:
            V_oc: Open-circuit voltage (V)
        """
        # Polynomial coefficients for typical Li-ion
        coeffs = [-1.031, 3.685, -2.396, 1.088, 3.131]
        V_oc = np.polyval(coeffs, SOC)
        return V_oc * self.specs.V_nom / 3.6  # Scale to nominal voltage
    
    def update_state(self, I: float, dt: float, T_ambient: float = 25.0) -> Tuple[float, float]:
        """Update battery state for one time step.
        
        Args:
            I: Current (A, positive = discharge)
            dt: Time step (s)
            T_ambient: Ambient temperature (°C)
            
        Returns:
            V_terminal: Terminal voltage (V)
            P_loss: Power loss (W)
        """
        # Coulombic efficiency
        eta = 0.99 if I > 0 else 1.01  # Slightly different for charge/discharge
        
        # Update SOC
        dSOC = -(I * eta * dt) / (self.specs.C_nom * 3600)
        self.SOC = np.clip(self.SOC + dSOC, self.specs.SOC_min, self.specs.SOC_max)
        
        # Calculate voltage
        V_oc = self.open_circuit_voltage(self.SOC)
        R_eff = self.specs.R_series * (1 + self.resistance_growth)
        V_terminal = V_oc - I * R_eff
        
        # Power loss
        P_loss = I**2 * R_eff
        
        # Update temperature (simplified thermal model)
        C_thermal = 1000  # Thermal capacitance (J/K)
        h_conv = 10  # Convection coefficient (W/K)
        dT = (P_loss - h_conv * (self.T_cell - T_ambient)) * dt / C_thermal
        self.T_cell += dT
        
        return V_terminal, P_loss
    
    def update_degradation(self, cycles_increment: float):
        """Update battery degradation models.
        
        Args:
            cycles_increment: Number of equivalent full cycles
        """
        self.cycles += cycles_increment
        
        # Capacity fade model (empirical)
        self.capacity_fade = 0.2 * (self.cycles / 5000)**0.5
        
        # Resistance growth model
        self.resistance_growth = 0.5 * (self.cycles / 5000)**0.75


# Standard battery specifications
BATTERY_TESLA_MEGAPACK = BatterySpecs(
    name="Tesla Megapack",
    C_nom=3000,  # 3000 Ah
    V_nom=400,   # 400 V nominal
    R_series=0.01,  # 10 mΩ
    SOC_min=0.1,
    SOC_max=0.95
)
