"""Transformer models for power delivery network.

This module implements three-phase transformer models with losses,
voltage regulation, and efficiency calculations.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class TransformerSpecs:
    """Transformer specifications."""
    name: str
    S_rated: float          # Rated power (VA)
    V_primary: float        # Primary voltage (V)
    V_secondary: float      # Secondary voltage (V)
    Z_leakage_pu: float     # Leakage impedance (per-unit)
    X_M_pu: float           # Magnetizing reactance (per-unit)
    core_loss_fraction: float = 0.01  # Core loss as fraction of rated power


class TransformerModel:
    """Three-phase transformer model with losses.
    
    Models utility transformer connecting medium-voltage grid
    (13.8 kV or 34.5 kV) to facility distribution voltage (480V or 600V).
    
    Equivalent circuit includes:
    - Leakage impedance (series)
    - Magnetizing reactance (shunt)
    - Core loss resistance (shunt)
    """
    
    def __init__(self, specs: TransformerSpecs):
        """Initialize transformer model.
        
        Args:
            specs: Transformer specifications
        """
        self.specs = specs
        self.turns_ratio = specs.V_primary / specs.V_secondary
        
        # Convert per-unit to actual values
        Z_base = specs.V_secondary**2 / specs.S_rated
        self.Z_leakage = specs.Z_leakage_pu * Z_base
        self.X_M = specs.X_M_pu * Z_base
        
        # Core loss resistance
        self.R_core = specs.V_secondary**2 / (specs.core_loss_fraction * specs.S_rated)
    
    def calculate_voltage_regulation(self, I_load: float, pf: float) -> Tuple[float, float]:
        """Calculate voltage drop and efficiency under load.
        
        Args:
            I_load: Load current (A)
            pf: Power factor (0-1)
            
        Returns:
            voltage_drop: Voltage drop magnitude (V)
            efficiency: Transformer efficiency (0-1)
        """
        # Voltage drop across leakage impedance
        # Assume resistive and reactive components
        R_leak = abs(self.Z_leakage) * 0.1  # Typical R/X ratio ~ 0.1
        X_leak = abs(self.Z_leakage) * np.sqrt(1 - 0.1**2)
        
        # Voltage drop components
        V_drop_R = I_load * R_leak * pf
        V_drop_X = I_load * X_leak * np.sqrt(1 - pf**2)
        voltage_drop = np.sqrt(V_drop_R**2 + V_drop_X**2)
        
        # Losses
        P_core = self.specs.V_secondary**2 / self.R_core
        P_copper = I_load**2 * R_leak
        
        # Efficiency
        P_out = self.specs.V_secondary * I_load * pf
        if P_out > 0:
            efficiency = P_out / (P_out + P_core + P_copper)
        else:
            efficiency = 0.0
        
        return voltage_drop, efficiency
    
    def calculate_losses(self, I_load: float) -> Tuple[float, float, float]:
        """Calculate transformer losses.
        
        Args:
            I_load: Load current (A)
            
        Returns:
            P_core: Core losses (W)
            P_copper: Copper losses (W)
            P_total: Total losses (W)
        """
        R_leak = abs(self.Z_leakage) * 0.1
        
        P_core = self.specs.V_secondary**2 / self.R_core
        P_copper = I_load**2 * R_leak
        P_total = P_core + P_copper
        
        return P_core, P_copper, P_total


# Typical transformer specifications for AI data centers
TRANSFORMER_13_8KV_TO_480V = TransformerSpecs(
    name="13.8kV/480V 5MVA",
    S_rated=5e6,           # 5 MVA
    V_primary=13800,       # 13.8 kV
    V_secondary=480,       # 480 V
    Z_leakage_pu=0.06,     # 6% leakage impedance
    X_M_pu=50.0,           # High magnetizing reactance
    core_loss_fraction=0.005  # 0.5% core loss
)

TRANSFORMER_34_5KV_TO_480V = TransformerSpecs(
    name="34.5kV/480V 10MVA",
    S_rated=10e6,          # 10 MVA
    V_primary=34500,       # 34.5 kV
    V_secondary=480,       # 480 V
    Z_leakage_pu=0.065,    # 6.5% leakage impedance
    X_M_pu=60.0,
    core_loss_fraction=0.004
)
