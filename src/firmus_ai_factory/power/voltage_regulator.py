"""Voltage regulator module (VRM) models.

This module implements multi-phase voltage regulators for GPU power delivery
with output impedance analysis and transient response simulation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class VRMSpecs:
    """Multi-phase VRM specifications."""
    name: str
    num_phases: int         # Number of converter phases
    V_in: float             # Input voltage (V)
    V_out: float            # Output voltage (V)
    L_phase: float          # Per-phase inductance (H)
    C_out: float            # Output capacitance (F)
    R_L: float              # Inductor ESR (Ω)
    R_C: float              # Capacitor ESR (Ω)
    f_sw: float             # Switching frequency (Hz)
    I_max_phase: float      # Max current per phase (A)


class MultiphaseVRM:
    """Multi-phase voltage regulator for GPU power delivery.
    
    Models point-of-load voltage regulators with:
    - Interleaved multi-phase operation
    - Reduced output ripple
    - Fast transient response
    - Current sharing between phases
    """
    
    def __init__(self, specs: VRMSpecs):
        """Initialize multi-phase VRM.
        
        Args:
            specs: VRM specifications
        """
        self.specs = specs
        self.phase_shift = 360 / specs.num_phases  # Degrees
        self.duty_cycle = specs.V_out / specs.V_in
    
    def calculate_output_impedance(self, freq: np.ndarray) -> np.ndarray:
        """Calculate output impedance vs frequency.
        
        Multi-phase operation reduces effective inductance by phase count.
        
        Args:
            freq: Frequency array (Hz)
            
        Returns:
            Z_out: Output impedance magnitude (Ω)
        """
        omega = 2 * np.pi * freq
        
        # Effective inductance (reduced by phase count)
        L_eff = self.specs.L_phase / self.specs.num_phases
        
        # Impedance components
        Z_L = 1j * omega * L_eff + self.specs.R_L / self.specs.num_phases
        Z_C = 1 / (1j * omega * self.specs.C_out) + self.specs.R_C
        
        # Parallel combination
        Z_out = (Z_L * Z_C) / (Z_L + Z_C)
        
        return np.abs(Z_out)
    
    def calculate_ripple_current(self) -> float:
        """Calculate output ripple current.
        
        Returns:
            I_ripple_rms: RMS ripple current (A)
        """
        # Ripple current per phase
        delta_I_phase = (self.specs.V_in - self.specs.V_out) * self.duty_cycle / \
                       (self.specs.L_phase * self.specs.f_sw)
        
        # RMS ripple (reduced by interleaving)
        I_ripple_rms = delta_I_phase / (np.sqrt(12) * np.sqrt(self.specs.num_phases))
        
        return I_ripple_rms
    
    def calculate_efficiency(self, I_out: float) -> float:
        """Calculate VRM efficiency.
        
        Args:
            I_out: Output current (A)
            
        Returns:
            efficiency: VRM efficiency (0-1)
        """
        if I_out <= 0:
            return 0.0
        
        # Current per phase
        I_phase = I_out / self.specs.num_phases
        
        # Conduction losses
        I_L_rms = I_phase / np.sqrt(1 - self.duty_cycle)
        P_cond_phase = I_L_rms**2 * self.specs.R_L
        P_cond_total = P_cond_phase * self.specs.num_phases
        
        # Capacitor ESR loss
        I_ripple = self.calculate_ripple_current()
        P_cap = I_ripple**2 * self.specs.R_C
        
        # Switching losses (per phase)
        t_sw = 1e-9  # Assume 1ns switching time
        P_sw_phase = 0.5 * self.specs.V_in * I_phase * self.specs.f_sw * t_sw
        P_sw_total = P_sw_phase * self.specs.num_phases
        
        # Output power
        P_out = self.specs.V_out * I_out
        
        # Total losses
        P_loss = P_cond_total + P_cap + P_sw_total
        
        # Efficiency
        efficiency = P_out / (P_out + P_loss)
        return min(efficiency, 1.0)
    
    def simulate_load_transient(self, I_initial: float, I_final: float,
                               t_rise: float = 1e-9) -> Tuple[np.ndarray, np.ndarray, float]:
        """Simulate voltage droop during load transient.
        
        Args:
            I_initial: Initial load current (A)
            I_final: Final load current (A)
            t_rise: Current rise time (s)
            
        Returns:
            t: Time array (s)
            v_out: Output voltage array (V)
            v_droop_max: Maximum voltage droop (V)
        """
        # Simulation parameters
        t_sim = 10 * t_rise  # Simulate 10x rise time
        dt = t_rise / 100    # 100 points during rise
        n_steps = int(t_sim / dt)
        
        # Arrays
        t = np.linspace(0, t_sim, n_steps)
        i_load = np.zeros(n_steps)
        v_out = np.zeros(n_steps)
        
        # Load current profile (linear rise)
        for k in range(n_steps):
            if t[k] < t_rise:
                i_load[k] = I_initial + (I_final - I_initial) * (t[k] / t_rise)
            else:
                i_load[k] = I_final
        
        # Initial voltage
        v_out[0] = self.specs.V_out
        
        # Effective inductance and capacitance
        L_eff = self.specs.L_phase / self.specs.num_phases
        C_eff = self.specs.C_out
        R_eff = self.specs.R_C + self.specs.R_L / self.specs.num_phases
        
        # Simple LC response
        for k in range(1, n_steps):
            # Voltage drop across ESR
            v_esr = i_load[k] * R_eff
            
            # Capacitor discharge during transient
            if k > 1:
                di_dt = (i_load[k] - i_load[k-1]) / dt
                v_L = L_eff * di_dt
                
                # Voltage droop
                v_out[k] = self.specs.V_out - v_esr - v_L
            else:
                v_out[k] = self.specs.V_out - v_esr
        
        # Maximum droop
        v_droop_max = self.specs.V_out - np.min(v_out)
        
        return t, v_out, v_droop_max
    
    def check_target_impedance(self, freq_range: Tuple[float, float], 
                              Z_target: float) -> bool:
        """Check if output impedance meets target specification.
        
        Args:
            freq_range: Frequency range to check (f_min, f_max) in Hz
            Z_target: Target impedance (Ω)
            
        Returns:
            meets_spec: True if impedance < Z_target across frequency range
        """
        freq = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), 1000)
        Z_out = self.calculate_output_impedance(freq)
        
        return np.all(Z_out <= Z_target)


# Typical VRM specifications for NVIDIA GPUs

# H100 SXM VRM (12V to 1.0V, 700W)
VRM_H100_SXM = VRMSpecs(
    name="H100 SXM VRM",
    num_phases=8,
    V_in=12.0,
    V_out=1.0,
    L_phase=0.22e-6,       # 0.22 μH per phase
    C_out=4400e-6,         # 4400 μF total output capacitance
    R_L=0.15e-3,           # 0.15 mΩ per phase
    R_C=0.08e-3,           # 0.08 mΩ total ESR
    f_sw=600e3,            # 600 kHz
    I_max_phase=90         # 90A per phase (720A total)
)

# B200 VRM (12V to 0.85V, 1000W)
VRM_B200 = VRMSpecs(
    name="B200 VRM",
    num_phases=12,
    V_in=12.0,
    V_out=0.85,
    L_phase=0.18e-6,       # 0.18 μH per phase
    C_out=6600e-6,         # 6600 μF total
    R_L=0.12e-3,           # 0.12 mΩ per phase
    R_C=0.06e-3,           # 0.06 mΩ total ESR
    f_sw=800e3,            # 800 kHz
    I_max_phase=100        # 100A per phase (1200A total)
)
