"""DC-DC converter models for power delivery network.

This module implements buck converter models with state-space dynamics,
efficiency calculations, and transient response analysis.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy import signal


@dataclass
class BuckConverterSpecs:
    """Buck converter specifications."""
    name: str
    V_in: float             # Input voltage (V)
    V_out: float            # Output voltage (V)
    L: float                # Inductance (H)
    C: float                # Capacitance (F)
    R_L: float              # Inductor ESR (Ω)
    R_C: float              # Capacitor ESR (Ω)
    f_sw: float             # Switching frequency (Hz)
    I_max: float            # Maximum output current (A)


class BuckConverterModel:
    """Buck converter with current-mode control.
    
    Models DC-DC buck converter for voltage step-down with:
    - State-space dynamics
    - Efficiency calculations
    - Output impedance analysis
    - Transient response simulation
    """
    
    def __init__(self, specs: BuckConverterSpecs):
        """Initialize buck converter model.
        
        Args:
            specs: Converter specifications
        """
        self.specs = specs
        self.duty_cycle = specs.V_out / specs.V_in
        
        # Validate duty cycle
        if not (0 < self.duty_cycle < 1):
            raise ValueError(f"Invalid duty cycle: {self.duty_cycle}")
    
    def state_space_model(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate state-space representation.
        
        State vector: x = [i_L, v_C]
        Input: u = [v_in, i_load]
        Output: y = v_out
        
        Returns:
            A: State matrix
            B: Input matrix
            C: Output matrix
            D: Feedthrough matrix
        """
        # State matrix
        A = np.array([
            [-self.specs.R_L/self.specs.L, -1/self.specs.L],
            [1/self.specs.C, -1/(self.specs.C*self.specs.R_C)]
        ])
        
        # Input matrix
        B = np.array([
            [self.duty_cycle/self.specs.L, 0],
            [0, -1/self.specs.C]
        ])
        
        # Output matrix
        C = np.array([[self.specs.R_C/self.specs.C, 1]])
        
        # Feedthrough matrix
        D = np.array([[0, -self.specs.R_C]])
        
        return A, B, C, D
    
    def calculate_efficiency(self, I_out: float) -> float:
        """Calculate converter efficiency.
        
        Args:
            I_out: Output current (A)
            
        Returns:
            efficiency: Converter efficiency (0-1)
        """
        if I_out <= 0:
            return 0.0
        
        # Conduction losses
        I_L_rms = I_out / np.sqrt(1 - self.duty_cycle)
        P_cond = I_L_rms**2 * self.specs.R_L + I_out**2 * self.specs.R_C
        
        # Switching losses (simplified model)
        # Assume 1ns rise/fall time
        t_sw = 1e-9
        P_sw = 0.5 * self.specs.V_in * I_out * self.specs.f_sw * t_sw
        
        # Output power
        P_out = self.specs.V_out * I_out
        
        # Efficiency
        efficiency = P_out / (P_out + P_cond + P_sw)
        return min(efficiency, 1.0)
    
    def calculate_output_impedance(self, freq: np.ndarray) -> np.ndarray:
        """Calculate output impedance vs frequency.
        
        Args:
            freq: Frequency array (Hz)
            
        Returns:
            Z_out: Output impedance (Ω)
        """
        omega = 2 * np.pi * freq
        
        # Parallel combination of L and C
        Z_L = 1j * omega * self.specs.L + self.specs.R_L
        Z_C = 1 / (1j * omega * self.specs.C) + self.specs.R_C
        
        Z_out = (Z_L * Z_C) / (Z_L + Z_C)
        return Z_out
    
    def simulate_load_step(self, I_step: float, t_sim: float = 100e-6, 
                          dt: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate transient response to load step.
        
        Args:
            I_step: Current step magnitude (A)
            t_sim: Simulation time (s)
            dt: Time step (s)
            
        Returns:
            t: Time array (s)
            v_out: Output voltage array (V)
        """
        n_steps = int(t_sim / dt)
        
        # State variables
        i_L = np.zeros(n_steps)
        v_C = np.zeros(n_steps)
        v_out = np.zeros(n_steps)
        
        # Initial conditions (steady state)
        i_L[0] = 0.0
        v_C[0] = self.specs.V_out
        v_out[0] = self.specs.V_out
        
        # Load current
        i_load = np.zeros(n_steps)
        i_load[int(n_steps/2):] = I_step  # Step at midpoint
        
        # Get state-space matrices
        A, B, C, D = self.state_space_model()
        
        # Simulation loop (simple Euler integration)
        for k in range(1, n_steps):
            x = np.array([i_L[k-1], v_C[k-1]])
            u = np.array([self.specs.V_in, i_load[k]])
            
            # State derivative
            dx_dt = A @ x + B @ u
            
            # Update state
            x_new = x + dx_dt * dt
            i_L[k] = x_new[0]
            v_C[k] = x_new[1]
            
            # Output
            v_out[k] = (C @ x_new + D @ u)[0]
        
        t = np.arange(n_steps) * dt
        return t, v_out


# Typical converter specifications for AI data centers

# 480V to 12V converter (rack-level)
BUCK_480V_TO_12V = BuckConverterSpecs(
    name="480V/12V 100kW",
    V_in=480.0,
    V_out=12.0,
    L=10e-6,               # 10 μH
    C=1000e-6,             # 1000 μF
    R_L=1e-3,              # 1 mΩ
    R_C=0.5e-3,            # 0.5 mΩ
    f_sw=100e3,            # 100 kHz
    I_max=8333             # 100 kW / 12V
)

# 12V to 1V converter (GPU-level)
BUCK_12V_TO_1V = BuckConverterSpecs(
    name="12V/1V 1.5kW",
    V_in=12.0,
    V_out=1.0,
    L=0.3e-6,              # 0.3 μH
    C=2200e-6,             # 2200 μF
    R_L=0.2e-3,            # 0.2 mΩ
    R_C=0.1e-3,            # 0.1 mΩ
    f_sw=500e3,            # 500 kHz
    I_max=1500             # 1.5 kW / 1V
)
