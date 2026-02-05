"""Immersion Cooling System Modeling.

Mathematical models for single-phase and two-phase immersion cooling
systems used in high-density AI data centers.

Key equations:
    Fluid: rho*c_p*(dT/dt + u.nabla(T)) = k_f*nabla^2(T)
    Solid: rho_s*c_ps*dT/dt = nabla.(k_s*nabla(T)) + q_gen
    Interface: T_f|_Gamma = T_s|_Gamma (continuity)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


class CoolantType(Enum):
    """Dielectric coolant types."""
    EC100 = "ec100"      # 3M Novec/EC-100
    FC72 = "fc72"        # 3M Fluorinert
    SINGLEPHASE = "single_phase"
    TWOPHASE = "two_phase"


@dataclass
class CoolantProperties:
    """Thermophysical properties of dielectric coolants."""
    name: str
    density: float        # kg/m^3
    specific_heat: float  # J/(kg*K)
    thermal_conductivity: float  # W/(m*K)
    viscosity: float      # Pa*s
    prandtl: float        # Prandtl number
    boiling_point: float  # Celsius (for two-phase)
    latent_heat: float    # J/kg (for two-phase)
    
    @property
    def thermal_diffusivity(self) -> float:
        return self.thermal_conductivity / (self.density * self.specific_heat)


# Pre-defined coolant properties
EC100_PROPERTIES = CoolantProperties(
    name="3M EC-100",
    density=1510.0,
    specific_heat=1100.0,
    thermal_conductivity=0.063,
    viscosity=0.00077,
    prandtl=13.4,
    boiling_point=61.0,
    latent_heat=88000.0
)

NOVEC_7100_PROPERTIES = CoolantProperties(
    name="3M Novec 7100",
    density=1510.0,
    specific_heat=1183.0,
    thermal_conductivity=0.069,
    viscosity=0.00058,
    prandtl=9.9,
    boiling_point=61.0,
    latent_heat=112000.0
)


@dataclass
class ThermalResult:
    """Results from thermal analysis."""
    T_junction_max: float  # Max junction temperature (C)
    T_junction_mean: float  # Mean junction temperature (C)
    T_coolant_out: float   # Coolant outlet temperature (C)
    P_cooling: float       # Cooling system power (W)
    thermal_resistance: float  # Total thermal resistance (K/W)
    heat_transfer_coeff: float  # Average HTC (W/m^2/K)
    pPUE: float           # Partial PUE


class ThermalResistanceNetwork:
    """Thermal resistance network model.
    
    Models the temperature rise from junction to ambient:
        T_j = T_ambient + P_GPU * (R_jc + R_ch + R_ha)
    
    where:
        R_jc: junction-to-case (die/package)
        R_ch: case-to-heatsink (TIM)
        R_ha: heatsink-to-ambient (convection)
    """
    
    def __init__(self,
                 R_jc: float = 0.15,    # K/W
                 R_ch: float = 0.08,    # K/W
                 R_ha: float = 0.05):   # K/W
        self.R_jc = R_jc
        self.R_ch = R_ch
        self.R_ha = R_ha
    
    @property
    def total_resistance(self) -> float:
        return self.R_jc + self.R_ch + self.R_ha
    
    def junction_temperature(self,
                            power: float,
                            T_ambient: float) -> float:
        return T_ambient + power * self.total_resistance


class ImmersionCoolingSystem:
    """Single-phase immersion cooling system model.
    
    Models heat transfer in immersion tanks using:
    - Forced convection correlations
    - Thermal resistance networks
    - Energy balance equations
    
    Achieves pPUE ~ 1.02 through direct liquid contact.
    """
    
    def __init__(self,
                 coolant: CoolantProperties = EC100_PROPERTIES,
                 flow_rate: float = 2.5,    # L/min per GPU
                 inlet_temp: float = 35.0,   # Celsius
                 tank_volume: float = 500.0):  # Liters
        
        self.coolant = coolant
        self.flow_rate = flow_rate / 60.0 / 1000.0  # Convert to m^3/s
        self.inlet_temp = inlet_temp
        self.tank_volume = tank_volume / 1000.0  # Convert to m^3
        
        # Thermal resistance network for GPU
        self.thermal_network = ThermalResistanceNetwork(
            R_jc=0.12,  # Optimized for immersion
            R_ch=0.05,  # Direct contact, no TIM paste
            R_ha=0.03   # Liquid convection
        )
        
        # Pump characteristics
        self.pump_efficiency = 0.65
        self.pressure_drop_pa = 15000  # Pa
        
    def compute_heat_transfer_coefficient(self,
                                          velocity: float,
                                          characteristic_length: float = 0.05
                                          ) -> float:
        """Calculate convective HTC using Nusselt correlations.
        
        For forced convection over GPU heatsink:
            Nu = 0.45 * Re^0.43
            h = Nu * k / L
        """
        # Reynolds number
        Re = (self.coolant.density * velocity * characteristic_length / 
              self.coolant.viscosity)
        
        # Nusselt correlation for immersion cooling
        Nu = 0.45 * (Re ** 0.43)
        
        # Heat transfer coefficient
        h = Nu * self.coolant.thermal_conductivity / characteristic_length
        
        return h
    
    def compute_coolant_temperature_rise(self,
                                         power: float,
                                         flow_rate: float) -> float:
        """Calculate coolant temperature rise.
        
        Energy balance: Q = m_dot * c_p * delta_T
        """
        m_dot = self.coolant.density * flow_rate
        delta_T = power / (m_dot * self.coolant.specific_heat)
        return delta_T
    
    def compute_pump_power(self, flow_rate: float) -> float:
        """Calculate pump power requirement.
        
        P_pump = (Q * delta_P) / eta
        """
        return (flow_rate * self.pressure_drop_pa) / self.pump_efficiency
    
    def analyze(self,
                power_profile: 'np.ndarray',
                num_gpus: int = 8) -> ThermalResult:
        """Analyze thermal performance for given power profile.
        
        Args:
            power_profile: Array of GPU power values (Watts)
            num_gpus: Number of GPUs in the system
            
        Returns:
            ThermalResult with thermal analysis
        """
        # Handle both array and scalar input
        if isinstance(power_profile, np.ndarray):
            mean_power = np.mean(power_profile)
            peak_power = np.max(power_profile)
        else:
            mean_power = power_profile
            peak_power = power_profile
        
        total_power = peak_power * num_gpus
        
        # Flow velocity estimate
        cross_section = 0.01  # m^2 approximate flow area
        velocity = self.flow_rate / cross_section
        
        # Heat transfer coefficient
        h = self.compute_heat_transfer_coefficient(velocity)
        
        # Update convective resistance based on HTC
        A_surface = 0.02  # m^2 GPU surface area
        R_conv = 1.0 / (h * A_surface)
        self.thermal_network.R_ha = R_conv
        
        # Junction temperatures
        T_j_peak = self.thermal_network.junction_temperature(
            peak_power, self.inlet_temp
        )
        T_j_mean = self.thermal_network.junction_temperature(
            mean_power, self.inlet_temp
        )
        
        # Coolant outlet temperature
        delta_T_coolant = self.compute_coolant_temperature_rise(
            total_power, self.flow_rate * num_gpus
        )
        T_coolant_out = self.inlet_temp + delta_T_coolant
        
        # Cooling system power (pumps + CDU)
        P_pump = self.compute_pump_power(self.flow_rate * num_gpus)
        P_cdu = total_power * 0.01  # ~1% for CDU
        P_cooling = P_pump + P_cdu
        
        # Partial PUE
        pPUE = (total_power + P_cooling) / total_power
        
        return ThermalResult(
            T_junction_max=T_j_peak,
            T_junction_mean=T_j_mean,
            T_coolant_out=T_coolant_out,
            P_cooling=P_cooling,
            thermal_resistance=self.thermal_network.total_resistance,
            heat_transfer_coeff=h,
            pPUE=pPUE
        )


class TwoPhaseImmersionSystem(ImmersionCoolingSystem):
    """Two-phase immersion cooling with boiling heat transfer.
    
    Enhanced heat transfer through phase change:
        h_boiling >> h_single_phase
    
    Uses pool boiling correlations modified for dielectric fluids.
    """
    
    def compute_heat_transfer_coefficient(self,
                                          heat_flux: float,
                                          T_surface: float) -> float:
        """Calculate boiling HTC using Rohsenow correlation.
        
        For nucleate pool boiling:
            h = C_sf * (c_p * delta_T_sat / (h_fg * Pr^n))^3 * ...
        """
        T_sat = self.coolant.boiling_point
        delta_T = T_surface - T_sat
        
        if delta_T <= 0:
            # Single-phase regime
            return super().compute_heat_transfer_coefficient(0.1)
        
        # Simplified boiling correlation
        C_sf = 0.013  # Surface-fluid coefficient
        n = 1.0
        
        # Boiling HTC (simplified)
        h_boiling = 5000 * (heat_flux / 10000) ** 0.7
        
        return min(h_boiling, 15000)  # Cap at 15000 W/m^2/K


if __name__ == "__main__":
    # Example: Analyze 8x H100 GPU system
    cooling = ImmersionCoolingSystem(
        coolant=EC100_PROPERTIES,
        flow_rate=2.5,
        inlet_temp=35.0
    )
    
    # Simulate power profile (700W per GPU)
    power = np.full(1000, 650.0)  # Mean 650W
    
    result = cooling.analyze(power, num_gpus=8)
    
    print(f"Max junction temp: {result.T_junction_max:.1f} C")
    print(f"Coolant outlet: {result.T_coolant_out:.1f} C")
    print(f"Cooling power: {result.P_cooling:.1f} W")
    print(f"pPUE: {result.pPUE:.3f}")
