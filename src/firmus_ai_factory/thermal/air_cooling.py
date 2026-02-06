"""Air Cooling System Modeling.

Mathematical models for forced-air cooling systems used in
traditional and hybrid AI data center deployments.

In NVL72 systems, air cooling handles low-power peripherals
(NICs, storage, BMC) while liquid cooling handles GPUs.

Key equations:
    Q = m_dot_air * c_p_air * delta_T
    h_air = Nu * k_air / L_c
    Nu = 0.023 * Re^0.8 * Pr^0.4  (Dittus-Boelter)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


# Air properties at ~35C (typical data center intake)
AIR_DENSITY = 1.145          # kg/m^3
AIR_SPECIFIC_HEAT = 1005.0   # J/(kg*K)
AIR_CONDUCTIVITY = 0.0271    # W/(m*K)
AIR_VISCOSITY = 1.895e-5     # Pa*s
AIR_PRANDTL = 0.707


@dataclass
class AirCoolingResult:
    """Results from air cooling analysis."""
    T_outlet: float             # Air outlet temperature (C)
    T_component_max: float      # Max component temperature (C)
    P_fan: float                # Fan power consumption (W)
    airflow_cfm: float          # Airflow in CFM
    heat_transfer_coeff: float  # Average HTC (W/m^2/K)
    pPUE_contribution: float    # PUE contribution from air cooling


class AirCoolingSystem:
    """Forced-air cooling system model.
    
    Models traditional air-cooled data center cooling using:
    - Fan laws and airflow calculations
    - Forced convection correlations (Dittus-Boelter)
    - Energy balance for air temperature rise
    - Fan power scaling with airflow^3
    
    Used for:
    - Standalone air-cooled GPU racks (up to ~40kW)
    - Peripheral cooling in hybrid liquid/air NVL72 racks
    - CRAH/CRAC unit modeling
    """
    
    def __init__(self,
                 inlet_temp: float = 27.0,         # ASHRAE A1 recommended
                 max_outlet_temp: float = 45.0,
                 fan_efficiency: float = 0.70,
                 num_fans: int = 6,
                 fan_max_cfm: float = 200.0):       # CFM per fan
        
        self.inlet_temp = inlet_temp
        self.max_outlet_temp = max_outlet_temp
        self.fan_efficiency = fan_efficiency
        self.num_fans = num_fans
        self.fan_max_cfm = fan_max_cfm
        self.total_max_cfm = num_fans * fan_max_cfm
        
        # Typical pressure drop across server chassis
        self.pressure_drop_pa = 250.0  # Pa
    
    def cfm_to_m3s(self, cfm: float) -> float:
        """Convert CFM to m^3/s."""
        return cfm * 0.000471947
    
    def required_airflow(self, power_watts: float) -> float:
        """Calculate required airflow (CFM) for given heat load.
        
        Q = m_dot * c_p * delta_T
        m_dot = Q / (c_p * delta_T)
        """
        delta_T = self.max_outlet_temp - self.inlet_temp
        m_dot = power_watts / (AIR_SPECIFIC_HEAT * delta_T)  # kg/s
        vol_flow = m_dot / AIR_DENSITY  # m^3/s
        return vol_flow / 0.000471947  # Convert to CFM
    
    def compute_fan_power(self, airflow_cfm: float) -> float:
        """Calculate fan power using fan affinity laws.
        
        P_fan = (Q * delta_P) / eta
        Fan power scales with flow^3: P ~ (Q/Q_max)^3
        """
        vol_flow = self.cfm_to_m3s(airflow_cfm)
        # Ideal power
        p_ideal = vol_flow * self.pressure_drop_pa
        # Account for efficiency
        p_fan = p_ideal / self.fan_efficiency
        # Scale all fans
        return p_fan * self.num_fans * (airflow_cfm / self.total_max_cfm)
    
    def compute_heat_transfer_coefficient(self,
                                          velocity: float,
                                          characteristic_length: float = 0.04
                                          ) -> float:
        """Calculate air-side HTC using Dittus-Boelter correlation.
        
        Nu = 0.023 * Re^0.8 * Pr^0.4
        h = Nu * k / L_c
        """
        Re = AIR_DENSITY * velocity * characteristic_length / AIR_VISCOSITY
        Nu = 0.023 * (Re ** 0.8) * (AIR_PRANDTL ** 0.4)
        h = Nu * AIR_CONDUCTIVITY / characteristic_length
        return h
    
    def analyze(self,
                power_watts: float,
                component_thermal_resistance: float = 0.5  # K/W
                ) -> AirCoolingResult:
        """Analyze air cooling performance.
        
        Args:
            power_watts: Total heat load (W)
            component_thermal_resistance: R_theta from junction to air (K/W)
            
        Returns:
            AirCoolingResult with thermal analysis
        """
        # Required airflow
        cfm_required = self.required_airflow(power_watts)
        cfm_actual = min(cfm_required, self.total_max_cfm)
        
        # Actual temperature rise
        vol_flow = self.cfm_to_m3s(cfm_actual)
        m_dot = AIR_DENSITY * vol_flow
        delta_T = power_watts / (m_dot * AIR_SPECIFIC_HEAT) if m_dot > 0 else 999
        T_outlet = self.inlet_temp + delta_T
        
        # Component temperature
        T_component = T_outlet + power_watts * component_thermal_resistance
        
        # Fan power
        P_fan = self.compute_fan_power(cfm_actual)
        
        # Air velocity estimate (through heatsink fins)
        flow_area = 0.02  # m^2 approximate
        velocity = vol_flow / flow_area if flow_area > 0 else 0
        h = self.compute_heat_transfer_coefficient(velocity)
        
        # PUE contribution
        pPUE_contribution = (power_watts + P_fan) / power_watts if power_watts > 0 else 1.0
        
        return AirCoolingResult(
            T_outlet=T_outlet,
            T_component_max=T_component,
            P_fan=P_fan,
            airflow_cfm=cfm_actual,
            heat_transfer_coeff=h,
            pPUE_contribution=pPUE_contribution
        )


class NVL72PeripheralAirCooling(AirCoolingSystem):
    """Air cooling for NVL72 rack peripherals.
    
    In GB200/GB300 NVL72 racks, low-power components are air-cooled
    using 40mm fans while GPUs use direct-to-chip liquid cooling.
    
    Peripheral components:
    - Network Interface Cards (NICs)
    - NVMe/E1.S storage drives
    - BMC and management controllers
    - Power shelf controllers
    """
    
    def __init__(self,
                 inlet_temp: float = 27.0,
                 num_fans: int = 12,
                 fan_max_cfm: float = 15.0,   # Small 40mm fans
                 peripheral_power: float = 2000.0):  # Total peripheral W
        
        super().__init__(
            inlet_temp=inlet_temp,
            max_outlet_temp=50.0,  # Higher limit for peripherals
            fan_efficiency=0.50,   # Small fans less efficient
            num_fans=num_fans,
            fan_max_cfm=fan_max_cfm
        )
        self.peripheral_power = peripheral_power
        self.pressure_drop_pa = 150.0  # Lower for open peripheral area
    
    def analyze_peripherals(self) -> AirCoolingResult:
        """Analyze peripheral air cooling for NVL72 rack."""
        return self.analyze(
            power_watts=self.peripheral_power,
            component_thermal_resistance=0.8  # Higher for small components
        )


if __name__ == "__main__":
    # Example 1: Traditional air-cooled rack (40kW)
    air = AirCoolingSystem(inlet_temp=27.0)
    result = air.analyze(power_watts=40000)
    print(f"Air-cooled 40kW rack:")
    print(f"  Outlet temp: {result.T_outlet:.1f} C")
    print(f"  Fan power: {result.P_fan:.0f} W")
    print(f"  Airflow: {result.airflow_cfm:.0f} CFM")
    
    # Example 2: NVL72 peripheral cooling
    peripheral = NVL72PeripheralAirCooling(peripheral_power=2000)
    result2 = peripheral.analyze_peripherals()
    print(f"\nNVL72 peripheral air cooling:")
    print(f"  Outlet temp: {result2.T_outlet:.1f} C")
    print(f"  Fan power: {result2.P_fan:.0f} W")
