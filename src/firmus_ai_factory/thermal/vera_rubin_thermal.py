"""Vera Rubin NVL72 Rack-Level Thermal Model.

Implements the thermal model from VR_NVL72_RackLevelThermalModel_v07.pdf
for 100% liquid-cooled compute with air-cooled power shelves.

Key specifications:
    - 18 Compute Trays + 9 NVLink Switch Trays
    - PG25 coolant (25% Propylene Glycol)
    - Supply inlet: 17°C to 45°C
    - Return outlet: max 65°C
    - Operating pressure: max 72 psig
    - Burst pressure: 217 psig
    - Filtration: 25 μm

Flow equations (from NVIDIA thermal model):
    MaxP: lpm = (1.737E4 / (5.72E1 - T_inlet))^0.796
    MaxQ: lpm = (2.288E4 / (5.739E1 - T_inlet))^0.718

Cooling region: Australia (Benmax HCU2500 air-liquid CDU)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
from enum import Enum


class CoolantSpec(Enum):
    """Coolant specifications."""
    PG25 = "pg25"   # 25% Propylene Glycol / 75% Water
    WATER = "water"


@dataclass
class PG25Properties:
    """Thermophysical properties of PG25 coolant at various temperatures.
    
    PG25 = 25% Propylene Glycol / 75% Water by volume.
    Properties interpolated from manufacturer data.
    """
    
    @staticmethod
    def density(T_celsius: float) -> float:
        """Density in kg/m³ at given temperature."""
        return 1032.0 - 0.35 * T_celsius
    
    @staticmethod
    def specific_heat(T_celsius: float) -> float:
        """Specific heat in J/(kg·K) at given temperature."""
        return 3850.0 + 1.5 * T_celsius
    
    @staticmethod
    def thermal_conductivity(T_celsius: float) -> float:
        """Thermal conductivity in W/(m·K) at given temperature."""
        return 0.45 + 0.001 * T_celsius
    
    @staticmethod
    def viscosity(T_celsius: float) -> float:
        """Dynamic viscosity in Pa·s at given temperature."""
        return 0.003 * np.exp(-0.02 * T_celsius)


@dataclass
class VRThermalLimits:
    """Vera Rubin NVL72 thermal operating limits.
    
    Source: VR_NVL72_RackLevelThermalModel_v07.pdf, p10
    """
    supply_inlet_min_c: float = 17.0
    supply_inlet_max_c: float = 45.0
    return_outlet_max_c: float = 65.0
    operating_pressure_max_psig: float = 72.0
    burst_pressure_psig: float = 217.0
    filtration_um: float = 25.0
    
    # ASHRAE A3 air envelope for power shelves
    air_temp_min_c: float = 5.0
    air_temp_max_c: float = 40.0
    air_rh_min_pct: float = 8.0
    air_rh_max_pct: float = 85.0
    max_altitude_m: float = 3050.0
    
    # Altitude derating
    altitude_derate_c_per_175m: float = 1.0
    altitude_derate_above_m: float = 900.0
    
    def max_air_temp_at_altitude(self, altitude_m: float) -> float:
        """Calculate derated max air temperature at altitude.
        
        Derate max allowable dry-bulb temp 1°C/175m above 900m.
        """
        if altitude_m <= self.altitude_derate_above_m:
            return self.air_temp_max_c
        excess_m = altitude_m - self.altitude_derate_above_m
        derate = excess_m / 175.0 * self.altitude_derate_c_per_175m
        return self.air_temp_max_c - derate


@dataclass
class VRFlowRequirement:
    """Flow requirement at a specific inlet temperature."""
    inlet_temp_c: float
    flowrate_lpm: float
    pressure_drop_psid: float
    lpm_per_kw: float


class VRNvl72ThermalModel:
    """Vera Rubin NVL72 Rack-Level Thermal Model.
    
    Implements the complete thermal model including:
    - Liquid cooling for compute and switch trays
    - Air cooling for power shelves
    - Flow rate calculations from NVIDIA empirical equations
    - Temperature distribution across rack components
    - Power shelf airflow requirements
    
    Attributes:
        rack_tdp_kw: Rack thermal design power (kW)
        power_mode: 'max_p' or 'max_q'
        limits: Thermal operating limits
    """
    
    def __init__(self, 
                 rack_tdp_kw: float = 227.0,
                 power_mode: str = "max_p"):
        """Initialize VR NVL72 thermal model.
        
        Args:
            rack_tdp_kw: Rack TDP in kW (227 for Max P, 187 for Max Q)
            power_mode: 'max_p' or 'max_q'
        """
        self.rack_tdp_kw = rack_tdp_kw
        self.power_mode = power_mode
        self.limits = VRThermalLimits()
        self.coolant = PG25Properties()
        
        # Rack liquid volumes (liters) from thermal model doc
        self.liquid_volumes = {
            'compute_tray': 0.778,      # per tray
            'switch_tray': 0.175,       # per tray
            'rack_manifold': 17.700,
            'rack_busbar': 0.311,
        }
        self.num_compute_trays = 18
        self.num_switch_trays = 9
        
        # Flow requirement lookup tables from NVIDIA thermal model
        self._flow_tables = self._init_flow_tables()
    
    def _init_flow_tables(self) -> Dict[str, List[VRFlowRequirement]]:
        """Initialize flow requirement tables from NVIDIA specifications."""
        max_p_table = [
            VRFlowRequirement(25.0, 150.0, 5.0, 0.7),
            VRFlowRequirement(30.0, 170.0, 6.0, 0.8),
            VRFlowRequirement(35.0, 200.0, 8.0, 1.0),
            VRFlowRequirement(40.0, 250.0, 12.0, 1.1),
            VRFlowRequirement(45.0, 320.0, 19.0, 1.5),
        ]
        
        max_q_table = [
            VRFlowRequirement(25.0, 120.0, 3.0, 0.7),
            VRFlowRequirement(30.0, 140.0, 4.0, 0.8),
            VRFlowRequirement(35.0, 160.0, 5.5, 1.0),
            VRFlowRequirement(40.0, 205.0, 8.5, 1.1),
            VRFlowRequirement(45.0, 280.0, 16.0, 1.5),
        ]
        
        return {
            'max_p': max_p_table,
            'max_q': max_q_table,
        }
    
    def required_flowrate_lpm(self, inlet_temp_c: float) -> float:
        """Calculate required flowrate using NVIDIA empirical equations.
        
        MaxP: lpm = (1.737E4 / (5.72E1 - T_inlet))^0.796
        MaxQ: lpm = (2.288E4 / (5.739E1 - T_inlet))^0.718
        
        Args:
            inlet_temp_c: Coolant supply inlet temperature (°C)
            
        Returns:
            Required flowrate in liters per minute
        """
        if inlet_temp_c < self.limits.supply_inlet_min_c:
            inlet_temp_c = self.limits.supply_inlet_min_c
        if inlet_temp_c > self.limits.supply_inlet_max_c:
            inlet_temp_c = self.limits.supply_inlet_max_c
        
        if self.power_mode == "max_p":
            return (1.737e4 / (57.2 - inlet_temp_c)) ** 0.796
        else:  # max_q
            return (2.288e4 / (57.39 - inlet_temp_c)) ** 0.718
    
    def required_flowrate_ls(self, inlet_temp_c: float) -> float:
        """Calculate required flowrate in liters per second."""
        return self.required_flowrate_lpm(inlet_temp_c) / 60.0
    
    def pressure_drop_psid(self, inlet_temp_c: float) -> float:
        """Estimate pressure drop across rack at given inlet temperature.
        
        Interpolated from NVIDIA flow requirement tables.
        
        Args:
            inlet_temp_c: Coolant supply inlet temperature (°C)
            
        Returns:
            Pressure drop in psid
        """
        table = self._flow_tables[self.power_mode]
        temps = [r.inlet_temp_c for r in table]
        pressures = [r.pressure_drop_psid for r in table]
        return float(np.interp(inlet_temp_c, temps, pressures))
    
    def pressure_drop_kpa(self, inlet_temp_c: float) -> float:
        """Pressure drop in kPa (1 psi = 6.895 kPa)."""
        return self.pressure_drop_psid(inlet_temp_c) * 6.895
    
    def coolant_temperature_rise(self, 
                                  inlet_temp_c: float,
                                  rack_power_kw: Optional[float] = None) -> float:
        """Calculate coolant temperature rise across rack.
        
        ΔT = Q / (ṁ × c_p)
        
        Args:
            inlet_temp_c: Coolant supply inlet temperature (°C)
            rack_power_kw: Rack power in kW (defaults to rack_tdp_kw)
            
        Returns:
            Temperature rise in °C
        """
        if rack_power_kw is None:
            rack_power_kw = self.rack_tdp_kw
        
        flow_lpm = self.required_flowrate_lpm(inlet_temp_c)
        flow_m3_s = flow_lpm / 60000.0
        
        # PG25 properties at mean temperature
        T_mean = inlet_temp_c + 10.0  # Estimate
        rho = self.coolant.density(T_mean)
        cp = self.coolant.specific_heat(T_mean)
        
        m_dot = rho * flow_m3_s  # kg/s
        delta_T = (rack_power_kw * 1000.0) / (m_dot * cp)
        
        return delta_T
    
    def outlet_temperature(self, 
                           inlet_temp_c: float,
                           rack_power_kw: Optional[float] = None) -> float:
        """Calculate coolant outlet temperature.
        
        Args:
            inlet_temp_c: Coolant supply inlet temperature (°C)
            rack_power_kw: Rack power in kW
            
        Returns:
            Outlet temperature in °C
        """
        delta_T = self.coolant_temperature_rise(inlet_temp_c, rack_power_kw)
        T_out = inlet_temp_c + delta_T
        return T_out
    
    def is_within_thermal_limits(self, 
                                 inlet_temp_c: float,
                                 rack_power_kw: Optional[float] = None) -> Tuple[bool, Dict[str, bool]]:
        """Check if operating conditions are within thermal limits.
        
        Args:
            inlet_temp_c: Coolant supply inlet temperature (°C)
            rack_power_kw: Rack power in kW
            
        Returns:
            Tuple of (overall_ok, dict of individual checks)
        """
        T_out = self.outlet_temperature(inlet_temp_c, rack_power_kw)
        
        checks = {
            'inlet_temp_min': inlet_temp_c >= self.limits.supply_inlet_min_c,
            'inlet_temp_max': inlet_temp_c <= self.limits.supply_inlet_max_c,
            'outlet_temp_max': T_out <= self.limits.return_outlet_max_c,
        }
        
        overall = all(checks.values())
        return overall, checks
    
    def power_shelf_airflow_cfm(self, air_inlet_temp_c: float) -> float:
        """Calculate required airflow for power shelves.
        
        Power shelves are air-cooled. Airflow requirements from thermal model.
        
        Args:
            air_inlet_temp_c: Air inlet temperature (°C)
            
        Returns:
            Required airflow in CFM
        """
        # Lookup tables from thermal model doc
        if self.power_mode == "max_p":
            temps = [20.0, 25.0, 30.0, 35.0, 40.0]
            cfms = [330.0, 440.0, 500.0, 670.0, 900.0]
        else:  # max_q
            temps = [20.0, 25.0, 30.0, 35.0, 40.0]
            cfms = [260.0, 350.0, 390.0, 520.0, 700.0]
        
        return float(np.interp(air_inlet_temp_c, temps, cfms))
    
    def total_liquid_volume_liters(self) -> float:
        """Calculate total liquid volume in rack (liters)."""
        vol = (self.num_compute_trays * self.liquid_volumes['compute_tray'] +
               self.num_switch_trays * self.liquid_volumes['switch_tray'] +
               self.liquid_volumes['rack_manifold'] +
               self.liquid_volumes['rack_busbar'])
        return vol
    
    def thermal_mass_j_per_k(self) -> float:
        """Calculate rack thermal mass (J/K) from liquid volume.
        
        Thermal mass = ρ × V × c_p
        """
        V_liters = self.total_liquid_volume_liters()
        V_m3 = V_liters / 1000.0
        T_avg = 40.0  # Assume average operating temperature
        rho = self.coolant.density(T_avg)
        cp = self.coolant.specific_heat(T_avg)
        return rho * V_m3 * cp
    
    def thermal_time_constant_s(self, inlet_temp_c: float = 35.0) -> float:
        """Calculate thermal time constant (seconds).
        
        τ = C_thermal / (ṁ × c_p)
        
        This represents how quickly the rack responds to power changes.
        """
        C_thermal = self.thermal_mass_j_per_k()
        flow_lpm = self.required_flowrate_lpm(inlet_temp_c)
        flow_m3_s = flow_lpm / 60000.0
        T_mean = inlet_temp_c + 10.0
        rho = self.coolant.density(T_mean)
        cp = self.coolant.specific_heat(T_mean)
        m_dot_cp = rho * flow_m3_s * cp
        
        if m_dot_cp > 0:
            return C_thermal / m_dot_cp
        return float('inf')
    
    def simulate_thermal_transient(self,
                                    power_profile_kw: np.ndarray,
                                    time_s: np.ndarray,
                                    inlet_temp_c: float = 35.0
                                    ) -> Dict[str, np.ndarray]:
        """Simulate thermal transient response to varying power.
        
        Uses lumped-parameter model:
            C_th * dT/dt = Q_in - ṁ*c_p*(T_out - T_in)
        
        Args:
            power_profile_kw: Time-varying rack power (kW)
            time_s: Time array (seconds)
            inlet_temp_c: Constant inlet temperature (°C)
            
        Returns:
            Dict with 'time', 'T_outlet', 'T_mean', 'flowrate_lpm'
        """
        C_th = self.thermal_mass_j_per_k()
        flow_lpm = self.required_flowrate_lpm(inlet_temp_c)
        flow_m3_s = flow_lpm / 60000.0
        
        T_mean_avg = inlet_temp_c + 10.0
        rho = self.coolant.density(T_mean_avg)
        cp = self.coolant.specific_heat(T_mean_avg)
        m_dot_cp = rho * flow_m3_s * cp
        
        n = len(time_s)
        T_out = np.zeros(n)
        T_mean = np.zeros(n)
        
        # Initial condition: steady state
        T_out[0] = inlet_temp_c + (power_profile_kw[0] * 1000.0) / m_dot_cp
        T_mean[0] = (inlet_temp_c + T_out[0]) / 2.0
        
        for i in range(1, n):
            dt = time_s[i] - time_s[i-1]
            Q_in = power_profile_kw[i] * 1000.0  # W
            Q_out = m_dot_cp * (T_out[i-1] - inlet_temp_c)
            
            dT = (Q_in - Q_out) / C_th * dt
            T_out[i] = T_out[i-1] + dT
            T_mean[i] = (inlet_temp_c + T_out[i]) / 2.0
        
        return {
            'time': time_s,
            'T_outlet': T_out,
            'T_mean': T_mean,
            'flowrate_lpm': np.full(n, flow_lpm),
        }
    
    def compute_cooling_power_kw(self, 
                                  inlet_temp_c: float,
                                  rack_power_kw: Optional[float] = None) -> Dict[str, float]:
        """Estimate cooling infrastructure power consumption.
        
        Includes CDU pump power and air-side fan power for power shelves.
        
        Args:
            inlet_temp_c: Coolant supply inlet temperature (°C)
            rack_power_kw: Rack power in kW
            
        Returns:
            Dict with cooling power breakdown
        """
        if rack_power_kw is None:
            rack_power_kw = self.rack_tdp_kw
        
        # Pump power estimate (based on flow rate and pressure drop)
        flow_lpm = self.required_flowrate_lpm(inlet_temp_c)
        flow_m3_s = flow_lpm / 60000.0
        dp_pa = self.pressure_drop_kpa(inlet_temp_c) * 1000.0
        pump_efficiency = 0.70
        pump_power_w = (flow_m3_s * dp_pa) / pump_efficiency
        
        # Fan power for power shelves
        cfm = self.power_shelf_airflow_cfm(inlet_temp_c)
        fan_power_w = cfm * 0.5  # Approximate: 0.5 W per CFM
        
        # CDU overhead (heat exchanger, controls)
        cdu_overhead_w = rack_power_kw * 10.0  # ~1% of rack power
        
        total_cooling_w = pump_power_w + fan_power_w + cdu_overhead_w
        
        return {
            'pump_power_kw': pump_power_w / 1000.0,
            'fan_power_kw': fan_power_w / 1000.0,
            'cdu_overhead_kw': cdu_overhead_w / 1000.0,
            'total_cooling_kw': total_cooling_w / 1000.0,
            'pPUE': (rack_power_kw + total_cooling_w / 1000.0) / rack_power_kw,
        }
    
    def generate_thermal_report(self, inlet_temp_c: float = 35.0) -> Dict:
        """Generate comprehensive thermal analysis report.
        
        Args:
            inlet_temp_c: Coolant supply inlet temperature (°C)
            
        Returns:
            Dict with complete thermal analysis
        """
        flow_lpm = self.required_flowrate_lpm(inlet_temp_c)
        T_out = self.outlet_temperature(inlet_temp_c)
        dp_psid = self.pressure_drop_psid(inlet_temp_c)
        dp_kpa = self.pressure_drop_kpa(inlet_temp_c)
        cfm = self.power_shelf_airflow_cfm(inlet_temp_c)
        within_limits, checks = self.is_within_thermal_limits(inlet_temp_c)
        cooling = self.compute_cooling_power_kw(inlet_temp_c)
        tau = self.thermal_time_constant_s(inlet_temp_c)
        
        return {
            'power_mode': self.power_mode,
            'rack_tdp_kw': self.rack_tdp_kw,
            'inlet_temp_c': inlet_temp_c,
            'outlet_temp_c': T_out,
            'delta_T_c': T_out - inlet_temp_c,
            'flowrate_lpm': flow_lpm,
            'flowrate_ls': flow_lpm / 60.0,
            'lpm_per_kw': flow_lpm / self.rack_tdp_kw,
            'pressure_drop_psid': dp_psid,
            'pressure_drop_kpa': dp_kpa,
            'power_shelf_airflow_cfm': cfm,
            'liquid_volume_liters': self.total_liquid_volume_liters(),
            'thermal_time_constant_s': tau,
            'within_limits': within_limits,
            'limit_checks': checks,
            'cooling_power': cooling,
        }


if __name__ == "__main__":
    # Example: Vera Rubin NVL72 Max P thermal analysis
    model = VRNvl72ThermalModel(rack_tdp_kw=227.0, power_mode="max_p")
    
    print("=== Vera Rubin NVL72 Max P Thermal Analysis ===")
    for T_in in [25, 30, 35, 40, 45]:
        flow = model.required_flowrate_lpm(T_in)
        T_out = model.outlet_temperature(T_in)
        dp = model.pressure_drop_psid(T_in)
        print(f"  Inlet {T_in}°C: Flow={flow:.0f} lpm, Outlet={T_out:.1f}°C, ΔP={dp:.1f} psid")
    
    print(f"\nTotal liquid volume: {model.total_liquid_volume_liters():.2f} liters")
    print(f"Thermal time constant: {model.thermal_time_constant_s(35.0):.1f} s")
    
    report = model.generate_thermal_report(35.0)
    print(f"\npPUE at 35°C inlet: {report['cooling_power']['pPUE']:.4f}")
"""
