"""Direct-to-Chip (DLC) Liquid Cooling System Modeling.

Mathematical models for direct-to-chip liquid cooling systems
used in high-density AI racks such as NVIDIA GB200/GB300 NVL72.

DLC uses cold plates mounted directly on GPU/CPU dies with a
closed-loop water/glycol circuit to a facility CDU or dry cooler.

Key equations:
    Cold plate: Q = h_cp * A_cp * (T_junction - T_fluid)
    Fluid loop: Q = m_dot * c_p * (T_out - T_in)
    CDU heat exchange: Q = U * A * LMTD
    Rack PUE: PUE = (P_IT + P_pump + P_cdu) / P_IT
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


# Water/Glycol (25% PG) properties at ~35C
WATER_GLYCOL_DENSITY = 1025.0       # kg/m^3
WATER_GLYCOL_SPECIFIC_HEAT = 3900.0 # J/(kg*K)
WATER_GLYCOL_CONDUCTIVITY = 0.52    # W/(m*K)
WATER_GLYCOL_VISCOSITY = 0.0012     # Pa*s
WATER_GLYCOL_PRANDTL = 9.0

# Pure water properties at ~35C
WATER_DENSITY = 994.0
WATER_SPECIFIC_HEAT = 4178.0
WATER_CONDUCTIVITY = 0.623
WATER_VISCOSITY = 0.00072
WATER_PRANDTL = 4.83


@dataclass
class DLCResult:
    """Results from direct-to-chip liquid cooling analysis."""
    T_junction_max: float       # Max GPU junction temperature (C)
    T_junction_mean: float      # Mean GPU junction temperature (C)
    T_coolant_supply: float     # Coolant supply temperature (C)
    T_coolant_return: float     # Coolant return temperature (C)
    P_pump: float               # Pump power (W)
    P_cdu: float                # CDU power (W)
    P_cooling_total: float      # Total cooling power (W)
    flow_rate_lpm: float        # Total flow rate (L/min)
    cold_plate_htc: float       # Cold plate HTC (W/m^2/K)
    pPUE: float                 # Partial PUE


@dataclass
class ColdPlateSpec:
    """Cold plate specifications for GPU/CPU."""
    name: str
    thermal_resistance: float   # Cold plate R_theta (K/W)
    contact_area: float         # Contact area (m^2)
    flow_rate_lpm: float        # Design flow rate per plate (L/min)
    pressure_drop_kpa: float    # Pressure drop at design flow (kPa)


# Pre-defined cold plate specs
GPU_COLD_PLATE = ColdPlateSpec(
    name="GPU High-Performance Cold Plate",
    thermal_resistance=0.03,   # K/W - very low for DLC
    contact_area=0.004,        # ~60mm x 60mm GPU die
    flow_rate_lpm=1.5,         # Per GPU
    pressure_drop_kpa=25.0
)

CPU_COLD_PLATE = ColdPlateSpec(
    name="CPU Cold Plate (Grace)",
    thermal_resistance=0.06,
    contact_area=0.003,
    flow_rate_lpm=0.8,
    pressure_drop_kpa=15.0
)

SWITCH_COLD_PLATE = ColdPlateSpec(
    name="NVSwitch Cold Plate",
    thermal_resistance=0.05,
    contact_area=0.002,
    flow_rate_lpm=0.5,
    pressure_drop_kpa=10.0
)


class DirectToChipCooling:
    """Direct-to-chip liquid cooling system model.
    
    Models the closed-loop liquid cooling used in NVL72 racks:
    - Cold plates on each GPU, CPU, and NVSwitch
    - Manifold distribution within the rack
    - In-rack CDU (Coolant Distribution Unit) for heat rejection
    - Pump system with redundancy
    
    Typical NVL72 configuration:
    - Coolant enters rack at 25C, exits at ~45C (20C delta)
    - 250kW CDU capacity per rack
    - Redundant pump system
    """
    
    def __init__(self,
                 supply_temp: float = 25.0,
                 max_return_temp: float = 45.0,
                 coolant_density: float = WATER_GLYCOL_DENSITY,
                 coolant_cp: float = WATER_GLYCOL_SPECIFIC_HEAT,
                 coolant_k: float = WATER_GLYCOL_CONDUCTIVITY,
                 coolant_viscosity: float = WATER_GLYCOL_VISCOSITY,
                 pump_efficiency: float = 0.75,
                 cdu_capacity_kw: float = 250.0):
        
        self.supply_temp = supply_temp
        self.max_return_temp = max_return_temp
        self.coolant_density = coolant_density
        self.coolant_cp = coolant_cp
        self.coolant_k = coolant_k
        self.coolant_viscosity = coolant_viscosity
        self.pump_efficiency = pump_efficiency
        self.cdu_capacity_kw = cdu_capacity_kw
        
        # Cold plate specs
        self.gpu_cold_plate = GPU_COLD_PLATE
        self.cpu_cold_plate = CPU_COLD_PLATE
        self.switch_cold_plate = SWITCH_COLD_PLATE
    
    def compute_cold_plate_htc(self,
                                flow_rate_lpm: float,
                                cold_plate: ColdPlateSpec) -> float:
        """Calculate cold plate heat transfer coefficient.
        
        Uses microchannel correlation for cold plates:
        h = k / R_theta / A (simplified from thermal resistance)
        """
        # Flow-dependent scaling of thermal resistance
        flow_ratio = flow_rate_lpm / cold_plate.flow_rate_lpm
        R_effective = cold_plate.thermal_resistance / (flow_ratio ** 0.4)
        h = 1.0 / (R_effective * cold_plate.contact_area)
        return h
    
    def compute_junction_temp(self,
                              power: float,
                              cold_plate: ColdPlateSpec,
                              fluid_temp: float) -> float:
        """Calculate junction temperature for a component.
        
        T_j = T_fluid + P * R_jc_total
        where R_jc_total includes die, TIM, and cold plate
        """
        # Die + TIM resistance (additional to cold plate)
        R_die_tim = 0.02  # K/W for direct die attach
        R_total = R_die_tim + cold_plate.thermal_resistance
        return fluid_temp + power * R_total
    
    def compute_required_flow(self,
                               total_power: float) -> float:
        """Calculate required coolant flow rate (L/min).
        
        Q = m_dot * c_p * delta_T
        """
        delta_T = self.max_return_temp - self.supply_temp
        m_dot = total_power / (self.coolant_cp * delta_T)  # kg/s
        vol_flow_m3s = m_dot / self.coolant_density
        return vol_flow_m3s * 60000  # Convert to L/min
    
    def compute_pump_power(self,
                            flow_rate_lpm: float,
                            total_pressure_drop_kpa: float) -> float:
        """Calculate pump electrical power.
        
        P_pump = (Q * delta_P) / eta
        """
        vol_flow = flow_rate_lpm / 60000  # m^3/s
        pressure_pa = total_pressure_drop_kpa * 1000
        return (vol_flow * pressure_pa) / self.pump_efficiency
    
    def analyze_nvl72_rack(self,
                            gpu_power: float,
                            num_gpus: int = 72,
                            num_cpus: int = 36,
                            cpu_power: float = 250.0,
                            switch_power: float = 3200.0
                            ) -> DLCResult:
        """Analyze DLC for a complete NVL72 rack.
        
        Args:
            gpu_power: Power per GPU (W)
            num_gpus: Number of GPUs (72 for NVL72)
            num_cpus: Number of CPUs (36 Grace CPUs)
            cpu_power: Power per CPU (W)
            switch_power: Total NVSwitch power (W)
            
        Returns:
            DLCResult with complete thermal analysis
        """
        # Total liquid-cooled power
        total_gpu_power = gpu_power * num_gpus
        total_cpu_power = cpu_power * num_cpus
        total_liquid_power = total_gpu_power + total_cpu_power + switch_power
        
        # Required flow rate
        flow_lpm = self.compute_required_flow(total_liquid_power)
        
        # Per-GPU flow rate (proportional to power)
        gpu_flow = (gpu_power / total_liquid_power) * flow_lpm / num_gpus
        
        # Cold plate HTC
        h_gpu = self.compute_cold_plate_htc(gpu_flow, self.gpu_cold_plate)
        
        # Fluid temperature at GPU (worst case - last in series)
        # Assume parallel manifold with slight temperature gradient
        T_fluid_avg = self.supply_temp + (self.max_return_temp - self.supply_temp) * 0.5
        T_fluid_worst = self.supply_temp + (self.max_return_temp - self.supply_temp) * 0.75
        
        # Junction temperatures
        T_j_mean = self.compute_junction_temp(
            gpu_power, self.gpu_cold_plate, T_fluid_avg
        )
        T_j_max = self.compute_junction_temp(
            gpu_power, self.gpu_cold_plate, T_fluid_worst
        )
        
        # Return temperature
        delta_T = total_liquid_power / (
            self.coolant_density * (flow_lpm / 60000) * self.coolant_cp
        )
        T_return = self.supply_temp + delta_T
        
        # Pump power (system pressure drop ~150 kPa for NVL72)
        system_pressure_drop = 150.0  # kPa total
        P_pump = self.compute_pump_power(flow_lpm, system_pressure_drop)
        
        # CDU power (~2-3% of heat load for fans/compressor)
        P_cdu = total_liquid_power * 0.025
        P_cooling_total = P_pump + P_cdu
        
        # IT power for PUE (GPUs + CPUs + switches)
        P_IT = total_liquid_power
        pPUE = (P_IT + P_cooling_total) / P_IT
        
        return DLCResult(
            T_junction_max=T_j_max,
            T_junction_mean=T_j_mean,
            T_coolant_supply=self.supply_temp,
            T_coolant_return=T_return,
            P_pump=P_pump,
            P_cdu=P_cdu,
            P_cooling_total=P_cooling_total,
            flow_rate_lpm=flow_lpm,
            cold_plate_htc=h_gpu,
            pPUE=pPUE
        )


if __name__ == "__main__":
    # Example: GB300 NVL72 rack analysis
    dlc = DirectToChipCooling(
        supply_temp=25.0,
        max_return_temp=45.0,
        cdu_capacity_kw=250.0
    )
    
    result = dlc.analyze_nvl72_rack(
        gpu_power=1400.0,  # GB300 at TDP
        num_gpus=72,
        num_cpus=36,
        cpu_power=250.0,
        switch_power=3200.0
    )
    
    print(f"GB300 NVL72 DLC Analysis:")
    print(f"  T_junction max: {result.T_junction_max:.1f} C")
    print(f"  T_junction mean: {result.T_junction_mean:.1f} C")
    print(f"  Coolant supply: {result.T_coolant_supply:.1f} C")
    print(f"  Coolant return: {result.T_coolant_return:.1f} C")
    print(f"  Flow rate: {result.flow_rate_lpm:.1f} L/min")
    print(f"  Pump power: {result.P_pump:.0f} W")
    print(f"  CDU power: {result.P_cdu:.0f} W")
    print(f"  Total cooling: {result.P_cooling_total:.0f} W")
    print(f"  pPUE: {result.pPUE:.3f}")
