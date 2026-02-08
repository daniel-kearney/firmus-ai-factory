"""UPS System Mathematical Model - Eaton 9395XR Series

Mathematical models for UPS power conditioning, battery energy storage,
and grid interaction based on Eaton 9395XR 1935kVA specifications.

Key equations:
    η_ups(P_load) = piecewise efficiency curve
    P_loss = P_load × (1/η - 1)
    E_battery = ∫ P_discharge dt
    SOC(t) = SOC(t-1) - (P_discharge × Δt) / (V_nom × Ah_capacity)

Platform Integration:
    Batam BT1-2 → 120 MW IT load → 16× Eaton 9395XR (1.875 MW each) → 30 MW total UPS capacity
    Singapore → H100/H200 immersion cooling → Similar UPS architecture

Author: daniel.kearney@firmus.co
Date: February 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum


class UPSMode(Enum):
    """UPS operating modes."""
    DOUBLE_CONVERSION = "double_conversion"  # VFI mode, full power conditioning
    ESS = "ess"  # Energy Saver System (Eco mode), bypass with monitoring
    BATTERY = "battery"  # Stored energy mode, running on battery
    BYPASS = "bypass"  # Static bypass, no conditioning
    MAINTENANCE_BYPASS = "maintenance_bypass"  # Manual bypass for service


class BatteryTechnology(Enum):
    """Battery chemistry options."""
    VRLA = "vrla"  # Valve Regulated Lead Acid
    LITHIUM_ION = "lithium_ion"  # Lithium-ion (certified)
    NI_CAD = "ni_cad"  # Nickel-Cadmium


@dataclass
class UPSSpecifications:
    """Technical specifications for Eaton 9395XR UPS system.
    
    Attributes:
        model: UPS model identifier
        rated_power_kw: Rated output power (kW)
        rated_apparent_power_kva: Rated apparent power (kVA)
        num_modules: Number of UPM (Unit Power Modules)
        module_power_kw: Power per module (kW)
        input_voltage_v: Nominal input voltage (line-to-line, 3-phase)
        output_voltage_v: Nominal output voltage (line-to-line, 3-phase)
        battery_voltage_v: Nominal battery voltage (DC)
        battery_cells: Number of battery cells in series
        efficiency_curve: Dict mapping load fraction to efficiency
        heat_dissipation_curve: Dict mapping load fraction to heat dissipation (kW)
        max_overload_pct: Maximum overload capacity (%)
        overload_duration_s: Duration for maximum overload (seconds)
        bypass_rating_kva: Bypass thyristor rating (kVA)
        bypass_i2t_a2s: Bypass thyristor I²t rating (A²s)
        rectifier_ramp_rate_a_per_s: Rectifier current ramp rate (A/s)
        input_power_factor: Input power factor at rated load
        input_thd_pct: Input current THD at rated load (%)
        transfer_time_ms: Transfer time to battery mode (ms)
    """
    model: str
    rated_power_kw: float
    rated_apparent_power_kva: float
    num_modules: int
    module_power_kw: float
    input_voltage_v: float
    output_voltage_v: float
    battery_voltage_v: float
    battery_cells: int
    efficiency_curve: dict = field(default_factory=dict)
    heat_dissipation_curve: dict = field(default_factory=dict)
    max_overload_pct: float = 140.0
    overload_duration_s: float = 30.0
    bypass_rating_kva: float = 1935.0
    bypass_i2t_a2s: float = 7220000.0
    input_power_factor: float = 0.99
    input_thd_pct: float = 3.0
    rectifier_ramp_rate_a_per_s: float = 10.0
    transfer_time_ms: float = 0.0  # No-break transfer


# =============================================================================
# Pre-defined UPS Specifications
# =============================================================================

EATON_9395XR_1935KVA_SPECS = UPSSpecifications(
    model="Eaton 9395XR-1935",
    rated_power_kw=1875.0,
    rated_apparent_power_kva=1935.0,
    num_modules=15,
    module_power_kw=125.0,
    input_voltage_v=415.0,
    output_voltage_v=415.0,
    battery_voltage_v=480.0,
    battery_cells=240,
    efficiency_curve={
        # Load fraction → efficiency (Double Conversion mode)
        0.25: 0.970,
        0.50: 0.973,
        0.75: 0.970,
        1.00: 0.964,
    },
    heat_dissipation_curve={
        # Load fraction → heat dissipation (kW) for 1875kW system
        0.25: 14.0,
        0.50: 26.0,
        0.75: 40.0,
        1.00: 66.0,
    },
    max_overload_pct=140.0,
    overload_duration_s=30.0,
    bypass_rating_kva=1935.0,
    bypass_i2t_a2s=7220000.0,
    rectifier_ramp_rate_a_per_s=10.0,
    input_power_factor=0.99,
    input_thd_pct=3.0,
    transfer_time_ms=0.0,
)

# ESS mode efficiency curve (Eco mode)
ESS_MODE_EFFICIENCY_CURVE = {
    0.25: 0.984,
    0.50: 0.990,
    0.75: 0.992,
    1.00: 0.992,
}


@dataclass
class BatterySpecifications:
    """Battery system specifications.
    
    Attributes:
        technology: Battery chemistry
        nominal_voltage_v: Nominal voltage (V)
        capacity_ah: Amp-hour capacity (Ah)
        num_cells: Number of cells in series
        float_charge_v_per_cell: Float charge voltage per cell (V)
        eod_v_per_cell: End of discharge voltage per cell (V)
        max_charge_v_per_cell: Maximum charge voltage per cell (V)
        charge_current_a: Charging current (A)
        max_charge_current_a: Maximum charging current (A)
        round_trip_efficiency: Battery round-trip efficiency
        max_dod: Maximum depth of discharge (fraction)
        design_life_years: Design life (years)
        recharge_time_multiplier: Recharge time = discharge time × multiplier
    """
    technology: BatteryTechnology
    nominal_voltage_v: float
    capacity_ah: float
    num_cells: int
    float_charge_v_per_cell: float = 2.30
    eod_v_per_cell: float = 1.67
    max_charge_v_per_cell: float = 2.35
    charge_current_a: float = 4.0
    max_charge_current_a: float = 40.0
    round_trip_efficiency: float = 0.85
    max_dod: float = 0.80
    design_life_years: int = 10
    recharge_time_multiplier: float = 10.0


class UPSSystem:
    """Eaton 9395XR UPS system model with battery energy storage.
    
    Models:
    - Power conditioning efficiency (load-dependent)
    - Battery charge/discharge dynamics
    - Grid interaction (power factor, THD, ramp rates)
    - Thermal dissipation
    - Mode transitions (double conversion, ESS, battery, bypass)
    """
    
    def __init__(
        self,
        ups_specs: UPSSpecifications,
        battery_specs: Optional[BatterySpecifications] = None,
        ambient_temp_c: float = 25.0,
    ):
        """Initialize UPS system model.
        
        Args:
            ups_specs: UPS technical specifications
            battery_specs: Battery specifications (optional)
            ambient_temp_c: Ambient temperature (°C)
        """
        self.ups_specs = ups_specs
        self.battery_specs = battery_specs
        self.ambient_temp_c = ambient_temp_c
        
        # State variables
        self.mode = UPSMode.DOUBLE_CONVERSION
        self.battery_soc = 1.0  # State of charge (0-1)
        self.battery_voltage_v = ups_specs.battery_voltage_v if battery_specs else 0.0
        
    def calculate_efficiency(self, load_kw: float, mode: UPSMode = UPSMode.DOUBLE_CONVERSION) -> float:
        """Calculate UPS efficiency at given load.
        
        Uses piecewise linear interpolation of measured efficiency curve.
        
        Args:
            load_kw: Output load (kW)
            mode: Operating mode
            
        Returns:
            Efficiency (0-1)
        """
        load_fraction = load_kw / self.ups_specs.rated_power_kw
        
        # Select efficiency curve based on mode
        if mode == UPSMode.ESS:
            curve = ESS_MODE_EFFICIENCY_CURVE
        elif mode == UPSMode.DOUBLE_CONVERSION:
            curve = self.ups_specs.efficiency_curve
        elif mode == UPSMode.BATTERY:
            # Battery mode uses inverter efficiency (similar to double conversion)
            curve = self.ups_specs.efficiency_curve
        elif mode in [UPSMode.BYPASS, UPSMode.MAINTENANCE_BYPASS]:
            return 1.0  # Bypass has no conversion losses
        else:
            curve = self.ups_specs.efficiency_curve
        
        # Piecewise linear interpolation
        load_points = sorted(curve.keys())
        
        if load_fraction <= load_points[0]:
            return curve[load_points[0]]
        elif load_fraction >= load_points[-1]:
            return curve[load_points[-1]]
        else:
            # Find bracketing points
            for i in range(len(load_points) - 1):
                if load_points[i] <= load_fraction <= load_points[i+1]:
                    x0, x1 = load_points[i], load_points[i+1]
                    y0, y1 = curve[x0], curve[x1]
                    # Linear interpolation
                    return y0 + (y1 - y0) * (load_fraction - x0) / (x1 - x0)
        
        return curve[load_points[-1]]
    
    def calculate_power_loss(self, load_kw: float, mode: UPSMode = UPSMode.DOUBLE_CONVERSION) -> float:
        """Calculate UPS power loss (heat dissipation).
        
        Args:
            load_kw: Output load (kW)
            mode: Operating mode
            
        Returns:
            Power loss (kW)
        """
        efficiency = self.calculate_efficiency(load_kw, mode)
        if efficiency >= 1.0:
            return 0.0
        return load_kw * (1.0 / efficiency - 1.0)
    
    def calculate_input_power(self, load_kw: float, mode: UPSMode = UPSMode.DOUBLE_CONVERSION) -> float:
        """Calculate UPS input power from grid.
        
        Args:
            load_kw: Output load (kW)
            mode: Operating mode
            
        Returns:
            Input power (kW)
        """
        efficiency = self.calculate_efficiency(load_kw, mode)
        if efficiency <= 0.0:
            return 0.0
        return load_kw / efficiency
    
    def calculate_battery_runtime(
        self,
        load_kw: float,
        initial_soc: float = 1.0,
        min_soc: float = 0.2,
    ) -> float:
        """Calculate battery runtime at constant load.
        
        Args:
            load_kw: Constant discharge load (kW)
            initial_soc: Initial state of charge (0-1)
            min_soc: Minimum allowed SOC (0-1)
            
        Returns:
            Runtime (hours)
        """
        if not self.battery_specs:
            return 0.0
        
        # Available energy
        usable_capacity_wh = (
            self.battery_specs.nominal_voltage_v *
            self.battery_specs.capacity_ah *
            (initial_soc - min_soc)
        )
        
        # Account for inverter efficiency and battery round-trip efficiency
        inverter_efficiency = self.calculate_efficiency(load_kw, UPSMode.BATTERY)
        total_efficiency = inverter_efficiency * self.battery_specs.round_trip_efficiency
        
        # Runtime = Energy / Power
        if load_kw <= 0:
            return float('inf')
        
        runtime_hours = (usable_capacity_wh / 1000.0) / (load_kw / total_efficiency)
        return runtime_hours
    
    def simulate_battery_discharge(
        self,
        load_kw: float,
        duration_hours: float,
        initial_soc: float = 1.0,
        time_step_s: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate battery discharge over time.
        
        Args:
            load_kw: Discharge load (kW)
            duration_hours: Simulation duration (hours)
            initial_soc: Initial state of charge (0-1)
            time_step_s: Time step (seconds)
            
        Returns:
            Tuple of (time_array, soc_array, voltage_array)
        """
        if not self.battery_specs:
            raise ValueError("Battery specifications required for discharge simulation")
        
        num_steps = int(duration_hours * 3600 / time_step_s)
        time_array = np.linspace(0, duration_hours, num_steps)
        soc_array = np.zeros(num_steps)
        voltage_array = np.zeros(num_steps)
        
        soc = initial_soc
        soc_array[0] = soc
        
        # Calculate discharge current
        inverter_efficiency = self.calculate_efficiency(load_kw, UPSMode.BATTERY)
        battery_power_kw = load_kw / inverter_efficiency
        discharge_current_a = (battery_power_kw * 1000.0) / self.battery_specs.nominal_voltage_v
        
        for i in range(1, num_steps):
            # Update SOC
            delta_ah = discharge_current_a * (time_step_s / 3600.0)
            soc -= delta_ah / self.battery_specs.capacity_ah
            soc = max(0.0, soc)  # Clamp to 0
            
            soc_array[i] = soc
            
            # Calculate voltage (linear approximation between float and EOD)
            v_float = self.battery_specs.float_charge_v_per_cell * self.battery_specs.num_cells
            v_eod = self.battery_specs.eod_v_per_cell * self.battery_specs.num_cells
            voltage_array[i] = v_eod + (v_float - v_eod) * soc
        
        voltage_array[0] = self.battery_specs.float_charge_v_per_cell * self.battery_specs.num_cells
        
        return time_array, soc_array, voltage_array
    
    def calculate_recharge_time(
        self,
        initial_soc: float,
        target_soc: float = 0.9,
    ) -> float:
        """Calculate battery recharge time.
        
        Args:
            initial_soc: Initial state of charge (0-1)
            target_soc: Target state of charge (0-1)
            
        Returns:
            Recharge time (hours)
        """
        if not self.battery_specs:
            return 0.0
        
        # Energy to restore
        delta_soc = target_soc - initial_soc
        if delta_soc <= 0:
            return 0.0
        
        energy_wh = (
            delta_soc *
            self.battery_specs.nominal_voltage_v *
            self.battery_specs.capacity_ah
        )
        
        # Charging power (per UPM, scaled by number of modules)
        charge_power_w = (
            self.battery_specs.charge_current_a *
            self.battery_specs.nominal_voltage_v *
            self.ups_specs.num_modules
        )
        
        # Recharge time (with round-trip efficiency loss)
        recharge_hours = (energy_wh / charge_power_w) / self.battery_specs.round_trip_efficiency
        
        return recharge_hours
    
    def estimate_grid_stress(
        self,
        load_profile_kw: np.ndarray,
        time_step_s: float = 1.0,
    ) -> dict:
        """Estimate grid stress metrics from load profile.
        
        Args:
            load_profile_kw: Time series of load (kW)
            time_step_s: Time step (seconds)
            
        Returns:
            Dict with grid stress metrics
        """
        # Calculate input power profile
        input_power_kw = np.array([
            self.calculate_input_power(load, self.mode) for load in load_profile_kw
        ])
        
        # Calculate ramp rates (kW/s)
        ramp_rates_kw_per_s = np.diff(input_power_kw) / time_step_s
        
        # Grid current (assuming 3-phase, line-to-line voltage)
        input_current_a = (input_power_kw * 1000.0) / (
            np.sqrt(3) * self.ups_specs.input_voltage_v * self.ups_specs.input_power_factor
        )
        
        # Current ramp rates (A/s)
        current_ramp_rates_a_per_s = np.diff(input_current_a) / time_step_s
        
        # Count violations of rectifier ramp rate limit
        ramp_violations = np.sum(
            np.abs(current_ramp_rates_a_per_s) > self.ups_specs.rectifier_ramp_rate_a_per_s
        )
        
        return {
            "peak_power_kw": float(np.max(input_power_kw)),
            "avg_power_kw": float(np.mean(input_power_kw)),
            "peak_current_a": float(np.max(input_current_a)),
            "max_ramp_rate_kw_per_s": float(np.max(np.abs(ramp_rates_kw_per_s))),
            "max_current_ramp_a_per_s": float(np.max(np.abs(current_ramp_rates_a_per_s))),
            "ramp_violations": int(ramp_violations),
            "ramp_violation_rate": float(ramp_violations / len(current_ramp_rates_a_per_s)),
            "power_factor": self.ups_specs.input_power_factor,
            "thd_pct": self.ups_specs.input_thd_pct,
        }


def calculate_ups_array_capacity(
    total_load_kw: float,
    redundancy: str = "N+1",
    ups_model: UPSSpecifications = EATON_9395XR_1935KVA_SPECS,
) -> dict:
    """Calculate required number of UPS units for a given load.
    
    Args:
        total_load_kw: Total IT load (kW)
        redundancy: Redundancy scheme ("N", "N+1", "2N")
        ups_model: UPS model specifications
        
    Returns:
        Dict with UPS array sizing
    """
    # Calculate base number of units
    n_base = int(np.ceil(total_load_kw / ups_model.rated_power_kw))
    
    # Apply redundancy
    if redundancy == "N":
        n_total = n_base
    elif redundancy == "N+1":
        n_total = n_base + 1
    elif redundancy == "2N":
        n_total = n_base * 2
    else:
        raise ValueError(f"Unknown redundancy scheme: {redundancy}")
    
    total_capacity_kw = n_total * ups_model.rated_power_kw
    utilization = total_load_kw / total_capacity_kw
    
    return {
        "num_ups_units": n_total,
        "total_capacity_kw": total_capacity_kw,
        "total_capacity_kva": n_total * ups_model.rated_apparent_power_kva,
        "utilization": utilization,
        "redundancy": redundancy,
        "redundant_units": n_total - n_base,
    }


if __name__ == "__main__":
    # Example: Batam BT1-2 UPS sizing
    print("=== Batam BT1-2 UPS Array Sizing ===")
    
    total_it_load_kw = 120000.0  # 120 MW
    sizing = calculate_ups_array_capacity(
        total_load_kw=total_it_load_kw,
        redundancy="N+1",
        ups_model=EATON_9395XR_1935KVA_SPECS,
    )
    
    print(f"Total IT load: {total_it_load_kw/1000:.1f} MW")
    print(f"Number of UPS units: {sizing['num_ups_units']}")
    print(f"Total UPS capacity: {sizing['total_capacity_kw']/1000:.1f} MW")
    print(f"Utilization: {sizing['utilization']*100:.1f}%")
    print(f"Redundancy: {sizing['redundancy']}")
    
    # Example: Single UPS performance
    print("\n=== Single UPS Performance Analysis ===")
    
    ups = UPSSystem(
        ups_specs=EATON_9395XR_1935KVA_SPECS,
        battery_specs=BatterySpecifications(
            technology=BatteryTechnology.VRLA,
            nominal_voltage_v=480.0,
            capacity_ah=1000.0,  # Example capacity
            num_cells=240,
        ),
    )
    
    load_kw = 1500.0  # 80% load
    efficiency = ups.calculate_efficiency(load_kw)
    power_loss = ups.calculate_power_loss(load_kw)
    runtime = ups.calculate_battery_runtime(load_kw)
    
    print(f"Load: {load_kw} kW ({load_kw/EATON_9395XR_1935KVA_SPECS.rated_power_kw*100:.1f}%)")
    print(f"Efficiency: {efficiency*100:.2f}%")
    print(f"Power loss (heat): {power_loss:.1f} kW")
    print(f"Battery runtime: {runtime*60:.1f} minutes")
