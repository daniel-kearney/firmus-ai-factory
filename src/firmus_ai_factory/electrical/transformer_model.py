"""Transformer System Mathematical Model

Mathematical models for power transformers including loading, losses,
thermal behavior, and lifetime degradation based on IEEE/IEC standards.

Key equations:
    P_loss = P_no_load + P_load × (S/S_rated)²
    θ_hs = θ_amb + Δθ_hs_rated × [(S/S_rated)^(2n)]
    L_aging = exp[(15000/383) - (15000/(θ_hs + 273))]

Standards:
    IEEE C57.91-2011: Loading Guide for Mineral-Oil-Immersed Transformers
    IEC 60076: Power transformers
    IEEE C57.12.00: General Requirements for Liquid-Immersed Distribution Transformers

Platform Integration:
    Batam BT1-2 → 120 MW IT load → Medium voltage distribution transformers
    Singapore → H100/H200 → Similar transformer architecture

Author: daniel.kearney@firmus.co
Date: February 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


class CoolingType(Enum):
    """Transformer cooling types per IEEE/IEC standards."""
    ONAN = "onan"  # Oil Natural, Air Natural (passive cooling)
    ONAF = "onaf"  # Oil Natural, Air Forced (fans)
    OFAF = "ofaf"  # Oil Forced, Air Forced (pumps + fans)
    ODAF = "odaf"  # Oil Directed, Air Forced (directed flow)


class InsulationClass(Enum):
    """Insulation temperature class."""
    CLASS_A = "class_a"  # 105°C
    CLASS_B = "class_b"  # 130°C
    CLASS_F = "class_f"  # 155°C
    CLASS_H = "class_h"  # 180°C


@dataclass
class TransformerSpecifications:
    """Technical specifications for power transformers.
    
    Attributes:
        model: Transformer model identifier
        rated_power_kva: Rated apparent power (kVA)
        primary_voltage_kv: Primary winding voltage (kV, line-to-line)
        secondary_voltage_kv: Secondary winding voltage (kV, line-to-line)
        frequency_hz: Operating frequency (Hz)
        cooling_type: Cooling system type
        insulation_class: Insulation temperature class
        no_load_loss_kw: No-load losses (core losses) at rated voltage (kW)
        load_loss_kw: Load losses (copper losses) at rated current (kW)
        impedance_pct: Impedance voltage (%)
        rated_top_oil_temp_c: Rated top oil temperature rise (°C)
        rated_winding_temp_c: Rated winding temperature rise (°C)
        ambient_temp_c: Design ambient temperature (°C)
        thermal_time_constant_oil_min: Oil thermal time constant (minutes)
        thermal_time_constant_winding_min: Winding thermal time constant (minutes)
        exponent_n: Thermal exponent (0.8-1.0 for ONAN, 0.9-1.0 for ONAF)
        max_hotspot_temp_c: Maximum allowable hotspot temperature (°C)
    """
    model: str
    rated_power_kva: float
    primary_voltage_kv: float
    secondary_voltage_kv: float
    frequency_hz: float = 50.0
    cooling_type: CoolingType = CoolingType.ONAN
    insulation_class: InsulationClass = InsulationClass.CLASS_A
    no_load_loss_kw: float = 0.0
    load_loss_kw: float = 0.0
    impedance_pct: float = 6.0
    rated_top_oil_temp_c: float = 55.0
    rated_winding_temp_c: float = 65.0
    ambient_temp_c: float = 40.0
    thermal_time_constant_oil_min: float = 180.0
    thermal_time_constant_winding_min: float = 10.0
    exponent_n: float = 0.9
    max_hotspot_temp_c: float = 110.0  # Class A limit


# =============================================================================
# Typical Transformer Specifications (estimated for data center applications)
# =============================================================================

# Medium voltage transformer for data center distribution
# Typical: 22kV/415V, 2.5 MVA
MV_TRANSFORMER_2500KVA_SPECS = TransformerSpecifications(
    model="MV-2500kVA-22kV/415V",
    rated_power_kva=2500.0,
    primary_voltage_kv=22.0,
    secondary_voltage_kv=0.415,
    frequency_hz=50.0,
    cooling_type=CoolingType.ONAN,
    insulation_class=InsulationClass.CLASS_A,
    no_load_loss_kw=3.5,  # ~0.14% of rated power
    load_loss_kw=18.0,  # ~0.72% of rated power
    impedance_pct=6.0,
    rated_top_oil_temp_c=55.0,
    rated_winding_temp_c=65.0,
    ambient_temp_c=40.0,
    thermal_time_constant_oil_min=180.0,
    thermal_time_constant_winding_min=10.0,
    exponent_n=0.9,
    max_hotspot_temp_c=110.0,
)

# Large distribution transformer for data center main feed
# Typical: 22kV/11kV, 10 MVA
MV_TRANSFORMER_10MVA_SPECS = TransformerSpecifications(
    model="MV-10MVA-22kV/11kV",
    rated_power_kva=10000.0,
    primary_voltage_kv=22.0,
    secondary_voltage_kv=11.0,
    frequency_hz=50.0,
    cooling_type=CoolingType.ONAF,
    insulation_class=InsulationClass.CLASS_A,
    no_load_loss_kw=12.0,  # ~0.12% of rated power
    load_loss_kw=60.0,  # ~0.60% of rated power
    impedance_pct=7.5,
    rated_top_oil_temp_c=55.0,
    rated_winding_temp_c=65.0,
    ambient_temp_c=40.0,
    thermal_time_constant_oil_min=210.0,
    thermal_time_constant_winding_min=12.0,
    exponent_n=0.95,
    max_hotspot_temp_c=110.0,
)


class TransformerModel:
    """Power transformer model with thermal and loss calculations.
    
    Models:
    - No-load and load losses
    - Top oil temperature (exponential rise)
    - Hotspot winding temperature
    - Aging acceleration factor
    - Lifetime consumption
    - Loading capability
    """
    
    def __init__(
        self,
        specs: TransformerSpecifications,
        ambient_temp_c: Optional[float] = None,
    ):
        """Initialize transformer model.
        
        Args:
            specs: Transformer specifications
            ambient_temp_c: Ambient temperature (°C), overrides spec default
        """
        self.specs = specs
        self.ambient_temp_c = ambient_temp_c if ambient_temp_c is not None else specs.ambient_temp_c
        
        # State variables
        self.top_oil_temp_c = self.ambient_temp_c
        self.hotspot_temp_c = self.ambient_temp_c
        
    def calculate_losses(self, load_kva: float) -> Tuple[float, float, float]:
        """Calculate transformer losses.
        
        Args:
            load_kva: Load (kVA)
            
        Returns:
            Tuple of (no_load_loss_kw, load_loss_kw, total_loss_kw)
        """
        load_fraction = load_kva / self.specs.rated_power_kva
        
        # No-load losses (constant)
        no_load_loss = self.specs.no_load_loss_kw
        
        # Load losses (proportional to I²)
        load_loss = self.specs.load_loss_kw * (load_fraction ** 2)
        
        total_loss = no_load_loss + load_loss
        
        return no_load_loss, load_loss, total_loss
    
    def calculate_efficiency(self, load_kva: float, power_factor: float = 1.0) -> float:
        """Calculate transformer efficiency.
        
        Args:
            load_kva: Load (kVA)
            power_factor: Load power factor
            
        Returns:
            Efficiency (0-1)
        """
        load_kw = load_kva * power_factor
        _, _, total_loss_kw = self.calculate_losses(load_kva)
        
        if load_kw <= 0:
            return 0.0
        
        efficiency = load_kw / (load_kw + total_loss_kw)
        return efficiency
    
    def calculate_top_oil_temperature(
        self,
        load_kva: float,
        steady_state: bool = True,
        time_hours: float = 0.0,
        initial_temp_c: Optional[float] = None,
    ) -> float:
        """Calculate top oil temperature.
        
        Uses IEEE C57.91 exponential rise equation.
        
        Args:
            load_kva: Load (kVA)
            steady_state: If True, return steady-state temperature
            time_hours: Time since load change (hours), used if not steady_state
            initial_temp_c: Initial top oil temperature (°C), used if not steady_state
            
        Returns:
            Top oil temperature (°C)
        """
        load_fraction = load_kva / self.specs.rated_power_kva
        
        # Steady-state top oil rise
        delta_theta_oil_rated = self.specs.rated_top_oil_temp_c
        delta_theta_oil = delta_theta_oil_rated * (load_fraction ** (2 * self.specs.exponent_n))
        
        if steady_state:
            return self.ambient_temp_c + delta_theta_oil
        else:
            # Exponential rise
            if initial_temp_c is None:
                initial_temp_c = self.top_oil_temp_c
            
            tau_oil_hours = self.specs.thermal_time_constant_oil_min / 60.0
            final_temp_c = self.ambient_temp_c + delta_theta_oil
            
            temp_c = final_temp_c - (final_temp_c - initial_temp_c) * np.exp(-time_hours / tau_oil_hours)
            return temp_c
    
    def calculate_hotspot_temperature(
        self,
        load_kva: float,
        top_oil_temp_c: Optional[float] = None,
    ) -> float:
        """Calculate hotspot winding temperature.
        
        Uses IEEE C57.91 hotspot model.
        
        Args:
            load_kva: Load (kVA)
            top_oil_temp_c: Top oil temperature (°C), calculated if not provided
            
        Returns:
            Hotspot temperature (°C)
        """
        if top_oil_temp_c is None:
            top_oil_temp_c = self.calculate_top_oil_temperature(load_kva)
        
        load_fraction = load_kva / self.specs.rated_power_kva
        
        # Hotspot rise over top oil
        delta_theta_hs_rated = self.specs.rated_winding_temp_c - self.specs.rated_top_oil_temp_c
        delta_theta_hs = delta_theta_hs_rated * (load_fraction ** (2 * self.specs.exponent_n))
        
        hotspot_temp_c = top_oil_temp_c + delta_theta_hs
        return hotspot_temp_c
    
    def calculate_aging_acceleration_factor(self, hotspot_temp_c: float) -> float:
        """Calculate aging acceleration factor per IEEE C57.91.
        
        Aging doubles for every 6°C above 110°C (Class A insulation).
        
        Args:
            hotspot_temp_c: Hotspot temperature (°C)
            
        Returns:
            Aging acceleration factor (1.0 = normal aging at 110°C)
        """
        # Reference temperature for Class A insulation
        theta_ref_c = 110.0
        
        # Arrhenius equation constants (IEEE C57.91)
        B = 15000.0  # Activation energy constant (K)
        T_ref_k = theta_ref_c + 273.0
        T_hs_k = hotspot_temp_c + 273.0
        
        F_aa = np.exp((B / T_ref_k) - (B / T_hs_k))
        return F_aa
    
    def calculate_lifetime_consumption(
        self,
        load_profile_kva: np.ndarray,
        time_step_hours: float = 1.0,
    ) -> dict:
        """Calculate transformer lifetime consumption from load profile.
        
        Args:
            load_profile_kva: Time series of load (kVA)
            time_step_hours: Time step (hours)
            
        Returns:
            Dict with aging metrics
        """
        # Calculate hotspot temperatures
        hotspot_temps = np.array([
            self.calculate_hotspot_temperature(load) for load in load_profile_kva
        ])
        
        # Calculate aging acceleration factors
        aging_factors = np.array([
            self.calculate_aging_acceleration_factor(temp) for temp in hotspot_temps
        ])
        
        # Lifetime consumption (hours of life consumed)
        lifetime_consumed_hours = np.sum(aging_factors) * time_step_hours
        
        # Normal life expectancy (IEEE C57.91: 180,000 hours at 110°C)
        normal_life_hours = 180000.0
        
        # Percentage of life consumed
        life_consumed_pct = (lifetime_consumed_hours / normal_life_hours) * 100.0
        
        return {
            "lifetime_consumed_hours": float(lifetime_consumed_hours),
            "life_consumed_pct": float(life_consumed_pct),
            "avg_aging_factor": float(np.mean(aging_factors)),
            "max_aging_factor": float(np.max(aging_factors)),
            "max_hotspot_temp_c": float(np.max(hotspot_temps)),
            "avg_hotspot_temp_c": float(np.mean(hotspot_temps)),
            "normal_life_hours": normal_life_hours,
        }
    
    def calculate_loading_capability(
        self,
        ambient_temp_c: float,
        max_hotspot_temp_c: Optional[float] = None,
    ) -> float:
        """Calculate maximum loading capability.
        
        Args:
            ambient_temp_c: Ambient temperature (°C)
            max_hotspot_temp_c: Maximum allowable hotspot (°C), uses spec default if None
            
        Returns:
            Maximum load fraction (per unit of rated power)
        """
        if max_hotspot_temp_c is None:
            max_hotspot_temp_c = self.specs.max_hotspot_temp_c
        
        # Solve for load fraction that gives max_hotspot_temp_c
        # θ_hs = θ_amb + Δθ_oil_rated × K^(2n) + Δθ_hs_rated × K^(2n)
        # where K = load fraction
        
        delta_theta_oil_rated = self.specs.rated_top_oil_temp_c
        delta_theta_hs_rated = self.specs.rated_winding_temp_c - self.specs.rated_top_oil_temp_c
        exponent = 2 * self.specs.exponent_n
        
        # θ_hs = θ_amb + (Δθ_oil_rated + Δθ_hs_rated) × K^(2n)
        delta_theta_total_rated = delta_theta_oil_rated + delta_theta_hs_rated
        
        # K^(2n) = (θ_hs - θ_amb) / (Δθ_oil_rated + Δθ_hs_rated)
        k_power = (max_hotspot_temp_c - ambient_temp_c) / delta_theta_total_rated
        
        if k_power <= 0:
            return 0.0
        
        # K = (k_power)^(1/(2n))
        load_fraction = k_power ** (1.0 / exponent)
        
        return load_fraction


def calculate_transformer_array_capacity(
    total_load_kva: float,
    redundancy: str = "N+1",
    transformer_model: TransformerSpecifications = MV_TRANSFORMER_2500KVA_SPECS,
) -> dict:
    """Calculate required number of transformers for a given load.
    
    Args:
        total_load_kva: Total load (kVA)
        redundancy: Redundancy scheme ("N", "N+1", "2N")
        transformer_model: Transformer model specifications
        
    Returns:
        Dict with transformer array sizing
    """
    # Calculate base number of units
    n_base = int(np.ceil(total_load_kva / transformer_model.rated_power_kva))
    
    # Apply redundancy
    if redundancy == "N":
        n_total = n_base
    elif redundancy == "N+1":
        n_total = n_base + 1
    elif redundancy == "2N":
        n_total = n_base * 2
    else:
        raise ValueError(f"Unknown redundancy scheme: {redundancy}")
    
    total_capacity_kva = n_total * transformer_model.rated_power_kva
    utilization = total_load_kva / total_capacity_kva
    
    # Calculate losses at this utilization
    load_per_transformer = total_load_kva / n_total
    xfmr = TransformerModel(transformer_model)
    _, _, total_loss_per_unit = xfmr.calculate_losses(load_per_transformer)
    total_losses_kw = total_loss_per_unit * n_total
    
    return {
        "num_transformers": n_total,
        "total_capacity_kva": total_capacity_kva,
        "utilization": utilization,
        "redundancy": redundancy,
        "redundant_units": n_total - n_base,
        "total_losses_kw": total_losses_kw,
        "loss_percentage": (total_losses_kw / (total_load_kva * 0.9)) * 100,  # Assume 0.9 PF
    }


if __name__ == "__main__":
    # Example: Batam BT1-2 transformer sizing
    print("=== Batam BT1-2 Transformer Array Sizing ===")
    
    total_it_load_kw = 120000.0  # 120 MW
    total_load_kva = total_it_load_kw / 0.9  # Assume 0.9 power factor
    
    sizing = calculate_transformer_array_capacity(
        total_load_kva=total_load_kva,
        redundancy="N+1",
        transformer_model=MV_TRANSFORMER_10MVA_SPECS,
    )
    
    print(f"Total IT load: {total_it_load_kw/1000:.1f} MW ({total_load_kva/1000:.1f} MVA)")
    print(f"Number of transformers: {sizing['num_transformers']}")
    print(f"Total transformer capacity: {sizing['total_capacity_kva']/1000:.1f} MVA")
    print(f"Utilization: {sizing['utilization']*100:.1f}%")
    print(f"Total losses: {sizing['total_losses_kw']/1000:.2f} MW ({sizing['loss_percentage']:.2f}%)")
    
    # Example: Single transformer thermal analysis
    print("\n=== Single Transformer Thermal Analysis ===")
    
    xfmr = TransformerModel(
        specs=MV_TRANSFORMER_10MVA_SPECS,
        ambient_temp_c=35.0,  # Batam ambient
    )
    
    load_kva = 8000.0  # 80% load
    top_oil_temp = xfmr.calculate_top_oil_temperature(load_kva)
    hotspot_temp = xfmr.calculate_hotspot_temperature(load_kva, top_oil_temp)
    aging_factor = xfmr.calculate_aging_acceleration_factor(hotspot_temp)
    efficiency = xfmr.calculate_efficiency(load_kva, power_factor=0.9)
    
    print(f"Load: {load_kva/1000:.1f} MVA ({load_kva/MV_TRANSFORMER_10MVA_SPECS.rated_power_kva*100:.1f}%)")
    print(f"Top oil temperature: {top_oil_temp:.1f}°C")
    print(f"Hotspot temperature: {hotspot_temp:.1f}°C")
    print(f"Aging acceleration factor: {aging_factor:.2f}×")
    print(f"Efficiency: {efficiency*100:.2f}%")
