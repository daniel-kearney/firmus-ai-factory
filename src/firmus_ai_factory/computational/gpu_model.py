"""GPU Power and Performance Modeling.

Mathematical models for GPU power consumption and computational
performance based on workload characteristics.

Key equations:
    P_GPU(t) = P_compute(t) + P_memory(t) + P_transfer(t)
    P_train(t) = P_base + sum(alpha_i * f_i(t)) + epsilon(t)

Platform-Region-Cooling Mapping:
    HGX H100/H200 → Singapore → Immersion Cooling
    GB300 NVL72   → Australia → Benmax HCU2500 Air-Liquid Cooling
    VR NVL72      → Australia → Benmax HCU2500 Air-Liquid Cooling
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum


class WorkloadType(Enum):
    """Types of GPU workloads."""
    TRAINING = "training"
    INFERENCE = "inference"
    IDLE = "idle"


class PowerProfile_Mode(Enum):
    """GPU power profile modes (per NVIDIA specifications)."""
    MAX_P = "max_p"   # Maximum power for unconstrained DCs
    MAX_Q = "max_q"   # Optimized perf/watt for power-constrained DCs


@dataclass
class GPUSpecifications:
    """Hardware specifications for GPU accelerators.
    
    Attributes:
        name: GPU model name
        tdp_watts: Thermal Design Power (W)
        peak_flops_fp16: Peak FP16 performance (TFLOPS)
        peak_flops_fp32: Peak FP32 performance (TFLOPS)
        hbm_bandwidth_tb_s: Memory bandwidth (TB/s)
        hbm_capacity_gb: HBM capacity (GB)
        nvlink_bandwidth_gb_s: NVLink bandwidth (GB/s)
        idle_power_fraction: Fraction of TDP at idle
        base_power_fraction: Base power during compute as fraction of TDP
        max_junction_temp_c: Maximum junction temperature (°C)
        energy_storage_j: On-board energy storage per GPU (J) for AC input current profiles
    """
    name: str
    tdp_watts: float
    peak_flops_fp16: float
    peak_flops_fp32: float
    hbm_bandwidth_tb_s: float
    hbm_capacity_gb: float
    nvlink_bandwidth_gb_s: float
    idle_power_fraction: float = 0.15
    base_power_fraction: float = 0.65
    max_junction_temp_c: float = 100.0
    energy_storage_j: float = 0.0


# =============================================================================
# Pre-defined GPU Specifications
# =============================================================================

# --- Hopper Architecture (Singapore + Immersion Cooling) ---

H100_SXM_SPECS = GPUSpecifications(
    name="NVIDIA H100 SXM",
    tdp_watts=700.0,
    peak_flops_fp16=1979.0,
    peak_flops_fp32=989.5,
    hbm_bandwidth_tb_s=3.35,
    hbm_capacity_gb=80.0,
    nvlink_bandwidth_gb_s=900.0,
    idle_power_fraction=0.12,
    base_power_fraction=0.70,
    max_junction_temp_c=83.0,
    energy_storage_j=0.0
)

H200_SPECS = GPUSpecifications(
    name="NVIDIA H200",
    tdp_watts=700.0,
    peak_flops_fp16=1979.0,
    peak_flops_fp32=989.5,
    hbm_bandwidth_tb_s=4.8,
    hbm_capacity_gb=141.0,
    nvlink_bandwidth_gb_s=900.0,
    idle_power_fraction=0.12,
    base_power_fraction=0.68,
    max_junction_temp_c=83.0,
    energy_storage_j=0.0
)

# --- Blackwell Architecture (Australia + Benmax HCU2500) ---

B200_SPECS = GPUSpecifications(
    name="NVIDIA B200",
    tdp_watts=1000.0,
    peak_flops_fp16=4500.0,
    peak_flops_fp32=2250.0,
    hbm_bandwidth_tb_s=8.0,
    hbm_capacity_gb=192.0,
    nvlink_bandwidth_gb_s=1800.0,
    idle_power_fraction=0.10,
    base_power_fraction=0.65,
    max_junction_temp_c=100.0,
    energy_storage_j=65.0
)

GB300_SPECS = GPUSpecifications(
    name="NVIDIA GB300 Blackwell Ultra",
    tdp_watts=1400.0,
    peak_flops_fp16=5000.0,
    peak_flops_fp32=2500.0,
    hbm_bandwidth_tb_s=8.0,
    hbm_capacity_gb=288.0,
    nvlink_bandwidth_gb_s=1800.0,
    idle_power_fraction=0.08,
    base_power_fraction=0.65,
    max_junction_temp_c=100.0,
    energy_storage_j=65.0  # GB300 energy storage per GPU
)

# --- Vera Rubin Architecture (Australia + Benmax HCU2500) ---
# Source: VRNVL72TechnicalBriefv02.pdf (December 2025)

VERA_RUBIN_MAX_P_SPECS = GPUSpecifications(
    name="NVIDIA Vera Rubin (Max P)",
    tdp_watts=2300.0,           # 2.3 kW TGP per GPU (Max P mode)
    peak_flops_fp16=6500.0,     # Estimated ~30% uplift over GB300
    peak_flops_fp32=3250.0,
    hbm_bandwidth_tb_s=12.0,    # Next-gen HBM4
    hbm_capacity_gb=384.0,      # HBM4 capacity
    nvlink_bandwidth_gb_s=3600.0,  # NVLink6
    idle_power_fraction=0.06,
    base_power_fraction=0.65,
    max_junction_temp_c=100.0,
    energy_storage_j=400.0      # 400 J/GPU for improved AC input current profiles
)

VERA_RUBIN_MAX_Q_SPECS = GPUSpecifications(
    name="NVIDIA Vera Rubin (Max Q)",
    tdp_watts=1800.0,           # 1.8 kW TGP per GPU (Max Q mode)
    peak_flops_fp16=5500.0,     # Reduced from Max P
    peak_flops_fp32=2750.0,
    hbm_bandwidth_tb_s=12.0,
    hbm_capacity_gb=384.0,
    nvlink_bandwidth_gb_s=3600.0,
    idle_power_fraction=0.07,
    base_power_fraction=0.68,   # Higher base fraction (optimized perf/watt)
    max_junction_temp_c=100.0,
    energy_storage_j=400.0
)


# =============================================================================
# NVL72 Rack Configurations
# =============================================================================

@dataclass
class NVL72RackConfig:
    """Configuration for an NVL72 rack-scale system.
    
    Models NVIDIA NVL72 rack architecture:
    - 72 GPUs + 36 Grace CPUs in a single rack
    - NVLink full-bisection bandwidth within L1 domain
    - Liquid-cooled compute with air-cooled power shelves
    
    Attributes:
        gpu_specs: GPU hardware specifications
        num_gpus: Number of GPUs per rack
        num_cpus: Number of Grace CPUs per rack
        num_compute_trays: Number of compute trays (each: 2 CPUs + 4 GPUs)
        num_switch_trays: Number of NVLink switch trays
        nvlink_aggregate_tb_s: Total NVLink bandwidth (TB/s)
        rack_tdp_kw: Rack Thermal Design Power (kW) - AC average power, 1s moving avg
        rack_edp_kva: Rack Electrical Design Power (kVA) - AC peak power, 50ms moving avg
        cpu_power_per_unit: Grace CPU TDP (W)
        switch_power_total: NVSwitch + transceivers total power (W)
        peripheral_power: NICs, storage, BMC, etc. (W)
        num_power_shelves: Number of power shelves (3+1 redundancy)
        power_shelf_rating_kw: Rating per power shelf (kW)
        power_shelf_voltage_range: AC voltage range for power shelves
        busbar_voltage: DC busbar voltage (V)
        busbar_current_rating: Busbar current rating (A)
        rack_dimensions_mm: (width, height, depth) in mm
        rack_weight_wet_kg: Wet populated weight (kg)
        liquid_volume_liters: Total rack liquid volume (liters)
        coolant_type: Coolant type (e.g., 'PG25')
        power_mode: Power profile mode (Max P or Max Q)
    """
    gpu_specs: GPUSpecifications
    num_gpus: int = 72
    num_cpus: int = 36
    num_compute_trays: int = 18
    num_switch_trays: int = 9
    nvlink_aggregate_tb_s: float = 130.0
    rack_tdp_kw: float = 150.0
    rack_edp_kva: float = 160.0
    cpu_power_per_unit: float = 250.0
    switch_power_total: float = 3200.0
    peripheral_power: float = 2000.0
    num_power_shelves: int = 4
    power_shelf_rating_kw: float = 110.0
    power_shelf_voltage_range: Tuple[float, float] = (415.0, 480.0)
    busbar_voltage: float = 54.0
    busbar_current_rating: float = 5000.0
    rack_dimensions_mm: Tuple[float, float, float] = (600.0, 2300.0, 1200.0)
    rack_weight_wet_kg: float = 1579.0
    liquid_volume_liters: float = 33.59
    coolant_type: str = "PG25"
    power_mode: PowerProfile_Mode = PowerProfile_Mode.MAX_P
    
    @property
    def total_gpu_power(self) -> float:
        """Total GPU power at TDP (W)."""
        return self.num_gpus * self.gpu_specs.tdp_watts
    
    @property
    def total_cpu_power(self) -> float:
        """Total CPU power (W)."""
        return self.num_cpus * self.cpu_power_per_unit
    
    @property
    def total_rack_power(self) -> float:
        """Total rack power at full load (W)."""
        return (self.total_gpu_power + self.total_cpu_power +
                self.switch_power_total + self.peripheral_power)
    
    @property
    def hbm_total_tb(self) -> float:
        """Total GPU HBM capacity in TB."""
        return self.num_gpus * self.gpu_specs.hbm_capacity_gb / 1000.0
    
    @property
    def total_energy_storage_j(self) -> float:
        """Total on-board energy storage (J) across all GPUs."""
        return self.num_gpus * self.gpu_specs.energy_storage_j
    
    @property
    def floor_loading_kg_m2(self) -> float:
        """Floor loading in kg/m² based on rack footprint."""
        width_m = self.rack_dimensions_mm[0] / 1000.0
        depth_m = self.rack_dimensions_mm[2] / 1000.0
        return self.rack_weight_wet_kg / (width_m * depth_m)
    
    @property
    def power_shelf_redundancy(self) -> str:
        """Power shelf redundancy configuration."""
        required = self.num_power_shelves - 1
        return f"{required}+1"


# --- Pre-defined NVL72 Rack Configurations ---

# GB300 NVL72 (Australia + Benmax HCU2500)
GB300_NVL72_CONFIG = NVL72RackConfig(
    gpu_specs=GB300_SPECS,
    num_gpus=72,
    num_cpus=36,
    num_compute_trays=18,
    num_switch_trays=9,
    nvlink_aggregate_tb_s=130.0,
    rack_tdp_kw=150.0,
    rack_edp_kva=160.0,
    cpu_power_per_unit=250.0,
    switch_power_total=3200.0,
    peripheral_power=2000.0,
    num_power_shelves=4,
    power_shelf_rating_kw=110.0,
    power_shelf_voltage_range=(415.0, 480.0),
    busbar_voltage=54.0,
    busbar_current_rating=5000.0,
    rack_dimensions_mm=(600.0, 2300.0, 1200.0),
    rack_weight_wet_kg=1579.0,
    liquid_volume_liters=33.59,
    coolant_type="PG25",
    power_mode=PowerProfile_Mode.MAX_P
)

# GB200 NVL72 (reference configuration)
GB200_NVL72_CONFIG = NVL72RackConfig(
    gpu_specs=B200_SPECS,
    num_gpus=72,
    num_cpus=36,
    nvlink_aggregate_tb_s=130.0,
    rack_tdp_kw=120.0,
    rack_edp_kva=128.0,
    rack_weight_wet_kg=1629.0,
    power_mode=PowerProfile_Mode.MAX_P
)

# Vera Rubin NVL72 Max P (Australia + Benmax HCU2500)
# Source: VRNVL72TechnicalBriefv02.pdf
VR_NVL72_MAX_P_CONFIG = NVL72RackConfig(
    gpu_specs=VERA_RUBIN_MAX_P_SPECS,
    num_gpus=72,
    num_cpus=36,
    num_compute_trays=18,
    num_switch_trays=9,
    nvlink_aggregate_tb_s=130.0,       # NVLink6
    rack_tdp_kw=227.0,                 # 227 kW TDP (Max P)
    rack_edp_kva=240.0,                # 240 kVA EDP (Max P)
    cpu_power_per_unit=250.0,
    switch_power_total=4000.0,          # Upgraded NVSwitch
    peripheral_power=2500.0,            # BF4 DPU, CX9, etc.
    num_power_shelves=4,                # 3+1 redundancy
    power_shelf_rating_kw=110.0,        # 3RU 110kW power shelf
    power_shelf_voltage_range=(415.0, 480.0),
    busbar_voltage=54.0,                # Single zone, liquid cooled
    busbar_current_rating=5000.0,
    rack_dimensions_mm=(600.0, 2300.0, 1200.0),
    rack_weight_wet_kg=1583.0,          # Per tech brief
    liquid_volume_liters=33.59,         # PG25 total rack volume
    coolant_type="PG25",
    power_mode=PowerProfile_Mode.MAX_P
)

# Vera Rubin NVL72 Max Q (Australia + Benmax HCU2500)
VR_NVL72_MAX_Q_CONFIG = NVL72RackConfig(
    gpu_specs=VERA_RUBIN_MAX_Q_SPECS,
    num_gpus=72,
    num_cpus=36,
    num_compute_trays=18,
    num_switch_trays=9,
    nvlink_aggregate_tb_s=130.0,
    rack_tdp_kw=187.0,                 # 187 kW TDP (Max Q)
    rack_edp_kva=198.0,                # 198 kVA EDP (Max Q)
    cpu_power_per_unit=250.0,
    switch_power_total=4000.0,
    peripheral_power=2500.0,
    num_power_shelves=4,
    power_shelf_rating_kw=110.0,
    power_shelf_voltage_range=(415.0, 480.0),
    busbar_voltage=54.0,
    busbar_current_rating=5000.0,
    rack_dimensions_mm=(600.0, 2300.0, 1200.0),
    rack_weight_wet_kg=1583.0,
    liquid_volume_liters=33.59,
    coolant_type="PG25",
    power_mode=PowerProfile_Mode.MAX_Q
)


# =============================================================================
# Power Profile
# =============================================================================

@dataclass
class PowerProfile:
    """Time-series power consumption profile."""
    time: np.ndarray
    total_power: np.ndarray
    compute_power: np.ndarray
    memory_power: np.ndarray
    transfer_power: np.ndarray
    
    @property
    def mean_power(self) -> float:
        return float(np.mean(self.total_power))
    
    @property
    def peak_power(self) -> float:
        return float(np.max(self.total_power))
    
    @property
    def energy_kwh(self) -> float:
        dt = np.diff(self.time)
        return float(np.sum(self.total_power[:-1] * dt) / 3.6e6)


# =============================================================================
# GPU Model
# =============================================================================

class GPUModel:
    """Mathematical model for GPU power and performance.
    
    Implements the power decomposition model:
        P_GPU(t) = P_compute(t) + P_memory(t) + P_transfer(t)
    
    For training workloads:
        P_train(t) = P_base + sum(alpha_i * f_i(t)) + epsilon(t)
    
    where:
        - P_base: sustained baseline (~0.65-0.70 * TDP)
        - alpha_i: phase intensity coefficients
        - f_i(t): phase indicator functions
        - epsilon(t): stochastic variations
    """
    
    def __init__(self, specs: GPUSpecifications):
        self.specs = specs
        
        # Phase coefficients for training workload
        self.phase_coefficients = {
            'forward': 0.85,
            'backward': 0.95,
            'gradient_sync': 0.60,
            'optimizer': 0.75
        }
        
        # Stochastic variation parameters
        self.noise_std = 0.02 * specs.tdp_watts
        
    def compute_flops_per_token(self, 
                                 num_params: float,
                                 sequence_length: int = 2048) -> float:
        """Calculate FLOPS per token for transformer models.
        
        For transformer: FLOP/token ~= 6 * num_params (forward + backward)
        """
        return 6.0 * num_params
    
    def compute_mfu(self,
                    tokens_per_second: float,
                    num_params: float,
                    num_gpus: int = 1) -> float:
        """Calculate Model FLOPS Utilization (MFU).
        
        MFU = (observed throughput * FLOP/token) / peak_FLOPS
        """
        flops_per_token = self.compute_flops_per_token(num_params)
        observed_flops = tokens_per_second * flops_per_token
        peak_flops = self.specs.peak_flops_fp16 * 1e12 * num_gpus
        return observed_flops / peak_flops
    
    def power_from_utilization(self, utilization: float) -> float:
        """Calculate power from GPU utilization (0-1).
        
        P = P_idle + (P_TDP - P_idle) * utilization^alpha
        
        where alpha accounts for non-linear scaling.
        """
        p_idle = self.specs.idle_power_fraction * self.specs.tdp_watts
        p_dynamic = (1 - self.specs.idle_power_fraction) * self.specs.tdp_watts
        alpha = 1.2  # Slight super-linear scaling
        return p_idle + p_dynamic * (utilization ** alpha)
    
    def simulate_training_workload(
        self,
        model_params: float,
        batch_size: int,
        duration: float,
        dt: float = 0.001
    ) -> PowerProfile:
        """Simulate power profile for training workload.
        
        Args:
            model_params: Number of model parameters
            batch_size: Training batch size
            duration: Simulation duration (seconds)
            dt: Time step (seconds)
            
        Returns:
            PowerProfile with time-series data
        """
        t = np.arange(0, duration, dt)
        n_steps = len(t)
        
        # Estimate iteration time based on model size
        iter_time = self._estimate_iteration_time(model_params, batch_size)
        
        # Phase durations within iteration
        forward_frac = 0.30
        backward_frac = 0.45
        sync_frac = 0.15
        opt_frac = 0.10
        
        # Generate phase indicator functions
        phase_in_iter = (t % iter_time) / iter_time
        
        forward_mask = phase_in_iter < forward_frac
        backward_mask = (phase_in_iter >= forward_frac) & \
                       (phase_in_iter < forward_frac + backward_frac)
        sync_mask = (phase_in_iter >= forward_frac + backward_frac) & \
                   (phase_in_iter < forward_frac + backward_frac + sync_frac)
        opt_mask = phase_in_iter >= (1 - opt_frac)
        
        # Base power
        p_base = self.specs.base_power_fraction * self.specs.tdp_watts
        
        # Compute power components
        p_compute = np.full(n_steps, p_base)
        p_compute[forward_mask] *= self.phase_coefficients['forward']
        p_compute[backward_mask] *= self.phase_coefficients['backward']
        p_compute[sync_mask] *= self.phase_coefficients['gradient_sync']
        p_compute[opt_mask] *= self.phase_coefficients['optimizer']
        
        # Memory power (proportional to bandwidth utilization)
        memory_util = 0.6 + 0.3 * np.sin(2 * np.pi * t / iter_time)
        p_memory = 0.15 * self.specs.tdp_watts * memory_util
        
        # Transfer power (spikes during sync)
        p_transfer = np.zeros(n_steps)
        p_transfer[sync_mask] = 0.10 * self.specs.tdp_watts
        
        # Add stochastic variations
        noise = np.random.normal(0, self.noise_std, n_steps)
        
        # Total power with clipping
        p_total = np.clip(
            p_compute + p_memory + p_transfer + noise,
            self.specs.idle_power_fraction * self.specs.tdp_watts,
            self.specs.tdp_watts
        )
        
        return PowerProfile(
            time=t,
            total_power=p_total,
            compute_power=p_compute,
            memory_power=p_memory,
            transfer_power=p_transfer
        )
    
    def _estimate_iteration_time(self,
                                  model_params: float,
                                  batch_size: int) -> float:
        """Estimate training iteration time.
        
        Based on roofline model analysis.
        """
        # Compute-bound estimate
        flops_per_iter = 6 * model_params * batch_size
        compute_time = flops_per_iter / (self.specs.peak_flops_fp16 * 1e12 * 0.5)
        
        # Memory-bound estimate  
        bytes_per_iter = 2 * model_params * 4  # FP32 gradients
        memory_time = bytes_per_iter / (self.specs.hbm_bandwidth_tb_s * 1e12)
        
        return max(compute_time, memory_time) * 1.2  # 20% overhead


if __name__ == "__main__":
    # Example: Vera Rubin NVL72 Max P configuration
    vr_gpu = GPUModel(VERA_RUBIN_MAX_P_SPECS)
    
    print("=== Vera Rubin NVL72 Max P Configuration ===")
    print(f"GPU TGP: {VERA_RUBIN_MAX_P_SPECS.tdp_watts:.0f} W")
    print(f"Rack TDP: {VR_NVL72_MAX_P_CONFIG.rack_tdp_kw:.0f} kW")
    print(f"Rack EDP: {VR_NVL72_MAX_P_CONFIG.rack_edp_kva:.0f} kVA")
    print(f"Total GPU Power: {VR_NVL72_MAX_P_CONFIG.total_gpu_power / 1000:.1f} kW")
    print(f"Total HBM: {VR_NVL72_MAX_P_CONFIG.hbm_total_tb:.1f} TB")
    print(f"Energy Storage: {VR_NVL72_MAX_P_CONFIG.total_energy_storage_j:.0f} J")
    print(f"Floor Loading: {VR_NVL72_MAX_P_CONFIG.floor_loading_kg_m2:.0f} kg/m²")
    print(f"Power Shelf Redundancy: {VR_NVL72_MAX_P_CONFIG.power_shelf_redundancy}")
    
    # Simulate training workload
    profile = vr_gpu.simulate_training_workload(
        model_params=70e9,
        batch_size=32,
        duration=10.0
    )
    
    print(f"\n=== 70B Model Training Simulation ===")
    print(f"Mean power: {profile.mean_power:.1f} W")
    print(f"Peak power: {profile.peak_power:.1f} W")
    print(f"Energy: {profile.energy_kwh:.4f} kWh")
"""
