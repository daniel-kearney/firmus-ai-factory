"""GPU Power and Performance Modeling.

Mathematical models for GPU power consumption and computational
performance based on workload characteristics.

Key equations:
    P_GPU(t) = P_compute(t) + P_memory(t) + P_transfer(t)
    P_train(t) = P_base + sum(alpha_i * f_i(t)) + epsilon(t)
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


@dataclass
class GPUSpecifications:
    """Hardware specifications for GPU accelerators."""
    name: str
    tdp_watts: float  # Thermal Design Power
    peak_flops_fp16: float  # Peak FP16 TFLOPS
    peak_flops_fp32: float  # Peak FP32 TFLOPS
    hbm_bandwidth_tb_s: float  # Memory bandwidth TB/s
    hbm_capacity_gb: float  # HBM capacity GB
    nvlink_bandwidth_gb_s: float  # NVLink bandwidth GB/s
    idle_power_fraction: float = 0.15  # Fraction of TDP at idle
    base_power_fraction: float = 0.65  # Base power during compute


# Pre-defined GPU specifications
H100_SXM_SPECS = GPUSpecifications(
    name="NVIDIA H100 SXM",
    tdp_watts=700.0,
    peak_flops_fp16=1979.0,  # TFLOPS
    peak_flops_fp32=989.5,
    hbm_bandwidth_tb_s=3.35,
    hbm_capacity_gb=80.0,
    nvlink_bandwidth_gb_s=900.0,
    idle_power_fraction=0.12,
    base_power_fraction=0.70
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
    base_power_fraction=0.68
)

B200_SPECS = GPUSpecifications(
    name="NVIDIA B200",
    tdp_watts=1000.0,
    peak_flops_fp16=4500.0,
    peak_flops_fp32=2250.0,
    hbm_bandwidth_tb_s=8.0,
    hbm_capacity_gb=192.0,
    nvlink_bandwidth_gb_s=1800.0,
    idle_power_fraction=0.10,
    base_power_fraction=0.65
)


@dataclass
class PowerProfile:
    """Time-series power consumption profile."""
    time: np.ndarray  # Time points (seconds)
    total_power: np.ndarray  # Total power (Watts)
    compute_power: np.ndarray  # Compute component
    memory_power: np.ndarray  # Memory component
    transfer_power: np.ndarray  # Data transfer component
    
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
            'forward': 0.85,      # Forward pass intensity
            'backward': 0.95,     # Backward pass intensity  
            'gradient_sync': 0.60,  # All-reduce communication
            'optimizer': 0.75     # Optimizer step
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
    # Example usage
    gpu = GPUModel(H100_SXM_SPECS)
    
    # Simulate 70B model training
    profile = gpu.simulate_training_workload(
        model_params=70e9,
        batch_size=32,
        duration=10.0
    )
    
    print(f"Mean power: {profile.mean_power:.1f} W")
    print(f"Peak power: {profile.peak_power:.1f} W")
