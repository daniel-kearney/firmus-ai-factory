"""
Synthetic Workload Generator for Firmus AI Factory

Generates realistic LLM inference workload profiles based on real H200 benchmark data
from the firmus-model-evaluation framework. Supports parametric and stochastic generation.

Author: Dr. Daniel Kearney
Date: February 2026
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from .profiles import (
    WorkloadProfile, PhaseCharacteristics, EnergyMetrics, ThermalProfile,
    WorkloadPhase, ModelTier
)


@dataclass
class ModelBenchmarkData:
    """
    Real benchmark data from firmus-model-evaluation framework.
    Source: TEMPORAL_POWER_ANALYSIS.md
    """
    model_name: str
    model_size_params: int
    ramp_rate_ws: float  # W/s
    fall_rate_ws: float  # W/s (negative)
    prefill_peak_w: float
    steady_avg_w: float
    steady_stdev_w: float
    steady_cv: float
    time_to_stable_s: float
    tier: ModelTier
    
    # Derived parameters
    idle_power_w: float = 50.0  # H200 idle baseline
    prefill_duration_s: float = 1.0  # Typical prefill duration
    
    @property
    def ramp_duration_s(self) -> float:
        """Calculate ramp-up duration based on ramp rate"""
        return (self.steady_avg_w - self.idle_power_w) / self.ramp_rate_ws
    
    @property
    def fall_duration_s(self) -> float:
        """Calculate fall-off duration based on fall rate"""
        return (self.steady_avg_w - self.idle_power_w) / abs(self.fall_rate_ws)


# Real benchmark data from TEMPORAL_POWER_ANALYSIS.md
BENCHMARK_MODELS = {
    'qwen3-235b': ModelBenchmarkData(
        model_name='Qwen3-235B-A22B-Instruct',
        model_size_params=235_000_000_000,
        ramp_rate_ws=768.9,
        fall_rate_ws=-312.6,
        prefill_peak_w=827.9,
        steady_avg_w=680.7,
        steady_stdev_w=21.4,
        steady_cv=0.0315,
        time_to_stable_s=2.0,
        tier=ModelTier.TIER_1_EFFICIENT
    ),
    'deepseek-r1-distill-32b': ModelBenchmarkData(
        model_name='DeepSeek-R1-Distill-Qwen-32B',
        model_size_params=32_000_000_000,
        ramp_rate_ws=576.9,
        fall_rate_ws=-232.6,
        prefill_peak_w=868.7,
        steady_avg_w=519.3,
        steady_stdev_w=86.8,
        steady_cv=0.1672,
        time_to_stable_s=2.0,
        tier=ModelTier.TIER_3_HIGH_VARIANCE
    ),
    'qwen3-32b': ModelBenchmarkData(
        model_name='Qwen3-32B-Instruct',
        model_size_params=32_000_000_000,
        ramp_rate_ws=546.9,
        fall_rate_ws=-220.1,
        prefill_peak_w=647.2,
        steady_avg_w=492.9,
        steady_stdev_w=36.4,
        steady_cv=0.0739,
        time_to_stable_s=2.0,
        tier=ModelTier.TIER_1_EFFICIENT
    ),
    'gpt-oss-20b': ModelBenchmarkData(
        model_name='GPT-OSS-20B',
        model_size_params=20_000_000_000,
        ramp_rate_ws=534.9,
        fall_rate_ws=-215.1,
        prefill_peak_w=581.5,
        steady_avg_w=485.1,
        steady_stdev_w=14.2,
        steady_cv=0.0293,
        time_to_stable_s=2.0,
        tier=ModelTier.TIER_1_EFFICIENT
    ),
    'llama4-maverick': ModelBenchmarkData(
        model_name='Llama-4-Maverick-17B-128E-Instruct',
        model_size_params=17_000_000_000,
        ramp_rate_ws=486.9,
        fall_rate_ws=-195.1,
        prefill_peak_w=594.4,
        steady_avg_w=445.3,
        steady_stdev_w=37.3,
        steady_cv=0.0838,
        time_to_stable_s=2.0,
        tier=ModelTier.TIER_1_EFFICIENT
    ),
    'llama4-scout': ModelBenchmarkData(
        model_name='Llama-4-Scout-17B-16E',
        model_size_params=17_000_000_000,
        ramp_rate_ws=456.9,
        fall_rate_ws=-182.6,
        prefill_peak_w=597.0,
        steady_avg_w=424.6,
        steady_stdev_w=52.1,
        steady_cv=0.1227,
        time_to_stable_s=2.0,
        tier=ModelTier.TIER_2_STANDARD
    ),
}


class SyntheticWorkloadGenerator:
    """
    Generate synthetic workload profiles based on real benchmark data.
    
    Supports both deterministic and stochastic generation modes.
    """
    
    def __init__(self, model_key: str, gpu_platform: str = "H200", cooling_type: str = "immersion"):
        """
        Initialize generator with model benchmark data.
        
        Args:
            model_key: Key from BENCHMARK_MODELS dict
            gpu_platform: GPU platform (e.g., "H200", "GB300")
            cooling_type: Cooling system type (e.g., "immersion", "liquid_cdu")
        """
        if model_key not in BENCHMARK_MODELS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(BENCHMARK_MODELS.keys())}")
        
        self.model_data = BENCHMARK_MODELS[model_key]
        self.gpu_platform = gpu_platform
        self.cooling_type = cooling_type
    
    def generate_deterministic(self, 
                              decode_duration_s: float = 6.0,
                              sampling_interval_s: float = 0.1) -> WorkloadProfile:
        """
        Generate deterministic workload profile with exact benchmark characteristics.
        
        Args:
            decode_duration_s: Duration of decode (steady-state) phase
            sampling_interval_s: Time resolution for power trace
        
        Returns:
            WorkloadProfile with deterministic power trace
        """
        # Phase timing
        t_idle_start = 0.0
        t_ramp_start = 0.0
        t_ramp_end = t_ramp_start + self.model_data.ramp_duration_s
        t_prefill_start = t_ramp_end
        t_prefill_end = t_prefill_start + self.model_data.prefill_duration_s
        t_decode_start = t_prefill_end
        t_decode_end = t_decode_start + decode_duration_s
        t_fall_start = t_decode_end
        t_fall_end = t_fall_start + self.model_data.fall_duration_s
        
        total_duration = t_fall_end
        
        # Generate power trace
        power_trace = []
        t = 0.0
        
        while t <= total_duration:
            if t < t_ramp_end:
                # Ramp phase: linear increase
                power = self.model_data.idle_power_w + self.model_data.ramp_rate_ws * t
            elif t < t_prefill_end:
                # Prefill phase: peak power
                power = self.model_data.prefill_peak_w
            elif t < t_decode_end:
                # Decode phase: steady-state with variance
                power = self.model_data.steady_avg_w
            elif t < t_fall_end:
                # Fall phase: linear decrease
                dt = t - t_fall_start
                power = self.model_data.steady_avg_w + self.model_data.fall_rate_ws * dt
            else:
                # Return to idle
                power = self.model_data.idle_power_w
            
            power_trace.append((t, max(power, self.model_data.idle_power_w)))
            t += sampling_interval_s
        
        # Create phase characteristics
        phases = self._create_phases(
            t_ramp_start, t_ramp_end,
            t_prefill_start, t_prefill_end,
            t_decode_start, t_decode_end,
            t_fall_start, t_fall_end,
            power_trace
        )
        
        # Calculate energy metrics
        energy_metrics = self._calculate_energy_metrics(power_trace, decode_duration_s)
        
        return WorkloadProfile(
            model_name=self.model_data.model_name,
            model_size_params=self.model_data.model_size_params,
            gpu_platform=self.gpu_platform,
            cooling_type=self.cooling_type,
            duration_seconds=total_duration,
            power_trace=power_trace,
            phases=phases,
            energy_metrics=energy_metrics,
            tier=self.model_data.tier
        )
    
    def generate_stochastic(self,
                           decode_duration_s: float = 6.0,
                           sampling_interval_s: float = 0.1,
                           seed: Optional[int] = None) -> WorkloadProfile:
        """
        Generate stochastic workload profile with realistic power variance.
        
        Adds gaussian noise to decode phase based on benchmark CV.
        
        Args:
            decode_duration_s: Duration of decode (steady-state) phase
            sampling_interval_s: Time resolution for power trace
            seed: Random seed for reproducibility
        
        Returns:
            WorkloadProfile with stochastic power trace
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Start with deterministic profile
        profile = self.generate_deterministic(decode_duration_s, sampling_interval_s)
        
        # Add stochastic variance to decode phase
        stochastic_trace = []
        for t, p in profile.power_trace:
            phase = profile.get_phase_at_time(t)
            
            if phase == WorkloadPhase.DECODE:
                # Add gaussian noise based on benchmark CV
                noise = np.random.normal(0, self.model_data.steady_stdev_w)
                p_noisy = p + noise
                # Clamp to reasonable bounds
                p_noisy = max(p_noisy, self.model_data.steady_avg_w * 0.5)
                p_noisy = min(p_noisy, self.model_data.prefill_peak_w)
                stochastic_trace.append((t, p_noisy))
            else:
                stochastic_trace.append((t, p))
        
        # Update power trace
        profile.power_trace = stochastic_trace
        
        # Recalculate phases with new trace
        t_ramp_end = self.model_data.ramp_duration_s
        t_prefill_end = t_ramp_end + self.model_data.prefill_duration_s
        t_decode_end = t_prefill_end + decode_duration_s
        t_fall_end = t_decode_end + self.model_data.fall_duration_s
        
        profile.phases = self._create_phases(
            0.0, t_ramp_end,
            t_ramp_end, t_prefill_end,
            t_prefill_end, t_decode_end,
            t_decode_end, t_fall_end,
            stochastic_trace
        )
        
        # Recalculate energy metrics
        profile.energy_metrics = self._calculate_energy_metrics(stochastic_trace, decode_duration_s)
        
        return profile
    
    def _create_phases(self,
                      t_ramp_start: float, t_ramp_end: float,
                      t_prefill_start: float, t_prefill_end: float,
                      t_decode_start: float, t_decode_end: float,
                      t_fall_start: float, t_fall_end: float,
                      power_trace: List[Tuple[float, float]]) -> Dict[str, PhaseCharacteristics]:
        """Create phase characteristics from power trace"""
        
        def extract_phase_stats(start: float, end: float) -> Tuple[float, float, float, float, float]:
            """Extract power statistics for a phase"""
            phase_powers = [p for t, p in power_trace if start <= t <= end]
            if not phase_powers:
                return 0, 0, 0, 0, 0
            avg = np.mean(phase_powers)
            peak = np.max(phase_powers)
            stdev = np.std(phase_powers)
            cv = stdev / avg if avg > 0 else 0
            energy = avg * (end - start)
            return avg, peak, stdev, cv, energy
        
        phases = {}
        
        # Ramp phase
        avg, peak, stdev, cv, energy = extract_phase_stats(t_ramp_start, t_ramp_end)
        phases['ramp'] = PhaseCharacteristics(
            phase=WorkloadPhase.RAMP,
            start_time=t_ramp_start,
            end_time=t_ramp_end,
            avg_power=avg,
            peak_power=peak,
            power_stdev=stdev,
            power_cv=cv,
            energy_joules=energy,
            ramp_rate_ws=self.model_data.ramp_rate_ws
        )
        
        # Prefill phase
        avg, peak, stdev, cv, energy = extract_phase_stats(t_prefill_start, t_prefill_end)
        phases['prefill'] = PhaseCharacteristics(
            phase=WorkloadPhase.PREFILL,
            start_time=t_prefill_start,
            end_time=t_prefill_end,
            avg_power=avg,
            peak_power=peak,
            power_stdev=stdev,
            power_cv=cv,
            energy_joules=energy
        )
        
        # Decode phase
        avg, peak, stdev, cv, energy = extract_phase_stats(t_decode_start, t_decode_end)
        phases['decode'] = PhaseCharacteristics(
            phase=WorkloadPhase.DECODE,
            start_time=t_decode_start,
            end_time=t_decode_end,
            avg_power=avg,
            peak_power=peak,
            power_stdev=stdev,
            power_cv=cv,
            energy_joules=energy
        )
        
        # Fall phase
        avg, peak, stdev, cv, energy = extract_phase_stats(t_fall_start, t_fall_end)
        phases['fall'] = PhaseCharacteristics(
            phase=WorkloadPhase.FALL,
            start_time=t_fall_start,
            end_time=t_fall_end,
            avg_power=avg,
            peak_power=peak,
            power_stdev=stdev,
            power_cv=cv,
            energy_joules=energy,
            ramp_rate_ws=self.model_data.fall_rate_ws
        )
        
        return phases
    
    def _calculate_energy_metrics(self, 
                                  power_trace: List[Tuple[float, float]],
                                  decode_duration_s: float) -> EnergyMetrics:
        """Calculate energy metrics from power trace"""
        
        # Trapezoidal integration for total energy
        total_energy_j = 0.0
        for i in range(len(power_trace) - 1):
            t1, p1 = power_trace[i]
            t2, p2 = power_trace[i + 1]
            dt = t2 - t1
            avg_power = (p1 + p2) / 2
            total_energy_j += avg_power * dt
        
        # Estimate tokens generated (based on typical inference rates)
        # Assume ~100 tokens/second during decode phase
        tokens_per_second = 100
        tokens_generated = int(decode_duration_s * tokens_per_second)
        
        # Calculate metrics
        duration = power_trace[-1][0] if power_trace else 0
        powers = [p for _, p in power_trace]
        avg_power = np.mean(powers) if powers else 0
        peak_power = np.max(powers) if powers else 0
        
        joules_per_token = total_energy_j / tokens_generated if tokens_generated > 0 else 0
        tokens_per_joule = tokens_generated / total_energy_j if total_energy_j > 0 else 0
        wh_per_1k_queries = (joules_per_token * 100 * 1000) / 3600
        
        return EnergyMetrics(
            total_energy_joules=total_energy_j,
            tokens_generated=tokens_generated,
            joules_per_token=joules_per_token,
            tokens_per_joule=tokens_per_joule,
            avg_power_watts=avg_power,
            peak_power_watts=peak_power,
            duration_seconds=duration,
            wh_per_1k_queries=wh_per_1k_queries
        )


def generate_workload(model_key: str,
                     duration_hours: float = 1.0,
                     stochastic: bool = True,
                     gpu_platform: str = "H200",
                     cooling_type: str = "immersion",
                     seed: Optional[int] = None) -> WorkloadProfile:
    """
    Convenience function to generate workload profile.
    
    Args:
        model_key: Model identifier from BENCHMARK_MODELS
        duration_hours: Total workload duration in hours
        stochastic: Use stochastic generation (adds realistic variance)
        gpu_platform: GPU platform
        cooling_type: Cooling system type
        seed: Random seed for stochastic generation
    
    Returns:
        WorkloadProfile ready for infrastructure simulation
    
    Example:
        >>> profile = generate_workload('deepseek-r1-distill-32b', duration_hours=10.0)
        >>> profile.to_json('workloads/deepseek_10h.json')
    """
    generator = SyntheticWorkloadGenerator(model_key, gpu_platform, cooling_type)
    
    # Calculate decode duration (total minus overhead phases)
    model_data = BENCHMARK_MODELS[model_key]
    overhead = (model_data.ramp_duration_s + 
                model_data.prefill_duration_s + 
                model_data.fall_duration_s)
    decode_duration_s = (duration_hours * 3600) - overhead
    
    if decode_duration_s < 0:
        raise ValueError(f"Duration too short for overhead phases ({overhead:.1f}s)")
    
    if stochastic:
        return generator.generate_stochastic(decode_duration_s, seed=seed)
    else:
        return generator.generate_deterministic(decode_duration_s)
