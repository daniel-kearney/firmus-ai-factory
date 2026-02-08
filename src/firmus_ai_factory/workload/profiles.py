"""
Workload Profile Data Structures for Firmus AI Factory Digital Twin

This module defines workload profiles based on real H200 benchmark data from
the firmus-model-evaluation framework. Profiles capture temporal power consumption
patterns for LLM inference workloads, enabling accurate infrastructure simulation.

Author: Dr. Daniel Kearney
Date: February 2026
"""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional
from enum import Enum
import json
import numpy as np


class WorkloadPhase(Enum):
    """Temporal phases in LLM inference workload"""
    IDLE = "idle"
    RAMP = "ramp"
    PREFILL = "prefill"
    DECODE = "decode"
    FALL = "fall"


class ModelTier(Enum):
    """Model-to-Grid pricing tiers based on power stability"""
    TIER_1_EFFICIENT = "tier_1_efficient"  # CV < 0.10, <150W avg
    TIER_2_STANDARD = "tier_2_standard"    # CV < 0.15, <200W avg
    TIER_3_HIGH_VARIANCE = "tier_3_high_variance"  # CV > 0.15 or >200W


@dataclass
class PhaseCharacteristics:
    """
    Characteristics of a temporal phase in the workload.
    
    Attributes:
        phase: Phase type (idle, ramp, prefill, decode, fall)
        start_time: Phase start time in seconds
        end_time: Phase end time in seconds
        avg_power: Average power consumption in watts
        peak_power: Peak power consumption in watts
        power_stdev: Standard deviation of power in watts
        power_cv: Coefficient of variation (stdev/mean)
        energy_joules: Total energy consumed in this phase
        ramp_rate_ws: Power ramp rate in W/s (for ramp/fall phases)
    """
    phase: WorkloadPhase
    start_time: float
    end_time: float
    avg_power: float
    peak_power: float
    power_stdev: float
    power_cv: float
    energy_joules: float
    ramp_rate_ws: Optional[float] = None
    
    @property
    def duration(self) -> float:
        """Phase duration in seconds"""
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['phase'] = self.phase.value
        return result


@dataclass
class ThermalProfile:
    """
    Thermal characteristics of the workload.
    
    Attributes:
        avg_temp_celsius: Average GPU temperature
        peak_temp_celsius: Peak GPU temperature
        temp_rise_rate: Temperature rise rate in °C/s
        thermal_time_constant: Time constant for thermal response in seconds
    """
    avg_temp_celsius: float
    peak_temp_celsius: float
    temp_rise_rate: float
    thermal_time_constant: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EnergyMetrics:
    """
    Energy efficiency metrics for the workload.
    
    Attributes:
        total_energy_joules: Total energy consumed
        tokens_generated: Total tokens generated
        joules_per_token: Energy efficiency (J/token)
        tokens_per_joule: Inverse efficiency metric
        avg_power_watts: Average power consumption
        peak_power_watts: Peak power consumption
        duration_seconds: Total workload duration
        wh_per_1k_queries: Scalability metric (Wh per 1000 queries @ 100 tokens/query)
    """
    total_energy_joules: float
    tokens_generated: int
    joules_per_token: float
    tokens_per_joule: float
    avg_power_watts: float
    peak_power_watts: float
    duration_seconds: float
    wh_per_1k_queries: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class WorkloadProfile:
    """
    Complete workload profile for infrastructure simulation.
    
    Based on real H200 benchmark data from firmus-model-evaluation framework.
    Includes temporal power trace, phase characteristics, and energy metrics.
    
    Attributes:
        model_name: Name of the LLM model
        model_size_params: Model size in parameters
        gpu_platform: GPU platform (e.g., "H200", "GB300")
        cooling_type: Cooling system type (e.g., "immersion", "liquid_cdu")
        duration_seconds: Total workload duration
        power_trace: Time-series power data [(timestamp, power_watts), ...]
        phases: Phase characteristics by phase name
        energy_metrics: Energy efficiency metrics
        tier: Model-to-Grid pricing tier
        thermal_profile: Optional thermal characteristics
    """
    model_name: str
    model_size_params: int
    gpu_platform: str
    cooling_type: str
    duration_seconds: float
    power_trace: List[Tuple[float, float]]
    phases: Dict[str, PhaseCharacteristics]
    energy_metrics: EnergyMetrics
    tier: ModelTier
    thermal_profile: Optional[ThermalProfile] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'model_name': self.model_name,
            'model_size_params': self.model_size_params,
            'gpu_platform': self.gpu_platform,
            'cooling_type': self.cooling_type,
            'duration_seconds': self.duration_seconds,
            'power_trace': self.power_trace,
            'phases': {name: phase.to_dict() for name, phase in self.phases.items()},
            'energy_metrics': self.energy_metrics.to_dict(),
            'tier': self.tier.value,
            'thermal_profile': self.thermal_profile.to_dict() if self.thermal_profile else None
        }
    
    def to_json(self, filepath: str):
        """Export workload profile as JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'WorkloadProfile':
        """Load workload profile from JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct phases
        phases = {}
        for name, phase_data in data['phases'].items():
            phase_data['phase'] = WorkloadPhase(phase_data['phase'])
            phases[name] = PhaseCharacteristics(**phase_data)
        
        # Reconstruct energy metrics
        energy_metrics = EnergyMetrics(**data['energy_metrics'])
        
        # Reconstruct thermal profile if present
        thermal_profile = None
        if data['thermal_profile']:
            thermal_profile = ThermalProfile(**data['thermal_profile'])
        
        # Reconstruct tier
        tier = ModelTier(data['tier'])
        
        return cls(
            model_name=data['model_name'],
            model_size_params=data['model_size_params'],
            gpu_platform=data['gpu_platform'],
            cooling_type=data['cooling_type'],
            duration_seconds=data['duration_seconds'],
            power_trace=data['power_trace'],
            phases=phases,
            energy_metrics=energy_metrics,
            tier=tier,
            thermal_profile=thermal_profile
        )
    
    def get_power_at_time(self, t: float) -> float:
        """
        Get power consumption at specific time via interpolation.
        
        Args:
            t: Time in seconds
        
        Returns:
            Power in watts
        """
        if not self.power_trace:
            return 0.0
        
        # Find surrounding points
        times = [pt[0] for pt in self.power_trace]
        powers = [pt[1] for pt in self.power_trace]
        
        if t <= times[0]:
            return powers[0]
        if t >= times[-1]:
            return powers[-1]
        
        # Linear interpolation
        return np.interp(t, times, powers)
    
    def get_phase_at_time(self, t: float) -> Optional[WorkloadPhase]:
        """
        Get workload phase at specific time.
        
        Args:
            t: Time in seconds
        
        Returns:
            WorkloadPhase or None if outside workload duration
        """
        for phase_char in self.phases.values():
            if phase_char.start_time <= t <= phase_char.end_time:
                return phase_char.phase
        return None
    
    def scale_duration(self, new_duration: float) -> 'WorkloadProfile':
        """
        Scale workload to new duration while preserving power characteristics.
        
        Useful for extending short benchmark runs to long production workloads.
        Primarily extends the decode phase while keeping ramp/prefill/fall unchanged.
        
        Args:
            new_duration: New total duration in seconds
        
        Returns:
            New WorkloadProfile with scaled duration
        """
        # Calculate decode phase extension
        decode_phase = self.phases.get('decode')
        if not decode_phase:
            raise ValueError("Cannot scale workload without decode phase")
        
        original_decode_duration = decode_phase.duration
        other_phases_duration = self.duration_seconds - original_decode_duration
        new_decode_duration = new_duration - other_phases_duration
        
        if new_decode_duration < 0:
            raise ValueError(f"New duration {new_duration}s too short for non-decode phases ({other_phases_duration}s)")
        
        # Scale power trace
        scale_factor = new_decode_duration / original_decode_duration
        new_power_trace = []
        
        for t, p in self.power_trace:
            phase = self.get_phase_at_time(t)
            if phase == WorkloadPhase.DECODE:
                # Scale decode phase timestamps
                offset = decode_phase.start_time
                new_t = offset + (t - offset) * scale_factor
                new_power_trace.append((new_t, p))
            elif phase and t > decode_phase.end_time:
                # Shift post-decode phases
                shift = new_decode_duration - original_decode_duration
                new_power_trace.append((t + shift, p))
            else:
                # Keep pre-decode phases unchanged
                new_power_trace.append((t, p))
        
        # Update phases
        new_phases = {}
        for name, phase in self.phases.items():
            if phase.phase == WorkloadPhase.DECODE:
                # Extend decode phase
                new_energy = phase.avg_power * new_decode_duration
                new_phases[name] = PhaseCharacteristics(
                    phase=phase.phase,
                    start_time=phase.start_time,
                    end_time=phase.start_time + new_decode_duration,
                    avg_power=phase.avg_power,
                    peak_power=phase.peak_power,
                    power_stdev=phase.power_stdev,
                    power_cv=phase.power_cv,
                    energy_joules=new_energy,
                    ramp_rate_ws=phase.ramp_rate_ws
                )
            elif phase.start_time > decode_phase.end_time:
                # Shift post-decode phases
                shift = new_decode_duration - original_decode_duration
                new_phases[name] = PhaseCharacteristics(
                    phase=phase.phase,
                    start_time=phase.start_time + shift,
                    end_time=phase.end_time + shift,
                    avg_power=phase.avg_power,
                    peak_power=phase.peak_power,
                    power_stdev=phase.power_stdev,
                    power_cv=phase.power_cv,
                    energy_joules=phase.energy_joules,
                    ramp_rate_ws=phase.ramp_rate_ws
                )
            else:
                # Keep pre-decode phases unchanged
                new_phases[name] = phase
        
        # Update energy metrics
        total_energy = sum(p.energy_joules for p in new_phases.values())
        new_tokens = int(self.energy_metrics.tokens_generated * (new_duration / self.duration_seconds))
        
        new_energy_metrics = EnergyMetrics(
            total_energy_joules=total_energy,
            tokens_generated=new_tokens,
            joules_per_token=total_energy / new_tokens if new_tokens > 0 else 0,
            tokens_per_joule=new_tokens / total_energy if total_energy > 0 else 0,
            avg_power_watts=total_energy / new_duration,
            peak_power_watts=self.energy_metrics.peak_power_watts,
            duration_seconds=new_duration,
            wh_per_1k_queries=self.energy_metrics.wh_per_1k_queries
        )
        
        return WorkloadProfile(
            model_name=self.model_name,
            model_size_params=self.model_size_params,
            gpu_platform=self.gpu_platform,
            cooling_type=self.cooling_type,
            duration_seconds=new_duration,
            power_trace=new_power_trace,
            phases=new_phases,
            energy_metrics=new_energy_metrics,
            tier=self.tier,
            thermal_profile=self.thermal_profile
        )
