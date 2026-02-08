"""
Blackwell Power Smoothing Module

Implements NVIDIA Blackwell GPU power smoothing feature for workload generation.
Based on NVIDIA DA-12033-001_v03 Application Note (September 2025).

Author: daniel.kearney@firmus.co
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class BlackwellPowerSmoothingProfile:
    """
    Blackwell power smoothing configuration profile.
    
    Attributes:
        enabled: Whether power smoothing is active
        power_floor_pct: Minimum power as % of TGP (0-90%)
        power_ceiling_w: Maximum power ceiling in watts (TGP value)
        ramp_up_rate_w_per_s: Ramp-up rate in watts per second (up to TGP)
        ramp_down_rate_w_per_s: Ramp-down rate in watts per second (up to TGP)
        hysteresis_ms: Delay before ramp-down in milliseconds
        profile_id: Preset profile identifier (0-4)
        profile_name: Human-readable profile name
    """
    enabled: bool = False
    power_floor_pct: float = 0.0  # 0-90% of TGP
    power_ceiling_w: float = 1000.0  # TGP value (GB200 default)
    ramp_up_rate_w_per_s: float = 1000.0  # Up to TGP W/s
    ramp_down_rate_w_per_s: float = 1000.0  # Up to TGP W/s
    hysteresis_ms: float = 2000.0  # Delay before ramp-down
    profile_id: int = 0  # 0-4 preset profiles
    profile_name: str = "Disabled"
    
    def __post_init__(self):
        """Validate profile parameters."""
        if not 0 <= self.power_floor_pct <= 90:
            raise ValueError(f"Power floor must be 0-90%, got {self.power_floor_pct}%")
        if self.ramp_up_rate_w_per_s > self.power_ceiling_w:
            raise ValueError(f"Ramp-up rate ({self.ramp_up_rate_w_per_s} W/s) cannot exceed TGP ({self.power_ceiling_w} W)")
        if self.ramp_down_rate_w_per_s > self.power_ceiling_w:
            raise ValueError(f"Ramp-down rate ({self.ramp_down_rate_w_per_s} W/s) cannot exceed TGP ({self.power_ceiling_w} W)")
        if not 0 <= self.profile_id <= 4:
            raise ValueError(f"Profile ID must be 0-4, got {self.profile_id}")


# Preset profiles for common use cases
PRESET_PROFILES = {
    0: BlackwellPowerSmoothingProfile(
        enabled=False,
        power_floor_pct=0.0,
        power_ceiling_w=1000.0,
        ramp_up_rate_w_per_s=1000.0,
        ramp_down_rate_w_per_s=1000.0,
        hysteresis_ms=0.0,
        profile_id=0,
        profile_name="Disabled (Hopper-like)"
    ),
    1: BlackwellPowerSmoothingProfile(
        enabled=True,
        power_floor_pct=90.0,
        power_ceiling_w=1000.0,
        ramp_up_rate_w_per_s=70.0,
        ramp_down_rate_w_per_s=70.0,
        hysteresis_ms=5000.0,
        profile_id=1,
        profile_name="Conservative (Grid-friendly)"
    ),
    2: BlackwellPowerSmoothingProfile(
        enabled=True,
        power_floor_pct=70.0,
        power_ceiling_w=1000.0,
        ramp_up_rate_w_per_s=200.0,
        ramp_down_rate_w_per_s=200.0,
        hysteresis_ms=2000.0,
        profile_id=2,
        profile_name="Moderate (UPS-optimized)"
    ),
    3: BlackwellPowerSmoothingProfile(
        enabled=True,
        power_floor_pct=50.0,
        power_ceiling_w=1000.0,
        ramp_up_rate_w_per_s=500.0,
        ramp_down_rate_w_per_s=500.0,
        hysteresis_ms=1000.0,
        profile_id=3,
        profile_name="Aggressive (Minimal overhead)"
    ),
    4: BlackwellPowerSmoothingProfile(
        enabled=True,
        power_floor_pct=80.0,
        power_ceiling_w=1000.0,
        ramp_up_rate_w_per_s=150.0,
        ramp_down_rate_w_per_s=150.0,
        hysteresis_ms=3000.0,
        profile_id=4,
        profile_name="Custom (Balanced)"
    ),
}


class BlackwellPowerSmoother:
    """
    Applies Blackwell power smoothing constraints to raw power traces.
    
    Implements the power smoothing algorithm described in NVIDIA DA-12033-001_v03:
    1. Apply power floor clamp (minimum power regardless of workload)
    2. Apply ramp rate limits (controlled power transitions)
    3. Apply hysteresis delay for ramp-down (prevent oscillation)
    4. Model power burn circuit overhead (extra power with no performance)
    """
    
    def __init__(self, profile: BlackwellPowerSmoothingProfile, dt: float = 1.0):
        """
        Initialize power smoother.
        
        Args:
            profile: Power smoothing configuration profile
            dt: Time step in seconds (default 1.0s)
        """
        self.profile = profile
        self.dt = dt
        self.power_floor_w = profile.power_ceiling_w * (profile.power_floor_pct / 100.0)
        self.hysteresis_steps = int(profile.hysteresis_ms / (dt * 1000.0))
        self.ramp_down_counter = 0
        
    def apply_smoothing(self, power_trace_w: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Apply power smoothing to raw power trace.
        
        Args:
            power_trace_w: Raw power trace in watts (1D array)
            
        Returns:
            Tuple of (smoothed_power_w, metrics_dict)
            - smoothed_power_w: Power trace with smoothing applied
            - metrics_dict: Dictionary of smoothing metrics
        """
        if not self.profile.enabled:
            return power_trace_w, self._compute_metrics(power_trace_w, power_trace_w)
        
        n_steps = len(power_trace_w)
        smoothed_power = np.zeros(n_steps)
        smoothed_power[0] = max(power_trace_w[0], self.power_floor_w)
        
        ramp_down_delayed_steps = 0
        
        for i in range(1, n_steps):
            target_power = power_trace_w[i]
            current_power = smoothed_power[i-1]
            
            # Apply power floor
            target_power = max(target_power, self.power_floor_w)
            
            # Compute desired change
            delta_power = target_power - current_power
            
            # Apply ramp rate limits
            if delta_power > 0:
                # Ramp up: apply ramp-up rate limit
                max_delta = self.profile.ramp_up_rate_w_per_s * self.dt
                delta_power = min(delta_power, max_delta)
                self.ramp_down_counter = 0  # Reset hysteresis counter
            elif delta_power < 0:
                # Ramp down: apply hysteresis delay
                if self.ramp_down_counter < self.hysteresis_steps:
                    # Still in hysteresis period, maintain current power
                    delta_power = 0
                    self.ramp_down_counter += 1
                    ramp_down_delayed_steps += 1
                else:
                    # Hysteresis expired, apply ramp-down rate limit
                    max_delta = self.profile.ramp_down_rate_w_per_s * self.dt
                    delta_power = max(delta_power, -max_delta)
            
            # Apply change
            smoothed_power[i] = current_power + delta_power
            
            # Enforce ceiling
            smoothed_power[i] = min(smoothed_power[i], self.profile.power_ceiling_w)
        
        metrics = self._compute_metrics(power_trace_w, smoothed_power)
        metrics['ramp_down_delayed_steps'] = ramp_down_delayed_steps
        metrics['hysteresis_activation_pct'] = 100.0 * ramp_down_delayed_steps / n_steps
        
        return smoothed_power, metrics
    
    def _compute_metrics(self, raw_power: np.ndarray, smoothed_power: np.ndarray) -> dict:
        """Compute smoothing performance metrics."""
        raw_energy_wh = np.sum(raw_power) * (self.dt / 3600.0)
        smoothed_energy_wh = np.sum(smoothed_power) * (self.dt / 3600.0)
        energy_overhead_wh = smoothed_energy_wh - raw_energy_wh
        energy_overhead_pct = 100.0 * energy_overhead_wh / raw_energy_wh if raw_energy_wh > 0 else 0.0
        
        raw_ramp_rate = np.max(np.abs(np.diff(raw_power))) / self.dt
        smoothed_ramp_rate = np.max(np.abs(np.diff(smoothed_power))) / self.dt
        ramp_rate_reduction_pct = 100.0 * (1.0 - smoothed_ramp_rate / raw_ramp_rate) if raw_ramp_rate > 0 else 0.0
        
        raw_power_swing = np.max(raw_power) - np.min(raw_power)
        smoothed_power_swing = np.max(smoothed_power) - np.min(smoothed_power)
        power_swing_reduction_pct = 100.0 * (1.0 - smoothed_power_swing / raw_power_swing) if raw_power_swing > 0 else 0.0
        
        return {
            'raw_energy_wh': raw_energy_wh,
            'smoothed_energy_wh': smoothed_energy_wh,
            'energy_overhead_wh': energy_overhead_wh,
            'energy_overhead_pct': energy_overhead_pct,
            'raw_peak_power_w': np.max(raw_power),
            'smoothed_peak_power_w': np.max(smoothed_power),
            'raw_min_power_w': np.min(raw_power),
            'smoothed_min_power_w': np.min(smoothed_power),
            'raw_avg_power_w': np.mean(raw_power),
            'smoothed_avg_power_w': np.mean(smoothed_power),
            'raw_ramp_rate_w_per_s': raw_ramp_rate,
            'smoothed_ramp_rate_w_per_s': smoothed_ramp_rate,
            'ramp_rate_reduction_pct': ramp_rate_reduction_pct,
            'raw_power_swing_w': raw_power_swing,
            'smoothed_power_swing_w': smoothed_power_swing,
            'power_swing_reduction_pct': power_swing_reduction_pct,
            'power_floor_w': self.power_floor_w,
            'power_ceiling_w': self.profile.power_ceiling_w,
        }


def apply_blackwell_smoothing(
    power_trace_w: np.ndarray,
    profile: Optional[BlackwellPowerSmoothingProfile] = None,
    profile_id: Optional[int] = None,
    dt: float = 1.0
) -> tuple[np.ndarray, dict]:
    """
    Convenience function to apply Blackwell power smoothing.
    
    Args:
        power_trace_w: Raw power trace in watts
        profile: Custom profile (overrides profile_id)
        profile_id: Preset profile ID (0-4), default 0 (disabled)
        dt: Time step in seconds
        
    Returns:
        Tuple of (smoothed_power_w, metrics_dict)
    """
    if profile is None:
        if profile_id is None:
            profile_id = 0
        profile = PRESET_PROFILES[profile_id]
    
    smoother = BlackwellPowerSmoother(profile, dt=dt)
    return smoother.apply_smoothing(power_trace_w)


def estimate_ups_stress_reduction(
    raw_power_w: np.ndarray,
    smoothed_power_w: np.ndarray,
    ups_capacity_w: float,
    ups_battery_trigger_pct: float = 50.0,
    dt: float = 1.0
) -> dict:
    """
    Estimate UPS stress reduction from power smoothing.
    
    Args:
        raw_power_w: Raw power trace
        smoothed_power_w: Smoothed power trace
        ups_capacity_w: UPS capacity in watts
        ups_battery_trigger_pct: % power swing that triggers battery mode
        dt: Time step in seconds
        
    Returns:
        Dictionary with UPS stress metrics
    """
    # Compute power swings (step changes)
    raw_swings = np.abs(np.diff(raw_power_w))
    smoothed_swings = np.abs(np.diff(smoothed_power_w))
    
    # Count battery mode triggers (swing > threshold)
    swing_threshold = ups_capacity_w * (ups_battery_trigger_pct / 100.0)
    raw_triggers = np.sum(raw_swings > swing_threshold)
    smoothed_triggers = np.sum(smoothed_swings > swing_threshold)
    
    # Estimate battery cycles (each trigger = 1 cycle)
    raw_battery_cycles = raw_triggers
    smoothed_battery_cycles = smoothed_triggers
    battery_cycle_reduction = raw_battery_cycles - smoothed_battery_cycles
    battery_cycle_reduction_pct = 100.0 * battery_cycle_reduction / raw_battery_cycles if raw_battery_cycles > 0 else 0.0
    
    # Estimate UPS lifetime extension (typical UPS rated for 200-500 cycles)
    typical_ups_cycle_life = 300
    raw_ups_lifetime_years = typical_ups_cycle_life / (raw_battery_cycles / (len(raw_power_w) * dt / (365.25 * 24 * 3600)))
    smoothed_ups_lifetime_years = typical_ups_cycle_life / (smoothed_battery_cycles / (len(smoothed_power_w) * dt / (365.25 * 24 * 3600))) if smoothed_battery_cycles > 0 else float('inf')
    ups_lifetime_extension_years = smoothed_ups_lifetime_years - raw_ups_lifetime_years
    
    return {
        'ups_capacity_w': ups_capacity_w,
        'battery_trigger_threshold_w': swing_threshold,
        'raw_battery_triggers': raw_triggers,
        'smoothed_battery_triggers': smoothed_triggers,
        'battery_trigger_reduction': raw_triggers - smoothed_triggers,
        'battery_trigger_reduction_pct': battery_cycle_reduction_pct,
        'raw_battery_cycles': raw_battery_cycles,
        'smoothed_battery_cycles': smoothed_battery_cycles,
        'battery_cycle_reduction': battery_cycle_reduction,
        'battery_cycle_reduction_pct': battery_cycle_reduction_pct,
        'raw_ups_lifetime_years': raw_ups_lifetime_years,
        'smoothed_ups_lifetime_years': smoothed_ups_lifetime_years,
        'ups_lifetime_extension_years': ups_lifetime_extension_years,
    }


def estimate_grid_stability_improvement(
    raw_power_w: np.ndarray,
    smoothed_power_w: np.ndarray,
    grid_ramp_limit_w_per_min: float = 10000.0,
    dt: float = 1.0
) -> dict:
    """
    Estimate grid stability improvement from power smoothing.
    
    Args:
        raw_power_w: Raw power trace
        smoothed_power_w: Smoothed power trace
        grid_ramp_limit_w_per_min: Grid ramp rate limit in W/min
        dt: Time step in seconds
        
    Returns:
        Dictionary with grid stability metrics
    """
    # Convert to W/min for comparison with grid limits
    raw_ramp_rates_w_per_min = np.abs(np.diff(raw_power_w)) / dt * 60.0
    smoothed_ramp_rates_w_per_min = np.abs(np.diff(smoothed_power_w)) / dt * 60.0
    
    # Count grid limit violations
    raw_violations = np.sum(raw_ramp_rates_w_per_min > grid_ramp_limit_w_per_min)
    smoothed_violations = np.sum(smoothed_ramp_rates_w_per_min > grid_ramp_limit_w_per_min)
    
    # Compute severity of violations
    raw_violation_severity = np.sum(np.maximum(0, raw_ramp_rates_w_per_min - grid_ramp_limit_w_per_min))
    smoothed_violation_severity = np.sum(np.maximum(0, smoothed_ramp_rates_w_per_min - grid_ramp_limit_w_per_min))
    
    return {
        'grid_ramp_limit_w_per_min': grid_ramp_limit_w_per_min,
        'raw_max_ramp_rate_w_per_min': np.max(raw_ramp_rates_w_per_min),
        'smoothed_max_ramp_rate_w_per_min': np.max(smoothed_ramp_rates_w_per_min),
        'raw_grid_violations': raw_violations,
        'smoothed_grid_violations': smoothed_violations,
        'grid_violation_reduction': raw_violations - smoothed_violations,
        'grid_violation_reduction_pct': 100.0 * (raw_violations - smoothed_violations) / raw_violations if raw_violations > 0 else 0.0,
        'raw_violation_severity_w_per_min': raw_violation_severity,
        'smoothed_violation_severity_w_per_min': smoothed_violation_severity,
        'violation_severity_reduction_pct': 100.0 * (raw_violation_severity - smoothed_violation_severity) / raw_violation_severity if raw_violation_severity > 0 else 0.0,
    }
