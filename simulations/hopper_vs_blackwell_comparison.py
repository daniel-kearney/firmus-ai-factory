"""
Hopper (H200) vs Blackwell (GB200) Power Profile Comparison

Demonstrates the impact of Blackwell power smoothing on:
- Grid stability (ramp rate reduction)
- UPS stress (battery mode trigger reduction)
- Energy overhead (power floor cost)
- Infrastructure sizing (transformer/UPS capacity)

Author: daniel.kearney@firmus.co
Date: February 2026
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import json

from firmus_ai_factory.workload import (
    generate_workload,
    BENCHMARK_MODELS,
    apply_blackwell_smoothing,
    PRESET_PROFILES,
    estimate_ups_stress_reduction,
    estimate_grid_stability_improvement,
)


def run_comparison_simulation(
    model_name: str = "deepseek-r1-distill-32b",
    duration_hours: float = 10.0,
    num_gpus: int = 8,  # Single HGX/NVL baseboard
    output_dir: str = "simulations/output"
):
    """
    Run Hopper vs Blackwell comparison for a specific workload.
    
    Args:
        model_name: Model to simulate
        duration_hours: Simulation duration
        num_gpus: Number of GPUs in the system
        output_dir: Output directory for results
    """
    print(f"\n{'='*80}")
    print(f"Hopper (H200) vs Blackwell (GB200) Power Smoothing Comparison")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Duration: {duration_hours} hours")
    print(f"GPUs: {num_gpus}")
    print(f"{'='*80}\n")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate base workload (Hopper-like, no smoothing)
    print("[1/6] Generating base workload profile...")
    workload_profile = generate_workload(
        model_key=model_name,
        duration_hours=duration_hours,
        stochastic=False,
        seed=42
    )
    
    # Extract power trace from profile
    # power_trace is List[Tuple[float, float]] = [(time_s, power_w), ...]
    power_trace = np.array(workload_profile.power_trace)
    time_s = power_trace[:, 0]
    hopper_power_w = power_trace[:, 1]
    dt = time_s[1] - time_s[0]
    
    print(f"  ✓ Generated {len(time_s)} time points ({dt:.1f}s resolution)")
    print(f"  ✓ Hopper peak power: {np.max(hopper_power_w):.1f} W")
    print(f"  ✓ Hopper avg power: {np.mean(hopper_power_w):.1f} W")
    
    # Scale to multi-GPU system
    hopper_power_w_system = hopper_power_w * num_gpus
    
    # Apply Blackwell smoothing with different profiles
    print("\n[2/6] Applying Blackwell power smoothing profiles...")
    blackwell_results = {}
    
    for profile_id in [1, 2, 3]:  # Conservative, Moderate, Aggressive
        profile = PRESET_PROFILES[profile_id]
        print(f"  → Profile {profile_id}: {profile.profile_name}")
        print(f"    Power floor: {profile.power_floor_pct}% TGP")
        print(f"    Ramp rates: {profile.ramp_up_rate_w_per_s:.0f} W/s up, {profile.ramp_down_rate_w_per_s:.0f} W/s down")
        
        # Scale profile to multi-GPU system
        profile_scaled = PRESET_PROFILES[profile_id]
        profile_scaled.power_ceiling_w = np.max(hopper_power_w_system)
        profile_scaled.ramp_up_rate_w_per_s *= num_gpus
        profile_scaled.ramp_down_rate_w_per_s *= num_gpus
        
        smoothed_power, metrics = apply_blackwell_smoothing(
            hopper_power_w_system,
            profile=profile_scaled,
            dt=dt
        )
        
        blackwell_results[profile_id] = {
            'profile': profile,
            'power_trace_w': smoothed_power,
            'metrics': metrics
        }
        
        print(f"    ✓ Energy overhead: +{metrics['energy_overhead_pct']:.1f}%")
        print(f"    ✓ Ramp rate reduction: -{metrics['ramp_rate_reduction_pct']:.1f}%")
        print(f"    ✓ Power swing reduction: -{metrics['power_swing_reduction_pct']:.1f}%")
    
    # Estimate UPS stress reduction
    print("\n[3/6] Estimating UPS stress reduction...")
    ups_capacity_w = np.max(hopper_power_w_system) * 1.2  # 20% headroom
    
    for profile_id, result in blackwell_results.items():
        ups_metrics = estimate_ups_stress_reduction(
            hopper_power_w_system,
            result['power_trace_w'],
            ups_capacity_w=ups_capacity_w,
            ups_battery_trigger_pct=50.0,
            dt=dt
        )
        result['ups_metrics'] = ups_metrics
        
        print(f"  Profile {profile_id} ({result['profile'].profile_name}):")
        print(f"    Battery triggers: {ups_metrics['raw_battery_triggers']} → {ups_metrics['smoothed_battery_triggers']} (-{ups_metrics['battery_trigger_reduction_pct']:.1f}%)")
        print(f"    UPS lifetime: {ups_metrics['raw_ups_lifetime_years']:.1f} → {ups_metrics['smoothed_ups_lifetime_years']:.1f} years (+{ups_metrics['ups_lifetime_extension_years']:.1f} years)")
    
    # Estimate grid stability improvement
    print("\n[4/6] Estimating grid stability improvement...")
    grid_ramp_limit_w_per_min = 50000.0  # 50 kW/min typical utility limit
    
    for profile_id, result in blackwell_results.items():
        grid_metrics = estimate_grid_stability_improvement(
            hopper_power_w_system,
            result['power_trace_w'],
            grid_ramp_limit_w_per_min=grid_ramp_limit_w_per_min,
            dt=dt
        )
        result['grid_metrics'] = grid_metrics
        
        print(f"  Profile {profile_id} ({result['profile'].profile_name}):")
        print(f"    Grid violations: {grid_metrics['raw_grid_violations']} → {grid_metrics['smoothed_grid_violations']} (-{grid_metrics['grid_violation_reduction_pct']:.1f}%)")
        print(f"    Max ramp rate: {grid_metrics['raw_max_ramp_rate_w_per_min']:.0f} → {grid_metrics['smoothed_max_ramp_rate_w_per_min']:.0f} W/min")
    
    # Generate visualizations
    print("\n[5/6] Generating visualizations...")
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Convert time to hours for plotting
    time_h = time_s / 3600.0
    
    # Plot 1: Power traces comparison (full duration)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_h, hopper_power_w_system / 1000, 'k-', linewidth=1.5, label='Hopper (H200) - No Smoothing', alpha=0.7)
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    for i, (profile_id, result) in enumerate(blackwell_results.items()):
        ax1.plot(time_h, result['power_trace_w'] / 1000, color=colors[i], linewidth=1.5, 
                label=f"Blackwell Profile {profile_id}: {result['profile'].profile_name}", alpha=0.8)
    ax1.set_xlabel('Time (hours)', fontsize=12)
    ax1.set_ylabel('System Power (kW)', fontsize=12)
    ax1.set_title(f'{model_name.upper()} - {num_gpus}× GPU System Power Profiles', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Zoomed startup transient (first 5 minutes)
    ax2 = fig.add_subplot(gs[1, 0])
    zoom_duration_s = 300  # 5 minutes
    zoom_mask = time_s <= zoom_duration_s
    ax2.plot(time_s[zoom_mask], hopper_power_w_system[zoom_mask] / 1000, 'k-', linewidth=2, label='Hopper', alpha=0.7)
    for i, (profile_id, result) in enumerate(blackwell_results.items()):
        ax2.plot(time_s[zoom_mask], result['power_trace_w'][zoom_mask] / 1000, color=colors[i], linewidth=2, alpha=0.8)
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_ylabel('Power (kW)', fontsize=11)
    ax2.set_title('Startup Transient (First 5 min)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Ramp rate comparison
    ax3 = fig.add_subplot(gs[1, 1])
    profiles = ['Hopper'] + [f"P{pid}" for pid in blackwell_results.keys()]
    ramp_rates = [np.max(np.abs(np.diff(hopper_power_w_system))) / dt / 1000]
    for result in blackwell_results.values():
        ramp_rates.append(np.max(np.abs(np.diff(result['power_trace_w']))) / dt / 1000)
    bars = ax3.bar(profiles, ramp_rates, color=['black'] + colors, alpha=0.7)
    ax3.axhline(grid_ramp_limit_w_per_min / 60 / 1000, color='red', linestyle='--', linewidth=2, label='Grid Limit')
    ax3.set_ylabel('Max Ramp Rate (kW/s)', fontsize=11)
    ax3.set_title('Peak Ramp Rate Comparison', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Energy overhead
    ax4 = fig.add_subplot(gs[1, 2])
    energy_overhead = [0] + [result['metrics']['energy_overhead_pct'] for result in blackwell_results.values()]
    bars = ax4.bar(profiles, energy_overhead, color=['black'] + colors, alpha=0.7)
    ax4.set_ylabel('Energy Overhead (%)', fontsize=11)
    ax4.set_title('Energy Cost of Power Smoothing', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, energy_overhead)):
        if val > 0:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'+{val:.1f}%', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 5: UPS battery triggers
    ax5 = fig.add_subplot(gs[2, 0])
    hopper_triggers = blackwell_results[1]['ups_metrics']['raw_battery_triggers']
    blackwell_triggers = [result['ups_metrics']['smoothed_battery_triggers'] for result in blackwell_results.values()]
    triggers = [hopper_triggers] + blackwell_triggers
    bars = ax5.bar(profiles, triggers, color=['black'] + colors, alpha=0.7)
    ax5.set_ylabel('UPS Battery Mode Triggers', fontsize=11)
    ax5.set_title('UPS Stress Reduction', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: UPS lifetime extension
    ax6 = fig.add_subplot(gs[2, 1])
    hopper_lifetime = blackwell_results[1]['ups_metrics']['raw_ups_lifetime_years']
    blackwell_lifetimes = [result['ups_metrics']['smoothed_ups_lifetime_years'] for result in blackwell_results.values()]
    lifetimes = [hopper_lifetime] + blackwell_lifetimes
    bars = ax6.bar(profiles, lifetimes, color=['black'] + colors, alpha=0.7)
    ax6.set_ylabel('UPS Lifetime (years)', fontsize=11)
    ax6.set_title('UPS Lifetime Extension', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Plot 7: Grid violations
    ax7 = fig.add_subplot(gs[2, 2])
    hopper_violations = blackwell_results[1]['grid_metrics']['raw_grid_violations']
    blackwell_violations = [result['grid_metrics']['smoothed_grid_violations'] for result in blackwell_results.values()]
    violations = [hopper_violations] + blackwell_violations
    bars = ax7.bar(profiles, violations, color=['black'] + colors, alpha=0.7)
    ax7.set_ylabel('Grid Ramp Limit Violations', fontsize=11)
    ax7.set_title('Grid Stability Improvement', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Add overall title
    fig.suptitle(f'Hopper (H200) vs Blackwell (GB200) Power Smoothing Analysis\n{model_name} | {duration_hours}h | {num_gpus}× GPUs', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = os.path.join(output_dir, 'hopper_vs_blackwell_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved visualization: {output_path}")
    plt.close()
    
    # Export summary JSON
    print("\n[6/6] Exporting summary data...")
    summary = {
        'simulation_metadata': {
            'model': model_name,
            'duration_hours': duration_hours,
            'num_gpus': num_gpus,
            'timestamp': datetime.now().isoformat(),
        },
        'hopper_h200': {
            'peak_power_w': float(np.max(hopper_power_w_system)),
            'avg_power_w': float(np.mean(hopper_power_w_system)),
            'energy_wh': float(np.sum(hopper_power_w_system) * dt / 3600),
            'max_ramp_rate_w_per_s': float(np.max(np.abs(np.diff(hopper_power_w_system))) / dt),
            'ups_battery_triggers': int(hopper_triggers),
            'ups_lifetime_years': float(hopper_lifetime),
            'grid_violations': int(hopper_violations),
        },
        'blackwell_gb200': {}
    }
    
    for profile_id, result in blackwell_results.items():
        summary['blackwell_gb200'][f'profile_{profile_id}'] = {
            'profile_name': result['profile'].profile_name,
            'power_floor_pct': result['profile'].power_floor_pct,
            'ramp_up_rate_w_per_s': result['profile'].ramp_up_rate_w_per_s,
            'ramp_down_rate_w_per_s': result['profile'].ramp_down_rate_w_per_s,
            'peak_power_w': float(result['metrics']['smoothed_peak_power_w']),
            'avg_power_w': float(result['metrics']['smoothed_avg_power_w']),
            'energy_wh': float(result['metrics']['smoothed_energy_wh']),
            'energy_overhead_pct': float(result['metrics']['energy_overhead_pct']),
            'max_ramp_rate_w_per_s': float(result['metrics']['smoothed_ramp_rate_w_per_s']),
            'ramp_rate_reduction_pct': float(result['metrics']['ramp_rate_reduction_pct']),
            'power_swing_reduction_pct': float(result['metrics']['power_swing_reduction_pct']),
            'ups_battery_triggers': int(result['ups_metrics']['smoothed_battery_triggers']),
            'ups_trigger_reduction_pct': float(result['ups_metrics']['battery_trigger_reduction_pct']),
            'ups_lifetime_years': float(result['ups_metrics']['smoothed_ups_lifetime_years']),
            'ups_lifetime_extension_years': float(result['ups_metrics']['ups_lifetime_extension_years']),
            'grid_violations': int(result['grid_metrics']['smoothed_grid_violations']),
            'grid_violation_reduction_pct': float(result['grid_metrics']['grid_violation_reduction_pct']),
        }
    
    summary_path = os.path.join(output_dir, 'hopper_vs_blackwell_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Saved summary: {summary_path}")
    
    print(f"\n{'='*80}")
    print("Simulation complete!")
    print(f"{'='*80}\n")
    
    return summary


if __name__ == "__main__":
    summary = run_comparison_simulation(
        model_name="deepseek-r1-distill-32b",
        duration_hours=10.0,
        num_gpus=8,
        output_dir="simulations/output"
    )
