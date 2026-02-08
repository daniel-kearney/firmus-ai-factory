"""
DeepSeek R1-Distill-32B 10-Hour Workload Demonstration
Showcasing Synthetic Workload Generation from Real H200 Benchmarks

This demonstration shows how the firmus-ai-factory workload module generates
realistic inference workload profiles based on real benchmark data from the
firmus-model-evaluation framework.

Author: Dr. Daniel Kearney
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

# Import Firmus AI Factory workload module
from firmus_ai_factory.workload import generate_workload, BENCHMARK_MODELS, WorkloadPhase

# Simulation parameters
WORKLOAD_DURATION_HOURS = 10.0
MODEL_KEY = 'deepseek-r1-distill-32b'
ELECTRICITY_RATE_USD_PER_KWH = 0.12  # Singapore industrial rate

print("="*80)
print("DeepSeek R1-Distill-32B: 10-Hour Workload Profile Demonstration")
print("="*80)
print(f"Model: {BENCHMARK_MODELS[MODEL_KEY].model_name}")
print(f"Duration: {WORKLOAD_DURATION_HOURS} hours")
print(f"Based on: Real H200 benchmark data from firmus-model-evaluation")
print("="*80)
print()

# ============================================================================
# Generate Workload Profile
# ============================================================================
print("[1/4] Generating workload profile...")

workload = generate_workload(
    model_key=MODEL_KEY,
    duration_hours=WORKLOAD_DURATION_HOURS,
    stochastic=True,  # Include realistic power variance
    gpu_platform='H200',
    cooling_type='immersion',
    seed=42  # Reproducible results
)

print(f"  ✓ Generated {len(workload.power_trace)} time points")
print(f"  ✓ Peak power: {workload.energy_metrics.peak_power_watts:.1f} W")
print(f"  ✓ Average power: {workload.energy_metrics.avg_power_watts:.1f} W")
print(f"  ✓ Total energy: {workload.energy_metrics.total_energy_joules/3.6e6:.2f} kWh")
print(f"  ✓ Tokens generated: {workload.energy_metrics.tokens_generated:,}")
print(f"  ✓ Energy efficiency: {workload.energy_metrics.joules_per_token:.4f} J/token")
print(f"  ✓ Model tier: {workload.tier.value}")
print()

# ============================================================================
# Analyze Phases
# ============================================================================
print("[2/4] Analyzing temporal phases...")

for phase_name, phase in workload.phases.items():
    print(f"\n  {phase_name.upper()} Phase:")
    print(f"    Duration:     {phase.duration:.2f} s")
    print(f"    Avg Power:    {phase.avg_power:.1f} W")
    print(f"    Peak Power:   {phase.peak_power:.1f} W")
    print(f"    Power CV:     {phase.power_cv:.4f}")
    print(f"    Energy:       {phase.energy_joules:.2f} J")
    if phase.ramp_rate_ws:
        print(f"    Ramp Rate:    {phase.ramp_rate_ws:.1f} W/s")

print()

# ============================================================================
# Economic Analysis
# ============================================================================
print("[3/4] Calculating economics...")

# Energy cost
energy_kwh = workload.energy_metrics.total_energy_joules / 3.6e6
base_cost_usd = energy_kwh * ELECTRICITY_RATE_USD_PER_KWH

# Model-to-Grid discount
if workload.tier.value == 'tier_1_efficient':
    discount_pct = 20.0
elif workload.tier.value == 'tier_2_standard':
    discount_pct = 10.0
else:
    discount_pct = 0.0

discounted_cost_usd = base_cost_usd * (1 - discount_pct / 100)
savings_usd = base_cost_usd - discounted_cost_usd

# Cost per token
cost_per_million_tokens = (base_cost_usd / workload.energy_metrics.tokens_generated) * 1_000_000
discounted_cost_per_million_tokens = (discounted_cost_usd / workload.energy_metrics.tokens_generated) * 1_000_000

print(f"  Energy consumed: {energy_kwh:.4f} kWh")
print(f"  Base cost: ${base_cost_usd:.4f}")
print(f"  Model tier: {workload.tier.value} ({discount_pct}% discount)")
print(f"  Discounted cost: ${discounted_cost_usd:.4f}")
print(f"  Savings: ${savings_usd:.4f}")
print(f"  Cost per M tokens: ${cost_per_million_tokens:.4f} → ${discounted_cost_per_million_tokens:.4f}")
print()

# ============================================================================
# Generate Visualization
# ============================================================================
print("[4/4] Generating visualizations...")

fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# Extract time series data
times_hours = np.array([t/3600 for t, _ in workload.power_trace])
powers_w = np.array([p for _, p in workload.power_trace])

# Color scheme
color_power = '#FF6B6B'
color_ramp = '#4ECDC4'
color_prefill = '#FFA07A'
color_decode = '#95E1D3'
color_fall = '#A8E6CF'

# Plot 1: Full 10-hour power profile
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(times_hours, powers_w, color=color_power, linewidth=1, alpha=0.8)

# Highlight phases
for phase_name, phase in workload.phases.items():
    start_h = phase.start_time / 3600
    end_h = phase.end_time / 3600
    
    if phase.phase == WorkloadPhase.RAMP:
        color = color_ramp
    elif phase.phase == WorkloadPhase.PREFILL:
        color = color_prefill
    elif phase.phase == WorkloadPhase.DECODE:
        color = color_decode
    elif phase.phase == WorkloadPhase.FALL:
        color = color_fall
    else:
        continue
    
    ax1.axvspan(start_h, end_h, alpha=0.2, color=color, label=phase_name.capitalize())

ax1.axhline(y=BENCHMARK_MODELS[MODEL_KEY].steady_avg_w, color='gray', linestyle='--', alpha=0.5, label='Steady Average')
ax1.axhline(y=BENCHMARK_MODELS[MODEL_KEY].prefill_peak_w, color='red', linestyle='--', alpha=0.5, label='Prefill Peak')
ax1.set_xlabel('Time (hours)', fontsize=12)
ax1.set_ylabel('GPU Power (W)', fontsize=12)
ax1.set_title('DeepSeek R1-Distill-32B: 10-Hour Power Profile (Single GPU)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', ncol=3, fontsize=9)
ax1.set_xlim(0, WORKLOAD_DURATION_HOURS)

# Plot 2: Zoom on first 10 seconds (startup transient)
ax2 = fig.add_subplot(gs[1, 0])
mask_startup = times_hours <= (10/3600)
ax2.plot(times_hours[mask_startup]*3600, powers_w[mask_startup], color=color_power, linewidth=2)
ax2.set_xlabel('Time (seconds)', fontsize=11)
ax2.set_ylabel('GPU Power (W)', fontsize=11)
ax2.set_title('Startup Transient (0-10s)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 10)

# Annotate phases
ax2.axvline(x=workload.phases['ramp'].end_time, color=color_ramp, linestyle='--', alpha=0.7, label='Ramp End')
ax2.axvline(x=workload.phases['prefill'].end_time, color=color_prefill, linestyle='--', alpha=0.7, label='Prefill End')
ax2.legend(fontsize=9)

# Plot 3: Power histogram
ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(powers_w, bins=50, color=color_power, alpha=0.7, edgecolor='black')
ax3.axvline(x=workload.energy_metrics.avg_power_watts, color='blue', linestyle='--', linewidth=2, label=f'Mean: {workload.energy_metrics.avg_power_watts:.1f}W')
ax3.axvline(x=workload.energy_metrics.peak_power_watts, color='red', linestyle='--', linewidth=2, label=f'Peak: {workload.energy_metrics.peak_power_watts:.1f}W')
ax3.set_xlabel('Power (W)', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.set_title('Power Distribution', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'DeepSeek R1-Distill-32B Workload Profile\nBased on Real H200 Benchmark Data | Tier: {workload.tier.value}',
             fontsize=16, fontweight='bold', y=0.995)

output_dir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'deepseek_10h_workload_profile.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  ✓ Visualization saved: {output_path}")

# ============================================================================
# Export Results
# ============================================================================

# Export workload profile as JSON
workload_json_path = os.path.join(output_dir, 'deepseek_10h_workload.json')
workload.to_json(workload_json_path)
print(f"  ✓ Workload JSON saved: {workload_json_path}")

# Export summary
summary = {
    'metadata': {
        'model': BENCHMARK_MODELS[MODEL_KEY].model_name,
        'model_size_params': BENCHMARK_MODELS[MODEL_KEY].model_size_params,
        'duration_hours': WORKLOAD_DURATION_HOURS,
        'gpu_platform': workload.gpu_platform,
        'cooling_type': workload.cooling_type,
        'timestamp': datetime.now().isoformat()
    },
    'benchmark_data': {
        'ramp_rate_ws': BENCHMARK_MODELS[MODEL_KEY].ramp_rate_ws,
        'fall_rate_ws': BENCHMARK_MODELS[MODEL_KEY].fall_rate_ws,
        'prefill_peak_w': BENCHMARK_MODELS[MODEL_KEY].prefill_peak_w,
        'steady_avg_w': BENCHMARK_MODELS[MODEL_KEY].steady_avg_w,
        'steady_stdev_w': BENCHMARK_MODELS[MODEL_KEY].steady_stdev_w,
        'steady_cv': BENCHMARK_MODELS[MODEL_KEY].steady_cv,
        'tier': workload.tier.value
    },
    'energy_metrics': {
        'total_energy_joules': workload.energy_metrics.total_energy_joules,
        'total_energy_kwh': energy_kwh,
        'tokens_generated': workload.energy_metrics.tokens_generated,
        'joules_per_token': workload.energy_metrics.joules_per_token,
        'tokens_per_joule': workload.energy_metrics.tokens_per_joule,
        'avg_power_watts': workload.energy_metrics.avg_power_watts,
        'peak_power_watts': workload.energy_metrics.peak_power_watts
    },
    'economics': {
        'electricity_rate_usd_per_kwh': ELECTRICITY_RATE_USD_PER_KWH,
        'base_cost_usd': base_cost_usd,
        'discount_pct': discount_pct,
        'discounted_cost_usd': discounted_cost_usd,
        'savings_usd': savings_usd,
        'cost_per_million_tokens': discounted_cost_per_million_tokens
    },
    'phases': {
        name: {
            'duration_s': phase.duration,
            'avg_power_w': phase.avg_power,
            'peak_power_w': phase.peak_power,
            'power_cv': phase.power_cv,
            'energy_joules': phase.energy_joules,
            'ramp_rate_ws': phase.ramp_rate_ws
        }
        for name, phase in workload.phases.items()
    }
}

summary_path = os.path.join(output_dir, 'deepseek_10h_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  ✓ Summary saved: {summary_path}")

print()
print("="*80)
print("DEMONSTRATION COMPLETE")
print("="*80)
print(f"Workload Duration: {WORKLOAD_DURATION_HOURS} hours")
print(f"Total Energy: {energy_kwh:.4f} kWh")
print(f"Total Cost: ${discounted_cost_usd:.4f} (after {discount_pct}% discount)")
print(f"Tokens Generated: {workload.energy_metrics.tokens_generated:,}")
print(f"Cost per M tokens: ${discounted_cost_per_million_tokens:.4f}")
print(f"Energy Efficiency: {workload.energy_metrics.joules_per_token:.4f} J/token")
print("="*80)
print()
print("This workload profile can now be used in:")
print("  • Cooling system simulations (thermal response)")
print("  • Electrical infrastructure sizing (peak demand, ramp rates)")
print("  • Economic modeling (TCO, Model-to-Grid discounts)")
print("  • Grid integration analysis (demand response, frequency regulation)")
print("="*80)
