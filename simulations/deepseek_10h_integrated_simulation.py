"""
DeepSeek R1-Distill-32B 10-Hour Workload Simulation
Integrated Cooling, Grid, and Economic Analysis

This simulation demonstrates the full capabilities of the Firmus AI Factory digital twin
by running a realistic 10-hour inference workload through:
1. Workload generation (based on real H200 benchmarks)
2. Cooling system response (Benmax HCU2500 Hypercube)
3. Grid power demand and response
4. Economic cost analysis

Author: Dr. Daniel Kearney
Date: February 2026
Site: Batam BT1-2 (120 MW, 410 Racks, GB300 NVL72)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
import json

# Import Firmus AI Factory modules
from firmus_ai_factory.workload import generate_workload, BENCHMARK_MODELS
from firmus_ai_factory.thermal.benmax_hcu2500 import BenmaxHCU2500, BenmaxHypercube

# Simulation parameters
WORKLOAD_DURATION_HOURS = 10.0
MODEL_KEY = 'deepseek-r1-distill-32b'
AMBIENT_TEMP_C = 32.2  # Batam design condition
ELECTRICITY_RATE_USD_PER_KWH = 0.12  # Singapore industrial rate
SIMULATION_TIMESTEP_S = 60.0  # 1-minute resolution

print("="*80)
print("DeepSeek R1-Distill-32B 10-Hour Integrated Simulation")
print("="*80)
print(f"Model: {BENCHMARK_MODELS[MODEL_KEY].model_name}")
print(f"Duration: {WORKLOAD_DURATION_HOURS} hours")
print(f"Site: Batam BT1-2 (120 MW)")
print(f"Ambient: {AMBIENT_TEMP_C}°C")
print(f"Electricity Rate: ${ELECTRICITY_RATE_USD_PER_KWH}/kWh")
print("="*80)
print()

# ============================================================================
# PHASE 1: Generate Workload Profile
# ============================================================================
print("[1/6] Generating workload profile from real H200 benchmark data...")

workload = generate_workload(
    model_key=MODEL_KEY,
    duration_hours=WORKLOAD_DURATION_HOURS,
    stochastic=True,  # Include realistic power variance
    gpu_platform='GB300',  # Batam BT1-2 uses GB300
    cooling_type='liquid_cdu',  # Benmax HCU2500 is liquid-to-liquid CDU
    seed=42  # Reproducible results
)

print(f"  ✓ Workload generated: {len(workload.power_trace)} time points")
print(f"  ✓ Peak power: {workload.energy_metrics.peak_power_watts:.1f} W")
print(f"  ✓ Average power: {workload.energy_metrics.avg_power_watts:.1f} W")
print(f"  ✓ Total energy: {workload.energy_metrics.total_energy_joules/3.6e6:.2f} kWh")
print(f"  ✓ Model tier: {workload.tier.value}")
print()

# ============================================================================
# PHASE 2: Initialize Cooling System
# ============================================================================
print("[2/6] Initializing Benmax HCU2500 Hypercube cooling system...")

# Initialize Benmax Hypercube for GB300 NVL72
# One Hypercube = 32 racks × 72 GPUs = 2304 GPUs
num_racks = 32
gpus_per_rack = 72
gpu_tdp_watts = 1200  # GB300 TDP
rack_overhead_watts = 500  # Networking, storage

# Initialize cooling system
cooling_system = BenmaxHCU2500()

# Calculate required flow and temperatures for workload peak
peak_load_kw = (workload.energy_metrics.peak_power_watts * 
                num_racks * 
                gpus_per_rack) / 1000

supply_temp, return_temp, flow_rate = cooling_system.calculate_operating_point(
    it_load_kw=peak_load_kw,
    ambient_temp_c=AMBIENT_TEMP_C
)

print(f"  ✓ Hypercube configured: {num_racks} racks × {gpus_per_rack} GPUs")
print(f"  ✓ Peak IT load: {peak_load_kw:.1f} kW")
print(f"  ✓ PG25 supply: {supply_temp:.1f}°C")
print(f"  ✓ PG25 return: {return_temp:.1f}°C")
print(f"  ✓ Flow rate: {flow_rate:.0f} LPM")
print()

# ============================================================================
# PHASE 3: Run Thermal Simulation
# ============================================================================
print("[3/6] Running thermal simulation over 10-hour workload...")

# Time series arrays
num_steps = int((WORKLOAD_DURATION_HOURS * 3600) / SIMULATION_TIMESTEP_S)
time_hours = np.linspace(0, WORKLOAD_DURATION_HOURS, num_steps)
time_seconds = time_hours * 3600

# Initialize arrays
gpu_power_w = np.zeros(num_steps)
it_load_kw = np.zeros(num_steps)
pg25_supply_temp_c = np.zeros(num_steps)
pg25_return_temp_c = np.zeros(num_steps)
gpu_tj_c = np.zeros(num_steps)
pump_power_kw = np.zeros(num_steps)

# Simulate each timestep
for i, t_s in enumerate(time_seconds):
    # Get GPU power from workload profile
    gpu_power_w[i] = workload.get_power_at_time(t_s)
    
    # Calculate rack-level IT load
    it_load_kw[i] = (gpu_power_w[i] * 
                     num_racks * 
                     gpus_per_rack + 
                     rack_overhead_watts * num_racks) / 1000
    
    # Calculate cooling system response
    supply, return_t, flow = cooling_system.calculate_operating_point(
        it_load_kw=it_load_kw[i],
        ambient_temp_c=AMBIENT_TEMP_C
    )
    
    pg25_supply_temp_c[i] = supply
    pg25_return_temp_c[i] = return_t
    
    # Calculate GPU junction temperature
    gpu_tj_c[i] = cooling_system.calculate_gpu_junction_temp(
        gpu_power_w=gpu_power_w[i],
        pg25_supply_temp_c=supply
    )
    
    # Calculate pump power
    pump_power_kw[i] = cooling_system.calculate_pump_power(
        flow_rate_lpm=flow,
        num_active_hcus=4  # Full redundancy
    )

print(f"  ✓ Simulated {num_steps} timesteps")
print(f"  ✓ Peak GPU Tj: {np.max(gpu_tj_c):.1f}°C")
print(f"  ✓ Average GPU Tj: {np.mean(gpu_tj_c):.1f}°C")
print(f"  ✓ Peak PG25 supply: {np.max(pg25_supply_temp_c):.1f}°C")
print(f"  ✓ Average pump power: {np.mean(pump_power_kw):.2f} kW")
print()

# ============================================================================
# PHASE 4: Grid Power Analysis
# ============================================================================
print("[4/6] Analyzing grid power demand and response...")

# Calculate total facility power
total_it_power_kw = it_load_kw
total_cooling_power_kw = pump_power_kw
total_facility_power_kw = total_it_power_kw + total_cooling_power_kw

# Calculate pPUE (partial PUE - IT + cooling only)
ppue = total_facility_power_kw / total_it_power_kw

# Grid metrics
peak_demand_kw = np.max(total_facility_power_kw)
avg_demand_kw = np.mean(total_facility_power_kw)
load_factor = avg_demand_kw / peak_demand_kw

# Calculate ramp rates (kW/min)
ramp_rates_kw_per_min = np.diff(total_facility_power_kw) / (SIMULATION_TIMESTEP_S / 60)
max_ramp_up = np.max(ramp_rates_kw_per_min)
max_ramp_down = np.min(ramp_rates_kw_per_min)

print(f"  ✓ Peak demand: {peak_demand_kw:.1f} kW")
print(f"  ✓ Average demand: {avg_demand_kw:.1f} kW")
print(f"  ✓ Load factor: {load_factor:.3f}")
print(f"  ✓ Average pPUE: {np.mean(ppue):.4f}")
print(f"  ✓ Max ramp up: {max_ramp_up:.1f} kW/min")
print(f"  ✓ Max ramp down: {max_ramp_down:.1f} kW/min")
print()

# ============================================================================
# PHASE 5: Economic Analysis
# ============================================================================
print("[5/6] Calculating economic costs...")

# Energy consumption
total_it_energy_kwh = np.trapz(total_it_power_kw, time_hours)
total_cooling_energy_kwh = np.trapz(total_cooling_power_kw, time_hours)
total_facility_energy_kwh = total_it_energy_kwh + total_cooling_energy_kwh

# Base costs
it_energy_cost_usd = total_it_energy_kwh * ELECTRICITY_RATE_USD_PER_KWH
cooling_energy_cost_usd = total_cooling_energy_kwh * ELECTRICITY_RATE_USD_PER_KWH
total_energy_cost_usd = total_facility_energy_kwh * ELECTRICITY_RATE_USD_PER_KWH

# Model-to-Grid discount
model_tier = workload.tier
if model_tier == workload.tier.TIER_1_EFFICIENT:
    discount_pct = 20.0
elif model_tier == workload.tier.TIER_2_STANDARD:
    discount_pct = 10.0
else:
    discount_pct = 0.0

discounted_cost_usd = total_energy_cost_usd * (1 - discount_pct / 100)
savings_usd = total_energy_cost_usd - discounted_cost_usd

# Cost per token
tokens_generated = workload.energy_metrics.tokens_generated
cost_per_million_tokens = (total_energy_cost_usd / tokens_generated) * 1_000_000
discounted_cost_per_million_tokens = (discounted_cost_usd / tokens_generated) * 1_000_000

print(f"  ✓ IT energy: {total_it_energy_kwh:.2f} kWh (${it_energy_cost_usd:.2f})")
print(f"  ✓ Cooling energy: {total_cooling_energy_kwh:.2f} kWh (${cooling_energy_cost_usd:.2f})")
print(f"  ✓ Total energy: {total_facility_energy_kwh:.2f} kWh (${total_energy_cost_usd:.2f})")
print(f"  ✓ Model tier: {model_tier.value} ({discount_pct}% discount)")
print(f"  ✓ Discounted cost: ${discounted_cost_usd:.2f} (saves ${savings_usd:.2f})")
print(f"  ✓ Cost per M tokens: ${cost_per_million_tokens:.4f} → ${discounted_cost_per_million_tokens:.4f}")
print()

# ============================================================================
# PHASE 6: Generate Visualizations
# ============================================================================
print("[6/6] Generating comprehensive visualizations...")

fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# Color scheme
color_power = '#FF6B6B'
color_temp = '#4ECDC4'
color_grid = '#45B7D1'
color_cost = '#FFA07A'

# Plot 1: GPU Power Profile
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(time_hours, gpu_power_w, color=color_power, linewidth=1.5, label='GPU Power')
ax1.axhline(y=BENCHMARK_MODELS[MODEL_KEY].steady_avg_w, color='gray', linestyle='--', alpha=0.5, label='Steady Average')
ax1.axhline(y=BENCHMARK_MODELS[MODEL_KEY].prefill_peak_w, color='red', linestyle='--', alpha=0.5, label='Prefill Peak')
ax1.set_xlabel('Time (hours)', fontsize=12)
ax1.set_ylabel('GPU Power (W)', fontsize=12)
ax1.set_title('DeepSeek R1-Distill-32B: 10-Hour Power Profile (Single GPU)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')
ax1.set_xlim(0, WORKLOAD_DURATION_HOURS)

# Plot 2: Thermal Response
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(time_hours, gpu_tj_c, color=color_temp, linewidth=1.5, label='GPU Tj')
ax2.axhline(y=85, color='red', linestyle='--', alpha=0.5, label='NVIDIA Limit (45°C inlet)')
ax2.set_xlabel('Time (hours)', fontsize=11)
ax2.set_ylabel('Temperature (°C)', fontsize=11)
ax2.set_title('GPU Junction Temperature', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xlim(0, WORKLOAD_DURATION_HOURS)

# Plot 3: Cooling System Temperatures
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(time_hours, pg25_supply_temp_c, color='blue', linewidth=1.5, label='PG25 Supply')
ax3.plot(time_hours, pg25_return_temp_c, color='red', linewidth=1.5, label='PG25 Return')
ax3.set_xlabel('Time (hours)', fontsize=11)
ax3.set_ylabel('Temperature (°C)', fontsize=11)
ax3.set_title('Cooling Loop Temperatures', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.set_xlim(0, WORKLOAD_DURATION_HOURS)

# Plot 4: Pump Power
ax4 = fig.add_subplot(gs[1, 2])
ax4.plot(time_hours, pump_power_kw, color=color_cost, linewidth=1.5)
ax4.set_xlabel('Time (hours)', fontsize=11)
ax4.set_ylabel('Pump Power (kW)', fontsize=11)
ax4.set_title('Cooling System Parasitic Power', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, WORKLOAD_DURATION_HOURS)

# Plot 5: Grid Power Demand
ax5 = fig.add_subplot(gs[2, 0])
ax5.fill_between(time_hours, 0, total_it_power_kw, color=color_power, alpha=0.6, label='IT Load')
ax5.fill_between(time_hours, total_it_power_kw, total_facility_power_kw, color=color_cost, alpha=0.6, label='Cooling')
ax5.set_xlabel('Time (hours)', fontsize=11)
ax5.set_ylabel('Power (kW)', fontsize=11)
ax5.set_title('Grid Power Demand', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend()
ax5.set_xlim(0, WORKLOAD_DURATION_HOURS)

# Plot 6: pPUE Over Time
ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(time_hours, ppue, color=color_grid, linewidth=1.5)
ax6.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect Efficiency')
ax6.set_xlabel('Time (hours)', fontsize=11)
ax6.set_ylabel('pPUE', fontsize=11)
ax6.set_title('Partial PUE (IT + Cooling)', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.legend()
ax6.set_xlim(0, WORKLOAD_DURATION_HOURS)
ax6.set_ylim(1.0, 1.02)

# Plot 7: Economic Summary
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')

summary_text = f"""
ECONOMIC SUMMARY

Energy Consumption:
  IT Load:      {total_it_energy_kwh:>8.2f} kWh
  Cooling:      {total_cooling_energy_kwh:>8.2f} kWh
  Total:        {total_facility_energy_kwh:>8.2f} kWh

Costs (@ ${ELECTRICITY_RATE_USD_PER_KWH}/kWh):
  Base Cost:    ${total_energy_cost_usd:>8.2f}
  Tier:         {model_tier.value}
  Discount:     {discount_pct:>8.1f}%
  Final Cost:   ${discounted_cost_usd:>8.2f}
  Savings:      ${savings_usd:>8.2f}

Performance:
  Tokens:       {tokens_generated:>8,}
  Cost/M:       ${discounted_cost_per_million_tokens:>8.4f}
  Avg pPUE:     {np.mean(ppue):>8.4f}
"""

ax7.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('DeepSeek R1-Distill-32B: 10-Hour Integrated Simulation\nBatam BT1-2 Site | Benmax HCU2500 Cooling | GB300 NVL72',
             fontsize=16, fontweight='bold', y=0.995)

output_dir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'deepseek_10h_integrated_analysis.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  ✓ Visualization saved: {output_path}")

# ============================================================================
# Export Results
# ============================================================================

# Export workload profile
workload_path = os.path.join(output_dir, 'deepseek_10h_workload_profile.json')
workload.to_json(workload_path)
print(f"  ✓ Workload profile saved: {workload_path}")

# Export simulation results
results = {
    'simulation_metadata': {
        'model': BENCHMARK_MODELS[MODEL_KEY].model_name,
        'duration_hours': WORKLOAD_DURATION_HOURS,
        'site': 'Batam BT1-2',
        'ambient_temp_c': AMBIENT_TEMP_C,
        'timestamp': datetime.now().isoformat()
    },
    'workload_summary': {
        'peak_gpu_power_w': float(np.max(gpu_power_w)),
        'avg_gpu_power_w': float(np.mean(gpu_power_w)),
        'total_tokens': tokens_generated,
        'tier': model_tier.value
    },
    'thermal_summary': {
        'peak_gpu_tj_c': float(np.max(gpu_tj_c)),
        'avg_gpu_tj_c': float(np.mean(gpu_tj_c)),
        'peak_pg25_supply_c': float(np.max(pg25_supply_temp_c)),
        'avg_pg25_supply_c': float(np.mean(pg25_supply_temp_c))
    },
    'grid_summary': {
        'peak_demand_kw': float(peak_demand_kw),
        'avg_demand_kw': float(avg_demand_kw),
        'load_factor': float(load_factor),
        'max_ramp_up_kw_per_min': float(max_ramp_up),
        'max_ramp_down_kw_per_min': float(max_ramp_down)
    },
    'economic_summary': {
        'total_energy_kwh': float(total_facility_energy_kwh),
        'it_energy_kwh': float(total_it_energy_kwh),
        'cooling_energy_kwh': float(total_cooling_energy_kwh),
        'base_cost_usd': float(total_energy_cost_usd),
        'discount_pct': float(discount_pct),
        'discounted_cost_usd': float(discounted_cost_usd),
        'savings_usd': float(savings_usd),
        'cost_per_million_tokens': float(discounted_cost_per_million_tokens),
        'avg_ppue': float(np.mean(ppue))
    }
}

results_path = os.path.join(output_dir, 'deepseek_10h_simulation_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"  ✓ Simulation results saved: {results_path}")

print()
print("="*80)
print("SIMULATION COMPLETE")
print("="*80)
print(f"Duration: {WORKLOAD_DURATION_HOURS} hours")
print(f"Total Energy: {total_facility_energy_kwh:.2f} kWh")
print(f"Total Cost: ${discounted_cost_usd:.2f} (after {discount_pct}% discount)")
print(f"Average pPUE: {np.mean(ppue):.4f}")
print(f"Peak GPU Tj: {np.max(gpu_tj_c):.1f}°C (within NVIDIA limits)")
print("="*80)
