"""Electrical Infrastructure Validation Simulation

Validates UPS and transformer models against Batam BT1-2 site requirements
using realistic workload profiles from the workload module.

Scenario:
- 120 MW IT load (410 racks × GB300 NVL72)
- Eaton 9395XR 1935kVA UPS array (N+1 redundancy)
- 10 MVA medium voltage transformers (N+1 redundancy)
- DeepSeek R1-Distill-32B inference workload (10 hours)

Author: daniel.kearney@firmus.co
Date: February 2026
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from firmus_ai_factory.electrical import (
    UPSSystem,
    EATON_9395XR_1935KVA_SPECS,
    BatterySpecifications,
    BatteryTechnology,
    calculate_ups_array_capacity,
    TransformerModel,
    MV_TRANSFORMER_10MVA_SPECS,
    calculate_transformer_array_capacity,
)

from firmus_ai_factory.workload import generate_workload

# =============================================================================
# Simulation Parameters
# =============================================================================

# Site configuration
TOTAL_IT_LOAD_KW = 120000.0  # 120 MW
NUM_RACKS = 410
GPUS_PER_RACK = 72
GPU_POWER_W = 1400.0  # GB300 at TDP

# Workload configuration
MODEL_KEY = "deepseek-r1-distill-32b"
DURATION_HOURS = 10.0
TIME_STEP_S = 1.0

# Electrical configuration
POWER_FACTOR = 0.9
AMBIENT_TEMP_C = 35.0  # Batam ambient

# Output directory
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("ELECTRICAL INFRASTRUCTURE VALIDATION SIMULATION")
print("Batam BT1-2 Site | 120 MW | Eaton 9395XR UPS + MV Transformers")
print("="*80)

# =============================================================================
# Phase 1: Generate Workload Profile
# =============================================================================

print("\n[1/6] Generating workload profile...")

workload = generate_workload(
    model_key=MODEL_KEY,
    duration_hours=DURATION_HOURS,
    stochastic=True,
    gpu_platform="GB300",
    cooling_type="liquid",
)

# Extract power trace (time, power per GPU)
time_array = np.array([p[0] for p in workload.power_trace])
power_per_gpu_w = np.array([p[1] for p in workload.power_trace])

# Scale to site level (all racks, all GPUs)
site_power_kw = (power_per_gpu_w * NUM_RACKS * GPUS_PER_RACK) / 1000.0

print(f"  ✓ Generated {len(time_array)} time points")
print(f"  ✓ Peak site power: {np.max(site_power_kw)/1000:.1f} MW")
print(f"  ✓ Average site power: {np.mean(site_power_kw)/1000:.1f} MW")

# =============================================================================
# Phase 2: UPS Array Sizing
# =============================================================================

print("\n[2/6] Sizing UPS array...")

ups_sizing = calculate_ups_array_capacity(
    total_load_kw=TOTAL_IT_LOAD_KW,
    redundancy="N+1",
    ups_model=EATON_9395XR_1935KVA_SPECS,
)

print(f"  ✓ Number of UPS units: {ups_sizing['num_ups_units']}")
print(f"  ✓ Total UPS capacity: {ups_sizing['total_capacity_kw']/1000:.1f} MW")
print(f"  ✓ Utilization: {ups_sizing['utilization']*100:.1f}%")
print(f"  ✓ Redundant units: {ups_sizing['redundant_units']}")

# =============================================================================
# Phase 3: Transformer Array Sizing
# =============================================================================

print("\n[3/6] Sizing transformer array...")

total_load_kva = TOTAL_IT_LOAD_KW / POWER_FACTOR

xfmr_sizing = calculate_transformer_array_capacity(
    total_load_kva=total_load_kva,
    redundancy="N+1",
    transformer_model=MV_TRANSFORMER_10MVA_SPECS,
)

print(f"  ✓ Number of transformers: {xfmr_sizing['num_transformers']}")
print(f"  ✓ Total transformer capacity: {xfmr_sizing['total_capacity_kva']/1000:.1f} MVA")
print(f"  ✓ Utilization: {xfmr_sizing['utilization']*100:.1f}%")
print(f"  ✓ Total losses: {xfmr_sizing['total_losses_kw']/1000:.2f} MW")

# =============================================================================
# Phase 4: UPS Performance Simulation
# =============================================================================

print("\n[4/6] Simulating UPS performance...")

# Create single UPS instance
ups = UPSSystem(
    ups_specs=EATON_9395XR_1935KVA_SPECS,
    battery_specs=BatterySpecifications(
        technology=BatteryTechnology.VRLA,
        nominal_voltage_v=480.0,
        capacity_ah=1000.0,  # 10-minute runtime at full load
        num_cells=240,
    ),
    ambient_temp_c=AMBIENT_TEMP_C,
)

# Calculate per-UPS load
load_per_ups_kw = site_power_kw / ups_sizing['num_ups_units']

# Calculate efficiency and losses
ups_efficiency = np.array([ups.calculate_efficiency(load) for load in load_per_ups_kw])
ups_loss_per_unit_kw = np.array([ups.calculate_power_loss(load) for load in load_per_ups_kw])
ups_total_loss_kw = ups_loss_per_unit_kw * ups_sizing['num_ups_units']
ups_input_power_kw = site_power_kw + ups_total_loss_kw

# Grid stress metrics
grid_metrics = ups.estimate_grid_stress(load_per_ups_kw, time_step_s=TIME_STEP_S)

print(f"  ✓ Average UPS efficiency: {np.mean(ups_efficiency)*100:.2f}%")
print(f"  ✓ Peak UPS loss: {np.max(ups_total_loss_kw)/1000:.2f} MW")
print(f"  ✓ Peak grid power: {np.max(ups_input_power_kw)/1000:.1f} MW")
print(f"  ✓ Max ramp rate: {grid_metrics['max_ramp_rate_kw_per_s']:.1f} kW/s")
print(f"  ✓ Ramp violations: {grid_metrics['ramp_violations']}")

# =============================================================================
# Phase 5: Transformer Thermal Simulation
# =============================================================================

print("\n[5/6] Simulating transformer thermal performance...")

# Create single transformer instance
xfmr = TransformerModel(
    specs=MV_TRANSFORMER_10MVA_SPECS,
    ambient_temp_c=AMBIENT_TEMP_C,
)

# Calculate per-transformer load (include UPS losses)
load_per_xfmr_kva = (ups_input_power_kw / POWER_FACTOR) / xfmr_sizing['num_transformers']

# Calculate thermal performance
xfmr_top_oil_temp = np.array([xfmr.calculate_top_oil_temperature(load) for load in load_per_xfmr_kva])
xfmr_hotspot_temp = np.array([xfmr.calculate_hotspot_temperature(load, oil_temp) 
                               for load, oil_temp in zip(load_per_xfmr_kva, xfmr_top_oil_temp)])
xfmr_aging_factor = np.array([xfmr.calculate_aging_acceleration_factor(temp) for temp in xfmr_hotspot_temp])

# Calculate losses
xfmr_losses = np.array([xfmr.calculate_losses(load)[2] for load in load_per_xfmr_kva])
xfmr_total_loss_kw = xfmr_losses * xfmr_sizing['num_transformers']

print(f"  ✓ Peak top oil temp: {np.max(xfmr_top_oil_temp):.1f}°C")
print(f"  ✓ Peak hotspot temp: {np.max(xfmr_hotspot_temp):.1f}°C")
print(f"  ✓ Max aging factor: {np.max(xfmr_aging_factor):.2f}×")
print(f"  ✓ Peak transformer loss: {np.max(xfmr_total_loss_kw)/1000:.2f} MW")

# =============================================================================
# Phase 6: Generate Visualizations
# =============================================================================

print("\n[6/6] Generating visualizations...")

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# Plot 1: Site Power Profile
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(time_array / 3600, site_power_kw / 1000, 'b-', linewidth=1.5, label='IT Load')
ax1.plot(time_array / 3600, ups_input_power_kw / 1000, 'r--', linewidth=1.5, label='UPS Input (incl. losses)')
ax1.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Power (MW)', fontsize=12, fontweight='bold')
ax1.set_title('Site-Level Power Profile', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

# Plot 2: UPS Efficiency
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(time_array / 3600, ups_efficiency * 100, 'g-', linewidth=1.5)
ax2.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Efficiency (%)', fontsize=11, fontweight='bold')
ax2.set_title('UPS Efficiency', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([95, 98])

# Plot 3: UPS Power Loss
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(time_array / 3600, ups_total_loss_kw / 1000, 'orange', linewidth=1.5)
ax3.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Power Loss (MW)', fontsize=11, fontweight='bold')
ax3.set_title(f'UPS Heat Dissipation ({ups_sizing["num_ups_units"]} units)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Grid Ramp Rate
ax4 = fig.add_subplot(gs[1, 2])
ramp_rates_kw_per_s = np.diff(ups_input_power_kw) / TIME_STEP_S
ax4.plot(time_array[1:] / 3600, ramp_rates_kw_per_s / 1000, 'purple', linewidth=1.0, alpha=0.7)
ax4.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Ramp Rate (MW/s)', fontsize=11, fontweight='bold')
ax4.set_title('Grid Power Ramp Rate', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.axhline(0, color='black', linestyle='-', linewidth=0.5)

# Plot 5: Transformer Top Oil Temperature
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(time_array / 3600, xfmr_top_oil_temp, 'darkred', linewidth=1.5)
ax5.axhline(MV_TRANSFORMER_10MVA_SPECS.rated_top_oil_temp_c + AMBIENT_TEMP_C, 
           color='red', linestyle='--', linewidth=2, label='Rated limit')
ax5.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold')
ax5.set_title('Transformer Top Oil Temperature', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=10)

# Plot 6: Transformer Hotspot Temperature
ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(time_array / 3600, xfmr_hotspot_temp, 'darkred', linewidth=1.5)
ax6.axhline(MV_TRANSFORMER_10MVA_SPECS.max_hotspot_temp_c, 
           color='red', linestyle='--', linewidth=2, label='Max limit (110°C)')
ax6.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold')
ax6.set_title('Transformer Hotspot Temperature', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.legend(fontsize=10)

# Plot 7: Transformer Aging Factor
ax7 = fig.add_subplot(gs[2, 2])
ax7.plot(time_array / 3600, xfmr_aging_factor, 'brown', linewidth=1.5)
ax7.axhline(1.0, color='green', linestyle='--', linewidth=2, label='Normal aging (1.0×)')
ax7.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax7.set_ylabel('Aging Factor', fontsize=11, fontweight='bold')
ax7.set_title('Transformer Aging Acceleration', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)
ax7.legend(fontsize=10)

fig.suptitle('Electrical Infrastructure Performance - Batam BT1-2 Site\n120 MW | DeepSeek R1-Distill-32B Workload | 10 Hours', 
            fontsize=16, fontweight='bold', y=0.995)

plt.savefig(OUTPUT_DIR / "electrical_infrastructure_validation.png", dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {OUTPUT_DIR / 'electrical_infrastructure_validation.png'}")

# =============================================================================
# Export Summary JSON
# =============================================================================

summary = {
    "site_configuration": {
        "total_it_load_kw": TOTAL_IT_LOAD_KW,
        "num_racks": NUM_RACKS,
        "gpus_per_rack": GPUS_PER_RACK,
        "gpu_power_w": GPU_POWER_W,
        "power_factor": POWER_FACTOR,
        "ambient_temp_c": AMBIENT_TEMP_C,
    },
    "workload": {
        "model": MODEL_KEY,
        "duration_hours": DURATION_HOURS,
        "peak_power_kw": float(np.max(site_power_kw)),
        "avg_power_kw": float(np.mean(site_power_kw)),
        "energy_kwh": float(np.sum(site_power_kw) * TIME_STEP_S / 3600),
    },
    "ups_array": {
        "model": EATON_9395XR_1935KVA_SPECS.model,
        "num_units": ups_sizing['num_ups_units'],
        "total_capacity_kw": ups_sizing['total_capacity_kw'],
        "utilization": ups_sizing['utilization'],
        "redundancy": ups_sizing['redundancy'],
        "avg_efficiency": float(np.mean(ups_efficiency)),
        "peak_loss_kw": float(np.max(ups_total_loss_kw)),
        "total_energy_loss_kwh": float(np.sum(ups_total_loss_kw) * TIME_STEP_S / 3600),
    },
    "transformer_array": {
        "model": MV_TRANSFORMER_10MVA_SPECS.model,
        "num_units": xfmr_sizing['num_transformers'],
        "total_capacity_kva": xfmr_sizing['total_capacity_kva'],
        "utilization": xfmr_sizing['utilization'],
        "redundancy": xfmr_sizing['redundancy'],
        "peak_top_oil_temp_c": float(np.max(xfmr_top_oil_temp)),
        "peak_hotspot_temp_c": float(np.max(xfmr_hotspot_temp)),
        "max_aging_factor": float(np.max(xfmr_aging_factor)),
        "peak_loss_kw": float(np.max(xfmr_total_loss_kw)),
        "total_energy_loss_kwh": float(np.sum(xfmr_total_loss_kw) * TIME_STEP_S / 3600),
    },
    "grid_interaction": grid_metrics,
    "total_system": {
        "peak_grid_power_kw": float(np.max(ups_input_power_kw)),
        "avg_grid_power_kw": float(np.mean(ups_input_power_kw)),
        "total_grid_energy_kwh": float(np.sum(ups_input_power_kw) * TIME_STEP_S / 3600),
        "total_losses_kwh": float(
            np.sum(ups_total_loss_kw + xfmr_total_loss_kw) * TIME_STEP_S / 3600
        ),
        "overall_efficiency": float(
            np.sum(site_power_kw) / np.sum(ups_input_power_kw)
        ),
    },
}

with open(OUTPUT_DIR / "electrical_infrastructure_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"  ✓ Saved: {OUTPUT_DIR / 'electrical_infrastructure_summary.json'}")

# =============================================================================
# Print Summary
# =============================================================================

print("\n" + "="*80)
print("SIMULATION SUMMARY")
print("="*80)

print(f"\nSite Configuration:")
print(f"  Total IT load: {TOTAL_IT_LOAD_KW/1000:.1f} MW")
print(f"  Number of racks: {NUM_RACKS}")
print(f"  GPUs per rack: {GPUS_PER_RACK}")

print(f"\nUPS Array ({ups_sizing['num_ups_units']}× Eaton 9395XR-1935):")
print(f"  Total capacity: {ups_sizing['total_capacity_kw']/1000:.1f} MW")
print(f"  Average efficiency: {summary['ups_array']['avg_efficiency']*100:.2f}%")
print(f"  Peak loss: {summary['ups_array']['peak_loss_kw']/1000:.2f} MW")
print(f"  Energy loss: {summary['ups_array']['total_energy_loss_kwh']/1000:.2f} MWh")

print(f"\nTransformer Array ({xfmr_sizing['num_transformers']}× 10 MVA):")
print(f"  Total capacity: {xfmr_sizing['total_capacity_kva']/1000:.1f} MVA")
print(f"  Peak hotspot temp: {summary['transformer_array']['peak_hotspot_temp_c']:.1f}°C")
print(f"  Max aging factor: {summary['transformer_array']['max_aging_factor']:.2f}×")
print(f"  Peak loss: {summary['transformer_array']['peak_loss_kw']/1000:.2f} MW")
print(f"  Energy loss: {summary['transformer_array']['total_energy_loss_kwh']/1000:.2f} MWh")

print(f"\nTotal System:")
print(f"  Peak grid power: {summary['total_system']['peak_grid_power_kw']/1000:.1f} MW")
print(f"  Total grid energy: {summary['total_system']['total_grid_energy_kwh']/1000:.1f} MWh")
print(f"  Total losses: {summary['total_system']['total_losses_kwh']/1000:.2f} MWh")
print(f"  Overall efficiency: {summary['total_system']['overall_efficiency']*100:.2f}%")

print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)
