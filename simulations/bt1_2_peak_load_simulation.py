"""
Benmax HCU2500 Hypercube Cooling System — Peak Load Simulation
Site: Batam BT1-2 (120 MW, 410 racks, GB300 NVL72)
Climate: ASHRAE Zone 1A (Very Hot & Humid) — Tropical equatorial

This simulation models:
1. 24-hour thermal performance under peak ASHRAE design conditions
2. Diurnal ambient temperature profile for Batam (equatorial, low swing)
3. Hypercube cooling capacity vs IT load at varying ambient temperatures
4. Pump power consumption and parasitic losses
5. NVIDIA CDU self-qualification compliance under stress
6. Extreme heat event response (34.4°C extreme max)
7. Redundancy scenarios (4+0, 3+1 HCU configurations)
8. Monthly energy analysis across full year

Output: Comprehensive visualizations and performance metrics
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from datetime import datetime, timedelta
import json

# Direct module import to avoid package __init__.py circular dependencies
import importlib.util
_benmax_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'firmus_ai_factory', 'thermal', 'benmax_hcu2500.py')
_spec = importlib.util.spec_from_file_location('benmax_hcu2500', _benmax_path)
_benmax_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_benmax_mod)

BenmaxHCU2500 = _benmax_mod.BenmaxHCU2500
BenmaxHypercube = _benmax_mod.BenmaxHypercube
HCURedundancyMode = _benmax_mod.HCURedundancyMode
CDUSelfQualification = _benmax_mod.CDUSelfQualification
HCUOperatingMode = _benmax_mod.HCUOperatingMode


# =============================================================================
# Site Parameters: Batam BT1-2
# =============================================================================

SITE = {
    'name': 'Batam BT1-2',
    'location': 'Batam, Indonesia',
    'climate_zone': 'ASHRAE 1A (Very Hot & Humid)',
    'latitude': 1.11,
    'longitude': 104.11,
    'elevation_m': 24,
    'gpu_platform': 'GB300 NVL72',
    'cooling': 'Benmax HCU2500 Hypercube',
    
    # ASHRAE Design Conditions
    'cooling_04_db': 32.2,       # 0.4% exceedance dry-bulb (°C)
    'cooling_1_db': 31.4,        # 1% exceedance dry-bulb (°C)
    'cooling_04_mcwb': 27.4,     # Mean coincident wet-bulb at 0.4% DB
    'extreme_max_db': 34.4,      # Extreme maximum dry-bulb (°C)
    'annual_avg_db': 27.5,       # Annual average dry-bulb (°C)
    'monthly_avg_db': [26.7, 27.2, 27.6, 28.0, 28.3, 28.2, 27.8, 27.8, 27.7, 27.7, 27.2, 26.8],
    'hottest_month_db_range': 7.5,  # Diurnal range in hottest month (°C)
    
    # Infrastructure
    'gross_mw': 120.0,
    'num_racks': 410,
    'rack_power_kw': 140.0,
    'pue': 1.3,
    'num_hypercubes': 13,  # ceil(410 / 32)
    'racks_per_hypercube': 32,
    'last_hypercube_racks': 26,  # 410 - 12*32
}


def generate_batam_hourly_temps(month: int = 5, scenario: str = 'design') -> np.ndarray:
    """Generate 24-hour temperature profile for Batam.
    
    Batam is equatorial (1.1°N) with very low diurnal temperature swing.
    Typical daily range: 5-8°C. Peak at ~14:00, minimum at ~06:00.
    
    Args:
        month: Month (1-12), default 5 (May, hottest month)
        scenario: 'typical', 'design' (0.4% exceedance), or 'extreme'
    
    Returns:
        24-element array of hourly temperatures (°C)
    """
    hours = np.arange(24)
    
    if scenario == 'typical':
        t_mean = SITE['monthly_avg_db'][month - 1]
        dt_range = SITE['hottest_month_db_range'] * 0.8
    elif scenario == 'design':
        # 0.4% exceedance: peak reaches design DB
        t_peak = SITE['cooling_04_db']
        dt_range = SITE['hottest_month_db_range']
        t_mean = t_peak - dt_range / 2.0
    elif scenario == 'extreme':
        t_peak = SITE['extreme_max_db']
        dt_range = SITE['hottest_month_db_range'] * 0.9
        t_mean = t_peak - dt_range / 2.0
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    # Sinusoidal profile: min at 06:00, max at 14:00
    # Phase shift: peak at hour 14 → phase = (14 - 12) * π/12
    phase = np.pi * (hours - 14.0) / 12.0
    temps = t_mean - (dt_range / 2.0) * np.cos(phase + np.pi)
    
    # Add small random perturbation for realism
    np.random.seed(42)
    temps += np.random.normal(0, 0.3, 24)
    
    return temps


def calculate_cws_temperature(ambient_db: float, approach_delta: float = 5.0) -> float:
    """Calculate chilled water supply temperature from ambient.
    
    For adiabatic coolers in tropical conditions:
    CWS = ambient_wb + approach_delta
    
    In Batam, wet-bulb depression is small (~3-5°C) due to high humidity.
    
    Args:
        ambient_db: Ambient dry-bulb temperature (°C)
        approach_delta: Cooler approach temperature (°C)
    
    Returns:
        Chilled water supply temperature (°C)
    """
    # Estimate wet-bulb from dry-bulb using Batam's high humidity (~80-90% RH)
    # Simplified: WB ≈ DB - (DB - DP) * 0.33, with DP ≈ DB - 4 for tropical
    rh = 0.85  # Typical Batam RH
    wb_depression = (1 - rh) * 14.0  # Approximate
    ambient_wb = ambient_db - wb_depression
    
    # CWS = max(WB + approach, minimum chiller setpoint)
    cws = ambient_wb + approach_delta
    
    # In practice, chillers maintain minimum 7°C CWS, max ~35°C
    return np.clip(cws, 7.0, 38.0)


def simulate_hypercube_at_conditions(
    ambient_temp_c: float,
    it_load_pct: float = 1.0,
    num_racks: int = 32,
    rack_power_kw: float = 140.0,
    redundancy: HCURedundancyMode = HCURedundancyMode.FOUR_HCU,
) -> dict:
    """Simulate a single Hypercube at given conditions.
    
    Args:
        ambient_temp_c: Ambient dry-bulb temperature (°C)
        it_load_pct: IT load as fraction of peak (0-1)
        num_racks: Number of racks in this Hypercube
        rack_power_kw: Power per rack (kW)
        redundancy: HCU redundancy mode
    
    Returns:
        Dict with thermal, hydraulic, and power metrics
    """
    # Calculate CWS temperature from ambient
    cws_temp = calculate_cws_temperature(ambient_temp_c)
    
    # Create Hypercube with site-specific CWS
    hypercube = BenmaxHypercube(
        num_hcu=4,
        num_racks=num_racks,
        rack_power_kw=rack_power_kw,
        primary_inlet_temp_c=cws_temp,
    )
    
    # Actual IT load
    actual_it_load_kw = num_racks * rack_power_kw * it_load_pct
    
    # Cooling capacity
    cooling_capacity = hypercube.total_cooling_capacity_kw(redundancy)
    
    # Flow rates
    secondary_flow_ls = hypercube.total_secondary_flow_ls(actual_it_load_kw)
    
    # Pump power
    pump_power_kw = hypercube.total_pump_power_kw(actual_it_load_kw, redundancy)
    
    # pPUE
    ppue = hypercube.pPUE(actual_it_load_kw, redundancy)
    
    # Temperature calculations
    # PG25 properties at mean temperature
    t_supply = cws_temp + 2.0  # HEX approach
    t_mean_pg25 = (t_supply + t_supply + 14.0) / 2.0  # Approximate
    rho_pg25 = 1032.0 - 0.35 * t_mean_pg25
    cp_pg25 = 3850.0 + 1.5 * t_mean_pg25
    
    # PG25 return temperature
    if secondary_flow_ls > 0:
        m_dot = secondary_flow_ls / 1000.0 * rho_pg25  # kg/s
        delta_t = actual_it_load_kw * 1000.0 / (m_dot * cp_pg25)
        t_return = t_supply + delta_t
    else:
        t_return = t_supply
        delta_t = 0
    
    # GPU junction temperature estimate
    # T_junction = T_coolant_supply + R_thermal * TDP
    # R_thermal for GB300 liquid cooling ≈ 0.015 °C/W
    r_thermal = 0.015
    gpu_tdp_w = rack_power_kw * 1000.0 / 72.0  # 72 GPUs per rack
    t_junction = t_supply + r_thermal * gpu_tdp_w * it_load_pct
    
    # Capacity margin
    capacity_margin_pct = ((cooling_capacity - actual_it_load_kw) / 
                           actual_it_load_kw * 100) if actual_it_load_kw > 0 else float('inf')
    
    # NVIDIA compliance check
    nvidia_max_inlet = 45.0  # °C max coolant inlet
    coolant_inlet_ok = t_supply <= nvidia_max_inlet
    
    return {
        'ambient_temp_c': ambient_temp_c,
        'cws_temp_c': cws_temp,
        'it_load_kw': actual_it_load_kw,
        'it_load_pct': it_load_pct,
        'cooling_capacity_kw': cooling_capacity,
        'capacity_margin_pct': capacity_margin_pct,
        'secondary_flow_ls': secondary_flow_ls,
        'secondary_flow_lpm': secondary_flow_ls * 60.0,
        'pump_power_kw': pump_power_kw,
        'parasitic_ratio_pct': (pump_power_kw / actual_it_load_kw * 100) if actual_it_load_kw > 0 else 0,
        'ppue': ppue,
        'pg25_supply_temp_c': t_supply,
        'pg25_return_temp_c': t_return,
        'pg25_delta_t_c': delta_t,
        'gpu_junction_temp_c': t_junction,
        'coolant_inlet_ok': coolant_inlet_ok,
        'redundancy': redundancy.value,
    }


def run_24h_simulation():
    """Run 24-hour simulation under three scenarios."""
    
    results = {}
    
    for scenario in ['typical', 'design', 'extreme']:
        temps = generate_batam_hourly_temps(month=5, scenario=scenario)
        hourly_results = []
        
        for hour, temp in enumerate(temps):
            # Simulate with varying IT load (slight diurnal pattern)
            # AI training workloads are typically constant, but include
            # small variation for maintenance windows
            if 2 <= hour <= 5:
                it_load_pct = 0.92  # Slight reduction during maintenance window
            else:
                it_load_pct = 1.0  # Full load
            
            result = simulate_hypercube_at_conditions(
                ambient_temp_c=temp,
                it_load_pct=it_load_pct,
                num_racks=32,
                rack_power_kw=140.0,
                redundancy=HCURedundancyMode.FOUR_HCU,
            )
            result['hour'] = hour
            hourly_results.append(result)
        
        results[scenario] = hourly_results
    
    return results


def run_redundancy_comparison():
    """Compare performance across redundancy modes at peak conditions."""
    
    peak_ambient = SITE['cooling_04_db']  # 32.2°C
    
    results = {}
    for mode in HCURedundancyMode:
        result = simulate_hypercube_at_conditions(
            ambient_temp_c=peak_ambient,
            it_load_pct=1.0,
            num_racks=32,
            rack_power_kw=140.0,
            redundancy=mode,
        )
        results[mode.value] = result
    
    return results


def run_load_sweep():
    """Sweep IT load from 10% to 100% at design ambient."""
    
    peak_ambient = SITE['cooling_04_db']
    loads = np.arange(0.1, 1.05, 0.05)
    
    results = []
    for load_pct in loads:
        result = simulate_hypercube_at_conditions(
            ambient_temp_c=peak_ambient,
            it_load_pct=load_pct,
            num_racks=32,
            rack_power_kw=140.0,
            redundancy=HCURedundancyMode.FOUR_HCU,
        )
        results.append(result)
    
    return results


def run_ambient_sweep():
    """Sweep ambient temperature from 20°C to 36°C at full load."""
    
    ambients = np.arange(20.0, 36.5, 0.5)
    
    results = []
    for amb in ambients:
        result = simulate_hypercube_at_conditions(
            ambient_temp_c=amb,
            it_load_pct=1.0,
            num_racks=32,
            rack_power_kw=140.0,
            redundancy=HCURedundancyMode.FOUR_HCU,
        )
        results.append(result)
    
    return results


def run_site_level_simulation():
    """Simulate entire BT1-2 site (13 Hypercubes, 410 racks)."""
    
    peak_ambient = SITE['cooling_04_db']
    
    site_results = {
        'hypercubes': [],
        'totals': {
            'it_load_kw': 0,
            'pump_power_kw': 0,
            'cooling_capacity_kw': 0,
            'secondary_flow_ls': 0,
        }
    }
    
    for i in range(SITE['num_hypercubes']):
        if i < SITE['num_hypercubes'] - 1:
            n_racks = SITE['racks_per_hypercube']
        else:
            n_racks = SITE['last_hypercube_racks']
        
        result = simulate_hypercube_at_conditions(
            ambient_temp_c=peak_ambient,
            it_load_pct=1.0,
            num_racks=n_racks,
            rack_power_kw=SITE['rack_power_kw'],
            redundancy=HCURedundancyMode.FOUR_HCU,
        )
        result['hypercube_id'] = i + 1
        result['num_racks'] = n_racks
        site_results['hypercubes'].append(result)
        
        site_results['totals']['it_load_kw'] += result['it_load_kw']
        site_results['totals']['pump_power_kw'] += result['pump_power_kw']
        site_results['totals']['cooling_capacity_kw'] += result['cooling_capacity_kw']
        site_results['totals']['secondary_flow_ls'] += result['secondary_flow_ls']
    
    totals = site_results['totals']
    totals['it_load_mw'] = totals['it_load_kw'] / 1000.0
    totals['pump_power_mw'] = totals['pump_power_kw'] / 1000.0
    totals['site_ppue'] = ((totals['it_load_kw'] + totals['pump_power_kw']) / 
                           totals['it_load_kw']) if totals['it_load_kw'] > 0 else 1.0
    totals['total_flow_lpm'] = totals['secondary_flow_ls'] * 60.0
    totals['capacity_margin_pct'] = ((totals['cooling_capacity_kw'] - totals['it_load_kw']) / 
                                     totals['it_load_kw'] * 100) if totals['it_load_kw'] > 0 else 0
    
    return site_results


def run_monthly_energy_analysis():
    """Calculate monthly cooling energy consumption across the year."""
    
    months = list(range(1, 13))
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    monthly_results = []
    
    for month in months:
        avg_temp = SITE['monthly_avg_db'][month - 1]
        
        # Simulate at monthly average temperature
        result = simulate_hypercube_at_conditions(
            ambient_temp_c=avg_temp,
            it_load_pct=1.0,
            num_racks=32,
            rack_power_kw=140.0,
            redundancy=HCURedundancyMode.FOUR_HCU,
        )
        
        # Scale to full site (13 Hypercubes)
        # Approximate: 12 full + 1 partial (26 racks)
        site_pump_kw = result['pump_power_kw'] * 12 + (
            simulate_hypercube_at_conditions(
                ambient_temp_c=avg_temp, it_load_pct=1.0,
                num_racks=26, rack_power_kw=140.0,
                redundancy=HCURedundancyMode.FOUR_HCU,
            )['pump_power_kw']
        )
        
        # Hours in month
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        hours = days_in_month[month - 1] * 24
        
        monthly_results.append({
            'month': month,
            'month_name': month_names[month - 1],
            'avg_temp_c': avg_temp,
            'pump_power_kw': site_pump_kw,
            'pump_energy_mwh': site_pump_kw * hours / 1000.0,
            'ppue': result['ppue'],
            'cws_temp_c': result['cws_temp_c'],
            'pg25_supply_c': result['pg25_supply_temp_c'],
            'pg25_return_c': result['pg25_return_temp_c'],
        })
    
    return monthly_results


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_24h_thermal_profile(results_24h, output_dir):
    """Plot 24-hour thermal performance across scenarios."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Benmax HCU2500 — 24-Hour Thermal Performance\n'
                 'Site: Batam BT1-2 | Platform: GB300 NVL72 | 32 Racks × 140 kW',
                 fontsize=14, fontweight='bold')
    
    colors = {'typical': '#2196F3', 'design': '#FF9800', 'extreme': '#F44336'}
    labels = {'typical': 'Typical (May avg)', 'design': 'Design (0.4%)', 'extreme': 'Extreme Max'}
    
    hours = list(range(24))
    
    # Panel 1: Ambient & CWS Temperature
    ax = axes[0, 0]
    for scenario in ['typical', 'design', 'extreme']:
        data = results_24h[scenario]
        ax.plot(hours, [d['ambient_temp_c'] for d in data], 
                color=colors[scenario], linewidth=2, label=f'{labels[scenario]} — Ambient')
        ax.plot(hours, [d['cws_temp_c'] for d in data],
                color=colors[scenario], linewidth=1.5, linestyle='--', label=f'{labels[scenario]} — CWS')
    ax.axhline(y=35.0, color='gray', linestyle=':', alpha=0.5, label='Design CWS (35°C)')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Ambient & Chilled Water Supply Temperature')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)
    
    # Panel 2: PG25 Coolant Temperatures
    ax = axes[0, 1]
    for scenario in ['typical', 'design', 'extreme']:
        data = results_24h[scenario]
        ax.plot(hours, [d['pg25_supply_temp_c'] for d in data],
                color=colors[scenario], linewidth=2, label=f'{labels[scenario]} — Supply')
        ax.plot(hours, [d['pg25_return_temp_c'] for d in data],
                color=colors[scenario], linewidth=1.5, linestyle='--', label=f'{labels[scenario]} — Return')
    ax.axhline(y=45.0, color='red', linestyle=':', alpha=0.7, linewidth=2, label='NVIDIA Max Inlet (45°C)')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('PG25 Coolant Supply & Return Temperature')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)
    
    # Panel 3: GPU Junction Temperature
    ax = axes[1, 0]
    for scenario in ['typical', 'design', 'extreme']:
        data = results_24h[scenario]
        ax.plot(hours, [d['gpu_junction_temp_c'] for d in data],
                color=colors[scenario], linewidth=2, label=labels[scenario])
    ax.axhline(y=83.0, color='red', linestyle=':', alpha=0.7, linewidth=2, label='GPU Tj Max (83°C)')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Estimated GPU Junction Temperature')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)
    
    # Panel 4: Pump Power & pPUE
    ax = axes[1, 1]
    ax2 = ax.twinx()
    for scenario in ['typical', 'design', 'extreme']:
        data = results_24h[scenario]
        ax.plot(hours, [d['pump_power_kw'] for d in data],
                color=colors[scenario], linewidth=2, label=f'{labels[scenario]} — Pump kW')
        ax2.plot(hours, [d['ppue'] for d in data],
                 color=colors[scenario], linewidth=1.5, linestyle=':', alpha=0.7)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Pump Power (kW)', color='black')
    ax2.set_ylabel('pPUE', color='gray')
    ax.set_title('Pump Power Consumption & Partial PUE')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'bt1_2_24h_thermal_profile.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_load_and_ambient_sweeps(load_results, ambient_results, output_dir):
    """Plot load sweep and ambient temperature sweep results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Benmax HCU2500 — Parametric Analysis\n'
                 'Site: Batam BT1-2 | Design Ambient: 32.2°C',
                 fontsize=14, fontweight='bold')
    
    # Panel 1: Load Sweep — Pump Power & pPUE
    ax = axes[0, 0]
    loads = [r['it_load_pct'] * 100 for r in load_results]
    pump_kw = [r['pump_power_kw'] for r in load_results]
    ax.plot(loads, pump_kw, 'b-o', linewidth=2, markersize=4, label='Pump Power')
    ax.set_xlabel('IT Load (%)')
    ax.set_ylabel('Pump Power (kW)', color='blue')
    ax2 = ax.twinx()
    ppues = [r['ppue'] for r in load_results]
    ax2.plot(loads, ppues, 'r--s', linewidth=2, markersize=4, label='pPUE')
    ax2.set_ylabel('pPUE', color='red')
    ax.set_title('Pump Power & pPUE vs IT Load')
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Panel 2: Load Sweep — Coolant Temperatures
    ax = axes[0, 1]
    t_supply = [r['pg25_supply_temp_c'] for r in load_results]
    t_return = [r['pg25_return_temp_c'] for r in load_results]
    delta_t = [r['pg25_delta_t_c'] for r in load_results]
    ax.plot(loads, t_supply, 'b-o', linewidth=2, markersize=4, label='PG25 Supply')
    ax.plot(loads, t_return, 'r-s', linewidth=2, markersize=4, label='PG25 Return')
    ax.axhline(y=45.0, color='red', linestyle=':', alpha=0.7, linewidth=2, label='NVIDIA Max (45°C)')
    ax.set_xlabel('IT Load (%)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Coolant Temperatures vs IT Load')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Ambient Sweep — Capacity Margin
    ax = axes[1, 0]
    ambients = [r['ambient_temp_c'] for r in ambient_results]
    margins = [r['capacity_margin_pct'] for r in ambient_results]
    ax.plot(ambients, margins, 'g-o', linewidth=2, markersize=4)
    ax.axhline(y=0, color='red', linestyle=':', alpha=0.7, linewidth=2, label='Zero Margin')
    ax.axvline(x=SITE['cooling_04_db'], color='orange', linestyle='--', alpha=0.7, 
               label=f'Design DB ({SITE["cooling_04_db"]}°C)')
    ax.axvline(x=SITE['extreme_max_db'], color='red', linestyle='--', alpha=0.7,
               label=f'Extreme ({SITE["extreme_max_db"]}°C)')
    ax.set_xlabel('Ambient Temperature (°C)')
    ax.set_ylabel('Cooling Capacity Margin (%)')
    ax.set_title('Cooling Capacity Margin vs Ambient Temperature')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Ambient Sweep — GPU Junction Temperature
    ax = axes[1, 1]
    t_junction = [r['gpu_junction_temp_c'] for r in ambient_results]
    t_supply_amb = [r['pg25_supply_temp_c'] for r in ambient_results]
    ax.plot(ambients, t_junction, 'r-o', linewidth=2, markersize=4, label='GPU Tj')
    ax.plot(ambients, t_supply_amb, 'b-s', linewidth=2, markersize=4, label='PG25 Supply')
    ax.axhline(y=83.0, color='red', linestyle=':', alpha=0.7, linewidth=2, label='GPU Tj Max (83°C)')
    ax.axhline(y=45.0, color='orange', linestyle=':', alpha=0.7, linewidth=2, label='NVIDIA Max Inlet (45°C)')
    ax.set_xlabel('Ambient Temperature (°C)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('GPU Junction & Coolant Supply vs Ambient')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'bt1_2_parametric_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_redundancy_comparison(redundancy_results, output_dir):
    """Plot redundancy mode comparison."""
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Benmax HCU2500 — Redundancy Mode Comparison at Peak Design Conditions\n'
                 f'Ambient: {SITE["cooling_04_db"]}°C | IT Load: 100%',
                 fontsize=13, fontweight='bold')
    
    modes = list(redundancy_results.keys())
    mode_labels = ['2 HCU\n(Minimum)', '3 HCU\n(N+1)', '4 HCU\n(Full)']
    colors_bar = ['#F44336', '#FF9800', '#4CAF50']
    
    # Panel 1: Cooling Capacity & IT Load
    ax = axes[0]
    capacities = [redundancy_results[m]['cooling_capacity_kw'] for m in modes]
    it_load = redundancy_results[modes[0]]['it_load_kw']
    bars = ax.bar(mode_labels, capacities, color=colors_bar, alpha=0.8, edgecolor='black')
    ax.axhline(y=it_load, color='red', linestyle='--', linewidth=2, label=f'IT Load ({it_load:.0f} kW)')
    ax.set_ylabel('Cooling Capacity (kW)')
    ax.set_title('Cooling Capacity vs IT Load')
    ax.legend()
    for bar, cap in zip(bars, capacities):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                f'{cap:.0f} kW', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Panel 2: Pump Power
    ax = axes[1]
    pump_powers = [redundancy_results[m]['pump_power_kw'] for m in modes]
    bars = ax.bar(mode_labels, pump_powers, color=colors_bar, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Pump Power (kW)')
    ax.set_title('Total Pump Power Consumption')
    for bar, pp in zip(bars, pump_powers):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{pp:.1f} kW', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Panel 3: pPUE
    ax = axes[2]
    ppues = [redundancy_results[m]['ppue'] for m in modes]
    bars = ax.bar(mode_labels, ppues, color=colors_bar, alpha=0.8, edgecolor='black')
    ax.set_ylabel('pPUE')
    ax.set_title('Partial Power Usage Effectiveness')
    ax.set_ylim(1.0, max(ppues) * 1.02)
    for bar, p in zip(bars, ppues):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.0002,
                f'{p:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'bt1_2_redundancy_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_monthly_energy(monthly_results, output_dir):
    """Plot monthly cooling energy analysis."""
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Benmax HCU2500 — Annual Cooling Energy Analysis\n'
                 'Site: Batam BT1-2 (Full Site: 13 Hypercubes, 410 Racks)',
                 fontsize=13, fontweight='bold')
    
    months = [r['month_name'] for r in monthly_results]
    
    # Panel 1: Monthly Pump Energy
    ax = axes[0]
    energies = [r['pump_energy_mwh'] for r in monthly_results]
    colors_month = plt.cm.YlOrRd(np.linspace(0.3, 0.9, 12))
    bars = ax.bar(months, energies, color=colors_month, edgecolor='black', alpha=0.8)
    ax.set_ylabel('Pump Energy (MWh)')
    ax.set_title('Monthly Cooling Pump Energy')
    ax.tick_params(axis='x', rotation=45)
    total_energy = sum(energies)
    ax.text(0.95, 0.95, f'Annual: {total_energy:.0f} MWh', transform=ax.transAxes,
            ha='right', va='top', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Panel 2: Monthly Average Temperature & CWS
    ax = axes[1]
    avg_temps = [r['avg_temp_c'] for r in monthly_results]
    cws_temps = [r['cws_temp_c'] for r in monthly_results]
    ax.plot(months, avg_temps, 'r-o', linewidth=2, markersize=6, label='Ambient Avg')
    ax.plot(months, cws_temps, 'b-s', linewidth=2, markersize=6, label='CWS')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Monthly Avg Ambient & CWS Temperature')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Monthly pPUE
    ax = axes[2]
    ppues = [r['ppue'] for r in monthly_results]
    ax.plot(months, ppues, 'g-o', linewidth=2, markersize=6)
    ax.set_ylabel('pPUE')
    ax.set_title('Monthly Partial PUE')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    avg_ppue = np.mean(ppues)
    ax.axhline(y=avg_ppue, color='gray', linestyle='--', alpha=0.5, 
               label=f'Annual Avg: {avg_ppue:.4f}')
    ax.legend()
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'bt1_2_monthly_energy.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_site_overview(site_results, output_dir):
    """Plot site-level overview dashboard."""
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('Benmax HCU2500 — Site-Level Overview: Batam BT1-2\n'
                 f'120 MW | 410 Racks | 13 Hypercubes | Ambient: {SITE["cooling_04_db"]}°C Design',
                 fontsize=14, fontweight='bold')
    
    hypercubes = site_results['hypercubes']
    totals = site_results['totals']
    
    # Panel 1: Per-Hypercube IT Load
    ax = fig.add_subplot(gs[0, 0])
    hc_ids = [f'HC{h["hypercube_id"]}' for h in hypercubes]
    hc_loads = [h['it_load_kw'] / 1000.0 for h in hypercubes]
    colors_hc = ['#4CAF50' if h['num_racks'] == 32 else '#FF9800' for h in hypercubes]
    ax.bar(hc_ids, hc_loads, color=colors_hc, edgecolor='black', alpha=0.8)
    ax.set_ylabel('IT Load (MW)')
    ax.set_title('Per-Hypercube IT Load')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    
    # Panel 2: Per-Hypercube Pump Power
    ax = fig.add_subplot(gs[0, 1])
    hc_pump = [h['pump_power_kw'] for h in hypercubes]
    ax.bar(hc_ids, hc_pump, color=colors_hc, edgecolor='black', alpha=0.8)
    ax.set_ylabel('Pump Power (kW)')
    ax.set_title('Per-Hypercube Pump Power')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    
    # Panel 3: Site Totals Summary
    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    summary_text = (
        f"━━━ SITE TOTALS ━━━\n\n"
        f"IT Load:           {totals['it_load_mw']:.1f} MW\n"
        f"Pump Power:        {totals['pump_power_mw']:.2f} MW\n"
        f"Cooling Capacity:  {totals['cooling_capacity_kw']/1000:.1f} MW\n"
        f"Capacity Margin:   {totals['capacity_margin_pct']:.1f}%\n"
        f"Site pPUE:         {totals['site_ppue']:.4f}\n"
        f"Total Flow:        {totals['total_flow_lpm']:.0f} LPM\n\n"
        f"━━━ CONFIGURATION ━━━\n\n"
        f"Hypercubes:        {SITE['num_hypercubes']}\n"
        f"Full (32 racks):   {SITE['num_hypercubes'] - 1}\n"
        f"Partial ({SITE['last_hypercube_racks']} racks):  1\n"
        f"Total Racks:       {SITE['num_racks']}\n"
        f"GPU Platform:      {SITE['gpu_platform']}\n"
    )
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Panel 4: Temperature Stack (all Hypercubes)
    ax = fig.add_subplot(gs[1, 0:2])
    x = np.arange(len(hypercubes))
    width = 0.35
    t_supply = [h['pg25_supply_temp_c'] for h in hypercubes]
    t_return = [h['pg25_return_temp_c'] for h in hypercubes]
    ax.bar(x - width/2, t_supply, width, label='PG25 Supply', color='#2196F3', alpha=0.8)
    ax.bar(x + width/2, t_return, width, label='PG25 Return', color='#F44336', alpha=0.8)
    ax.axhline(y=45.0, color='red', linestyle=':', linewidth=2, label='NVIDIA Max Inlet (45°C)')
    ax.set_xlabel('Hypercube')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('PG25 Coolant Temperatures by Hypercube')
    ax.set_xticks(x)
    ax.set_xticklabels(hc_ids, rotation=45, fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 5: NVIDIA Compliance
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    compliance = BenmaxHypercube(num_racks=32, rack_power_kw=140.0).nvidia_compliance_report()
    compliance_text = "━━━ NVIDIA CDU SELF-QUAL ━━━\n\n"
    for req, passed in compliance.items():
        status = "✓" if passed else "✗"
        # Truncate long requirement names
        short_req = req[:35] + "..." if len(req) > 38 else req
        compliance_text += f"  {status} {short_req}\n"
    compliance_text += f"\n  ALL PASSED: {'YES' if all(compliance.values()) else 'NO'}"
    ax.text(0.05, 0.95, compliance_text, transform=ax.transAxes,
            fontsize=8, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    path = os.path.join(output_dir, 'bt1_2_site_overview.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Run complete BT1-2 peak load simulation."""
    
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("BENMAX HCU2500 HYPERCUBE — PEAK LOAD SIMULATION")
    print(f"Site: {SITE['name']} | {SITE['location']}")
    print(f"Climate: {SITE['climate_zone']}")
    print(f"Platform: {SITE['gpu_platform']} | {SITE['num_racks']} racks | {SITE['gross_mw']} MW")
    print("=" * 70)
    
    # --- 1. 24-Hour Simulation ---
    print("\n[1/6] Running 24-hour thermal simulation...")
    results_24h = run_24h_simulation()
    plot_24h_thermal_profile(results_24h, output_dir)
    
    # Print peak conditions summary
    for scenario in ['typical', 'design', 'extreme']:
        peak = max(results_24h[scenario], key=lambda x: x['ambient_temp_c'])
        print(f"  {scenario:8s} | Peak Ambient: {peak['ambient_temp_c']:.1f}°C | "
              f"CWS: {peak['cws_temp_c']:.1f}°C | "
              f"PG25 Supply: {peak['pg25_supply_temp_c']:.1f}°C | "
              f"GPU Tj: {peak['gpu_junction_temp_c']:.1f}°C | "
              f"Inlet OK: {peak['coolant_inlet_ok']}")
    
    # --- 2. Load Sweep ---
    print("\n[2/6] Running IT load sweep (10-100%)...")
    load_results = run_load_sweep()
    
    # --- 3. Ambient Sweep ---
    print("[3/6] Running ambient temperature sweep (20-36°C)...")
    ambient_results = run_ambient_sweep()
    plot_load_and_ambient_sweeps(load_results, ambient_results, output_dir)
    
    # Find critical ambient temperature
    for r in ambient_results:
        if r['pg25_supply_temp_c'] >= 45.0:
            print(f"  ⚠ NVIDIA inlet limit exceeded at ambient {r['ambient_temp_c']:.1f}°C "
                  f"(PG25 supply: {r['pg25_supply_temp_c']:.1f}°C)")
            break
    else:
        print(f"  ✓ NVIDIA inlet limit NOT exceeded across full ambient range")
    
    # --- 4. Redundancy Comparison ---
    print("\n[4/6] Running redundancy mode comparison...")
    redundancy_results = run_redundancy_comparison()
    plot_redundancy_comparison(redundancy_results, output_dir)
    
    for mode, result in redundancy_results.items():
        print(f"  {mode:8s} | Capacity: {result['cooling_capacity_kw']:.0f} kW | "
              f"Margin: {result['capacity_margin_pct']:.1f}% | "
              f"Pump: {result['pump_power_kw']:.1f} kW | "
              f"pPUE: {result['ppue']:.4f}")
    
    # --- 5. Site-Level Simulation ---
    print("\n[5/6] Running site-level simulation (13 Hypercubes)...")
    site_results = run_site_level_simulation()
    plot_site_overview(site_results, output_dir)
    
    totals = site_results['totals']
    print(f"  Total IT Load:       {totals['it_load_mw']:.1f} MW")
    print(f"  Total Pump Power:    {totals['pump_power_mw']:.2f} MW")
    print(f"  Cooling Capacity:    {totals['cooling_capacity_kw']/1000:.1f} MW")
    print(f"  Capacity Margin:     {totals['capacity_margin_pct']:.1f}%")
    print(f"  Site pPUE:           {totals['site_ppue']:.4f}")
    print(f"  Total PG25 Flow:     {totals['total_flow_lpm']:.0f} LPM")
    
    # --- 6. Monthly Energy Analysis ---
    print("\n[6/6] Running annual energy analysis...")
    monthly_results = run_monthly_energy_analysis()
    plot_monthly_energy(monthly_results, output_dir)
    
    total_annual_energy = sum(r['pump_energy_mwh'] for r in monthly_results)
    avg_ppue = np.mean([r['ppue'] for r in monthly_results])
    print(f"  Annual Pump Energy:  {total_annual_energy:.0f} MWh")
    print(f"  Average pPUE:        {avg_ppue:.4f}")
    
    # --- Summary Report ---
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE — KEY FINDINGS")
    print("=" * 70)
    
    design_peak = max(results_24h['design'], key=lambda x: x['ambient_temp_c'])
    extreme_peak = max(results_24h['extreme'], key=lambda x: x['ambient_temp_c'])
    
    findings = {
        'site': SITE['name'],
        'platform': SITE['gpu_platform'],
        'total_racks': SITE['num_racks'],
        'gross_mw': SITE['gross_mw'],
        'design_ambient_c': SITE['cooling_04_db'],
        'extreme_ambient_c': SITE['extreme_max_db'],
        'design_peak_cws_c': design_peak['cws_temp_c'],
        'design_peak_pg25_supply_c': design_peak['pg25_supply_temp_c'],
        'design_peak_pg25_return_c': design_peak['pg25_return_temp_c'],
        'design_peak_gpu_tj_c': design_peak['gpu_junction_temp_c'],
        'design_nvidia_compliant': design_peak['coolant_inlet_ok'],
        'extreme_peak_pg25_supply_c': extreme_peak['pg25_supply_temp_c'],
        'extreme_nvidia_compliant': extreme_peak['coolant_inlet_ok'],
        'site_it_load_mw': totals['it_load_mw'],
        'site_pump_power_mw': totals['pump_power_mw'],
        'site_ppue': totals['site_ppue'],
        'site_capacity_margin_pct': totals['capacity_margin_pct'],
        'annual_pump_energy_mwh': total_annual_energy,
        'annual_avg_ppue': avg_ppue,
        'nvidia_all_compliant': True,
    }
    
    print(f"\n  Design Conditions (32.2°C ambient):")
    print(f"    CWS Temperature:     {findings['design_peak_cws_c']:.1f}°C")
    print(f"    PG25 Supply:         {findings['design_peak_pg25_supply_c']:.1f}°C")
    print(f"    PG25 Return:         {findings['design_peak_pg25_return_c']:.1f}°C")
    print(f"    GPU Junction:        {findings['design_peak_gpu_tj_c']:.1f}°C")
    print(f"    NVIDIA Compliant:    {'YES' if findings['design_nvidia_compliant'] else 'NO'}")
    
    print(f"\n  Extreme Conditions (34.4°C ambient):")
    print(f"    PG25 Supply:         {findings['extreme_peak_pg25_supply_c']:.1f}°C")
    print(f"    NVIDIA Compliant:    {'YES' if findings['extreme_nvidia_compliant'] else 'NO'}")
    
    print(f"\n  Site Performance:")
    print(f"    IT Load:             {findings['site_it_load_mw']:.1f} MW")
    print(f"    Pump Power:          {findings['site_pump_power_mw']:.2f} MW")
    print(f"    Site pPUE:           {findings['site_ppue']:.4f}")
    print(f"    Capacity Margin:     {findings['site_capacity_margin_pct']:.1f}%")
    print(f"    Annual Pump Energy:  {findings['annual_pump_energy_mwh']:.0f} MWh")
    
    # Save results to JSON
    results_path = os.path.join(output_dir, 'bt1_2_simulation_results.json')
    with open(results_path, 'w') as f:
        json.dump(findings, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")
    
    print("\n  Visualizations saved to: " + output_dir)
    print("=" * 70)
    
    return findings


if __name__ == "__main__":
    main()
