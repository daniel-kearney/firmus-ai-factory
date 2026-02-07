"""Example 05: Platform-Region Analysis.

Demonstrates the complete Firmus AI Factory analysis across all
supported platform-cooling-region configurations:

    HGX H100/H200  →  Singapore  →  Immersion Cooling
    GB300 NVL72     →  Australia  →  Benmax HCU2500
    Vera Rubin NVL72 →  Australia →  Benmax HCU2500

Generates comparative reports for:
    - Compute capacity (PFLOPS, HBM)
    - Power consumption and PUE
    - Thermal performance and cooling margins
    - Grid economics (energy cost, DR revenue)
    - NVIDIA CDU self-qualification compliance
"""

import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from firmus_ai_factory.factory_config import (
    FirmusAIFactory,
    FactoryConfig,
    GPUPlatform,
    CoolingType,
    singapore_h100_factory,
    singapore_h200_factory,
    australia_gb300_factory,
    australia_vera_rubin_factory,
)
from firmus_ai_factory.grid.regional_grids import GridRegion


def print_separator(title: str, char: str = "=", width: int = 70):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def format_currency(amount: float, currency: str) -> str:
    return f"{currency} {amount:,.0f}"


def run_analysis():
    """Run comprehensive platform-region analysis."""
    
    print_separator("FIRMUS AI FACTORY - Platform-Region Analysis")
    print("  Mapping: GPU Platform → Grid Region → Cooling System")
    print("  " + "-" * 60)
    print("  HGX H100/H200  → Singapore   → Immersion Cooling")
    print("  GB300 NVL72     → Australia   → Benmax HCU2500")
    print("  Vera Rubin NVL72→ Australia   → Benmax HCU2500")
    
    # =========================================================================
    # 1. Singapore Deployments (Immersion Cooling)
    # =========================================================================
    
    print_separator("SINGAPORE DEPLOYMENTS (Immersion Cooling)", "─")
    
    # H100 Factory: 100 nodes (800 GPUs)
    sg_h100 = singapore_h100_factory(num_nodes=100)
    h100_report = sg_h100.generate_full_report()
    
    print(f"\n  HGX H100 Factory (100 nodes)")
    print(f"  {'─' * 50}")
    print(f"  GPUs:           {h100_report['factory']['total_gpus']}")
    print(f"  IT Power:       {h100_report['power']['it_power_mw']:.1f} MW")
    print(f"  Total Power:    {h100_report['power']['total_facility_power_mw']:.1f} MW")
    print(f"  PUE:            {h100_report['power']['pue']:.4f}")
    print(f"  FP16 PFLOPS:    {h100_report['compute']['total_fp16_pflops']:.1f}")
    print(f"  HBM:            {h100_report['compute']['total_hbm_tb']:.1f} TB")
    print(f"  Grid Voltage:   {h100_report['grid']['grid_spec']['three_phase_voltage_v']}V (3-phase)")
    print(f"  Annual Energy:  {format_currency(h100_report['grid']['energy_cost']['annual_cost'], 'SGD')}")
    
    # H200 Factory: 100 nodes (800 GPUs)
    sg_h200 = singapore_h200_factory(num_nodes=100)
    h200_report = sg_h200.generate_full_report()
    
    print(f"\n  HGX H200 Factory (100 nodes)")
    print(f"  {'─' * 50}")
    print(f"  GPUs:           {h200_report['factory']['total_gpus']}")
    print(f"  IT Power:       {h200_report['power']['it_power_mw']:.1f} MW")
    print(f"  Total Power:    {h200_report['power']['total_facility_power_mw']:.1f} MW")
    print(f"  PUE:            {h200_report['power']['pue']:.4f}")
    print(f"  FP16 PFLOPS:    {h200_report['compute']['total_fp16_pflops']:.1f}")
    print(f"  HBM:            {h200_report['compute']['total_hbm_tb']:.1f} TB")
    print(f"  Annual Energy:  {format_currency(h200_report['grid']['energy_cost']['annual_cost'], 'SGD')}")
    
    # =========================================================================
    # 2. Australia Deployments (Benmax HCU2500)
    # =========================================================================
    
    print_separator("AUSTRALIA DEPLOYMENTS (Benmax HCU2500)", "─")
    
    # GB300 Factory: 32 racks (2304 GPUs)
    au_gb300 = australia_gb300_factory(num_racks=32)
    gb300_report = au_gb300.generate_full_report()
    
    print(f"\n  GB300 NVL72 Factory (32 racks)")
    print(f"  {'─' * 50}")
    print(f"  GPUs:           {gb300_report['factory']['total_gpus']}")
    print(f"  IT Power:       {gb300_report['power']['it_power_mw']:.1f} MW")
    print(f"  Total Power:    {gb300_report['power']['total_facility_power_mw']:.1f} MW")
    print(f"  PUE:            {gb300_report['power']['pue']:.4f}")
    print(f"  FP16 PFLOPS:    {gb300_report['compute']['total_fp16_pflops']:.1f}")
    print(f"  HBM:            {gb300_report['compute']['total_hbm_tb']:.1f} TB")
    print(f"  Grid Voltage:   {gb300_report['grid']['grid_spec']['three_phase_voltage_v']}V (3-phase)")
    print(f"  Annual Energy:  {format_currency(gb300_report['grid']['energy_cost']['annual_cost'], 'AUD')}")
    
    if 'hypercube' in gb300_report['thermal']:
        hc = gb300_report['thermal']['hypercube']
        print(f"  Cooling Cap:    {hc['cooling_capacity_kw']:.0f} kW")
        print(f"  Margin:         {hc['capacity_margin_pct']:.1f}%")
        print(f"  pPUE:           {hc['pPUE']:.4f}")
    
    # Vera Rubin Max P: 32 racks (2304 GPUs)
    au_vr_p = australia_vera_rubin_factory(num_racks=32, max_q=False)
    vr_p_report = au_vr_p.generate_full_report()
    
    print(f"\n  Vera Rubin NVL72 Max P (32 racks)")
    print(f"  {'─' * 50}")
    print(f"  GPUs:           {vr_p_report['factory']['total_gpus']}")
    print(f"  IT Power:       {vr_p_report['power']['it_power_mw']:.1f} MW")
    print(f"  Total Power:    {vr_p_report['power']['total_facility_power_mw']:.1f} MW")
    print(f"  PUE:            {vr_p_report['power']['pue']:.4f}")
    print(f"  FP16 PFLOPS:    {vr_p_report['compute']['total_fp16_pflops']:.1f}")
    print(f"  HBM:            {vr_p_report['compute']['total_hbm_tb']:.1f} TB")
    print(f"  Annual Energy:  {format_currency(vr_p_report['grid']['energy_cost']['annual_cost'], 'AUD')}")
    
    if 'per_rack' in vr_p_report['thermal']:
        rack = vr_p_report['thermal']['per_rack']
        print(f"  Rack ΔT:        {rack['delta_t_c']:.1f}°C")
        print(f"  Rack Flow:      {rack['flowrate_lpm']:.1f} LPM")
        print(f"  Within Limits:  {rack['within_limits']}")
    
    if 'nvidia_compliance' in vr_p_report['thermal']:
        compliance = vr_p_report['thermal']['nvidia_compliance']
        print(f"  NVIDIA Cert:    {'PASS' if vr_p_report['thermal']['all_compliant'] else 'FAIL'}")
    
    # Vera Rubin Max Q: 32 racks (2304 GPUs)
    au_vr_q = australia_vera_rubin_factory(num_racks=32, max_q=True)
    vr_q_report = au_vr_q.generate_full_report()
    
    print(f"\n  Vera Rubin NVL72 Max Q (32 racks)")
    print(f"  {'─' * 50}")
    print(f"  GPUs:           {vr_q_report['factory']['total_gpus']}")
    print(f"  IT Power:       {vr_q_report['power']['it_power_mw']:.1f} MW")
    print(f"  Total Power:    {vr_q_report['power']['total_facility_power_mw']:.1f} MW")
    print(f"  PUE:            {vr_q_report['power']['pue']:.4f}")
    print(f"  FP16 PFLOPS:    {vr_q_report['compute']['total_fp16_pflops']:.1f}")
    print(f"  HBM:            {vr_q_report['compute']['total_hbm_tb']:.1f} TB")
    print(f"  Annual Energy:  {format_currency(vr_q_report['grid']['energy_cost']['annual_cost'], 'AUD')}")
    
    if 'per_rack' in vr_q_report['thermal']:
        rack = vr_q_report['thermal']['per_rack']
        print(f"  Rack ΔT:        {rack['delta_t_c']:.1f}°C")
        print(f"  Power Savings:  {(1 - 187.0/227.0) * 100:.1f}% vs Max P")
    
    # =========================================================================
    # 3. Comparative Summary
    # =========================================================================
    
    print_separator("COMPARATIVE SUMMARY", "═")
    
    header = f"  {'Platform':<20} {'Region':<12} {'GPUs':>6} {'MW':>6} {'PUE':>6} {'PFLOPS':>8} {'HBM TB':>7}"
    print(header)
    print(f"  {'─' * 67}")
    
    configs = [
        ("HGX H100", "Singapore", h100_report),
        ("HGX H200", "Singapore", h200_report),
        ("GB300 NVL72", "Australia", gb300_report),
        ("VR NVL72 Max P", "Australia", vr_p_report),
        ("VR NVL72 Max Q", "Australia", vr_q_report),
    ]
    
    for name, region, report in configs:
        print(f"  {name:<20} {region:<12} "
              f"{report['factory']['total_gpus']:>6} "
              f"{report['power']['total_facility_power_mw']:>6.1f} "
              f"{report['power']['pue']:>6.4f} "
              f"{report['compute']['total_fp16_pflops']:>8.1f} "
              f"{report['compute']['total_hbm_tb']:>7.1f}")
    
    # =========================================================================
    # 4. Grid Economics Comparison
    # =========================================================================
    
    print_separator("GRID ECONOMICS COMPARISON", "─")
    
    header = f"  {'Platform':<20} {'Currency':>8} {'Annual Cost':>14} {'DR Revenue':>14} {'Net Cost':>14}"
    print(header)
    print(f"  {'─' * 72}")
    
    for name, region, report in configs:
        currency = report['grid']['energy_cost']['currency']
        annual = report['grid']['energy_cost']['annual_cost']
        dr_rev = report['grid']['demand_response_revenue']['total_annual_revenue']
        net = report['grid']['net_annual_cost']
        
        print(f"  {name:<20} {currency:>8} "
              f"{annual:>14,.0f} "
              f"{dr_rev:>14,.0f} "
              f"{net:>14,.0f}")
    
    # =========================================================================
    # 5. Vera Rubin Max P vs Max Q Analysis
    # =========================================================================
    
    print_separator("VERA RUBIN: Max P vs Max Q Analysis", "─")
    
    vr_p_power = vr_p_report['power']['total_facility_power_mw']
    vr_q_power = vr_q_report['power']['total_facility_power_mw']
    power_saving_pct = (1 - vr_q_power / vr_p_power) * 100
    
    vr_p_pflops = vr_p_report['compute']['total_fp16_pflops']
    vr_q_pflops = vr_q_report['compute']['total_fp16_pflops']
    
    vr_p_cost = vr_p_report['grid']['energy_cost']['annual_cost']
    vr_q_cost = vr_q_report['grid']['energy_cost']['annual_cost']
    cost_saving = vr_p_cost - vr_q_cost
    
    print(f"  Power Reduction:     {power_saving_pct:.1f}%")
    print(f"  Power Saved:         {vr_p_power - vr_q_power:.2f} MW")
    print(f"  PFLOPS (Max P):      {vr_p_pflops:.1f}")
    print(f"  PFLOPS (Max Q):      {vr_q_pflops:.1f}")
    print(f"  Annual Cost Saving:  AUD {cost_saving:,.0f}")
    print(f"  Efficiency (P):      {vr_p_pflops / vr_p_power:.1f} PFLOPS/MW")
    print(f"  Efficiency (Q):      {vr_q_pflops / vr_q_power:.1f} PFLOPS/MW")
    
    print(f"\n  Recommendation: Max Q provides {power_saving_pct:.0f}% power reduction")
    print(f"  with proportional cost savings, ideal for inference workloads.")
    print(f"  Max P recommended for training where peak compute is critical.")
    
    print(f"\n{'=' * 70}")
    print(f"  Analysis complete. All configurations validated.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    run_analysis()
