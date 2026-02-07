"""Example 06: Site-Aware Environmental Analysis.

Demonstrates the site-aware environmental conditions module that
integrates ASHRAE climatic data, local grid energy mix, and
Benmax HCU2500 cooling performance for all Firmus data center sites.

Data sourced from:
- Firmus-Southgate Master Capacity Planning spreadsheet
- ASHRAE Climatic Design Conditions 2021
- Australian NEM grid mix data
- NVIDIA CDU Self-Qualification Guidelines DA-12515-001

Usage:
    python examples/06_site_aware_analysis.py
"""

import json
from firmus_ai_factory.environment.site_conditions import (
    ALL_SITES, get_site, get_sites_by_region, get_sites_by_gpu,
    portfolio_summary,
)
from firmus_ai_factory.environment.site_factory import (
    create_site_factory, site_thermal_analysis,
    site_comparison_report, extreme_condition_analysis,
)


def section_header(title: str):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def main():
    # =========================================================================
    # 1. Portfolio Overview
    # =========================================================================
    section_header("FIRMUS AI FACTORY — Portfolio Overview")
    
    summary = portfolio_summary()
    print(f"\n  Total Sites:     {summary['total_sites']}")
    print(f"  Total Capacity:  {summary['total_gross_mw']:.0f} MW")
    print(f"  Total GPUs:      {summary['total_gpus']:,}")
    print(f"  Weighted PUE:    {summary['weighted_avg_pue']:.3f}")
    
    print(f"\n  GPU Breakdown:")
    for gpu, count in summary['gpu_breakdown'].items():
        print(f"    {gpu}: {count:,} GPUs")
    
    print(f"\n  Region Breakdown:")
    for region, data in summary['region_breakdown'].items():
        print(f"    {region}: {data['sites']} sites, "
              f"{data['total_mw']:.0f} MW, {data['total_gpus']:,} GPUs")
    
    # =========================================================================
    # 2. Per-Site Environmental Conditions
    # =========================================================================
    section_header("Site Environmental Conditions (ASHRAE 2021)")
    
    print(f"\n  {'Site':<12} {'Campus':<16} {'Region':<16} "
          f"{'Climate':<8} {'Avg°C':>6} {'Design°C':>9} {'Extreme°C':>10} "
          f"{'Free Cool':>10}")
    print(f"  {'-'*12} {'-'*16} {'-'*16} {'-'*8} {'-'*6} {'-'*9} "
          f"{'-'*10} {'-'*10}")
    
    for code, site in ALL_SITES.items():
        ashrae = site.ashrae
        free_hours = ashrae.free_cooling_hours()
        print(f"  {code:<12} {site.campus:<16} {site.region:<16} "
              f"{ashrae.climate_zone.value:<8} "
              f"{ashrae.annual_avg_db:>5.1f} "
              f"{ashrae.cooling_04_db:>8.1f} "
              f"{ashrae.extreme_max_db:>9.1f} "
              f"{free_hours:>9}")
    
    # =========================================================================
    # 3. Grid Energy Mix Comparison
    # =========================================================================
    section_header("Grid Energy Mix by Site")
    
    print(f"\n  {'Site':<12} {'Grid Mix':<20} {'Renewable':>10} "
          f"{'Carbon':>12} {'Annual CO2':>12}")
    print(f"  {'':12} {'':20} {'Fraction':>10} "
          f"{'kg/MWh':>12} {'tonnes/yr':>12}")
    print(f"  {'-'*12} {'-'*20} {'-'*10} {'-'*12} {'-'*12}")
    
    for code, site in ALL_SITES.items():
        grid = site.grid
        annual_co2 = grid.energy_mix.annual_carbon_tonnes(site.it_power_mw)
        mix_desc = f"H{grid.energy_mix.hydro_pct:.0f}/W{grid.energy_mix.wind_pct:.0f}/S{grid.energy_mix.solar_pct:.0f}/G{grid.energy_mix.gas_pct:.0f}"
        print(f"  {code:<12} {mix_desc:<20} "
              f"{grid.energy_mix.renewable_fraction:>9.1%} "
              f"{grid.energy_mix.carbon_intensity_kg_mwh:>11.0f} "
              f"{annual_co2:>11,.0f}")
    
    # =========================================================================
    # 4. Site-Aware Factory Analysis (Selected Sites)
    # =========================================================================
    section_header("Site-Aware Factory Analysis")
    
    # Analyze a selection of sites across different climates
    analysis_sites = ["LN2/LN3", "RT1", "BK2", "MP1"]
    
    for code in analysis_sites:
        site = get_site(code)
        print(f"\n  --- {code}: {site.campus} ({site.region}) ---")
        print(f"  GPU: {site.gpu_series} | Racks: {site.num_racks} | "
              f"Gross: {site.gross_mw:.0f} MW | PUE: {site.pue:.2f}")
        
        try:
            factory = create_site_factory(code)
            report = factory.generate_full_report()
            
            print(f"  IT Power: {report['power']['it_power_mw']:.1f} MW")
            print(f"  Total Facility: {report['power']['total_facility_power_mw']:.1f} MW")
            print(f"  Computed PUE: {report['power']['pue']:.3f}")
            
            thermal = report.get('thermal', {})
            if 'per_rack' in thermal:
                rack = thermal['per_rack']
                print(f"  Coolant Inlet: {rack.get('inlet_temp_c', 'N/A')}°C")
                print(f"  Coolant Outlet: {rack.get('outlet_temp_c', 'N/A')}°C")
                print(f"  Within Limits: {rack.get('within_limits', 'N/A')}")
            
            if 'hypercube' in thermal:
                hc = thermal['hypercube']
                print(f"  Cooling Capacity: {hc.get('cooling_capacity_kw', 'N/A')} kW")
                print(f"  Capacity Margin: {hc.get('capacity_margin_pct', 'N/A')}%")
                print(f"  pPUE: {hc.get('pPUE', 'N/A')}")
            
            if 'nvidia_compliance' in thermal:
                print(f"  NVIDIA Compliant: {thermal.get('all_compliant', 'N/A')}")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    # =========================================================================
    # 5. Monthly Thermal Analysis (Robertstown — Hot Dry Climate)
    # =========================================================================
    section_header("Monthly Thermal Analysis: Robertstown (RT1)")
    
    try:
        thermal_report = site_thermal_analysis("RT1")
        
        print(f"\n  Climate Zone: {thermal_report['climate_zone']}")
        print(f"  GPU: {thermal_report['gpu_series']} | "
              f"Racks: {thermal_report['num_racks']}")
        
        print(f"\n  {'Month':>6} {'Ambient':>9} {'Inlet':>8} "
              f"{'Outlet':>8} {'ΔT':>6} {'Flow':>8} {'OK':>5}")
        print(f"  {'-'*6} {'-'*9} {'-'*8} {'-'*8} {'-'*6} {'-'*8} {'-'*5}")
        
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        
        for r in thermal_report['monthly']:
            m = months[r['month'] - 1]
            outlet = r.get('outlet_temp_c', 'N/A')
            delta = r.get('delta_t_c', 'N/A')
            flow = r.get('flowrate_lpm', 'N/A')
            ok = r.get('within_limits', 'N/A')
            
            outlet_str = f"{outlet:.1f}" if isinstance(outlet, (int, float)) else str(outlet)
            delta_str = f"{delta:.1f}" if isinstance(delta, (int, float)) else str(delta)
            flow_str = f"{flow:.1f}" if isinstance(flow, (int, float)) else str(flow)
            
            print(f"  {m:>6} {r['ambient_temp_c']:>8.1f} "
                  f"{r['coolant_inlet_c']:>7.1f} "
                  f"{outlet_str:>7} {delta_str:>5} "
                  f"{flow_str:>7} {'✓' if ok else '✗':>4}")
        
        annual = thermal_report['annual_summary']
        print(f"\n  Annual Summary:")
        print(f"    Months Within Limits: {annual['months_within_thermal_limits']}/12")
        print(f"    NVIDIA Compliant: {annual['months_nvidia_compliant']}/12")
        print(f"    Worst Month: {months[annual['worst_month']-1]} "
              f"({annual['worst_ambient_c']:.1f}°C)")
        print(f"    Best Month: {months[annual['best_month']-1]} "
              f"({annual['best_ambient_c']:.1f}°C)")
        print(f"    Free Cooling Hours: {annual['free_cooling_hours']}")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    # =========================================================================
    # 6. Extreme Condition Analysis (Alice Springs — Hottest Site)
    # =========================================================================
    section_header("Extreme Condition Analysis: Alice Springs (MP1)")
    
    try:
        extreme = extreme_condition_analysis("MP1")
        
        print(f"\n  Site: {extreme['site']} ({extreme['campus']})")
        print(f"  Region: {extreme['region']}")
        
        for scenario, data in extreme['scenarios'].items():
            status = "✓ OK" if data.get('thermal_ok') else "✗ FAIL"
            print(f"\n  {scenario.replace('_', ' ').title()}:")
            print(f"    Ambient: {data['ambient_temp_c']:.1f}°C")
            print(f"    Coolant Inlet: {data['coolant_inlet_c']:.1f}°C")
            print(f"    Status: {status}")
            if 'pue' in data:
                print(f"    PUE: {data['pue']:.3f}")
            if 'error' in data:
                print(f"    Error: {data['error']}")
        
        headroom = extreme['thermal_headroom']
        print(f"\n  Thermal Headroom:")
        print(f"    NVIDIA Max Inlet: {headroom['nvidia_max_inlet_c']}°C")
        print(f"    Extreme Coolant: {headroom['extreme_coolant_inlet_c']:.1f}°C")
        print(f"    Headroom: {headroom['headroom_c']:.1f}°C")
        print(f"    Operable at Extreme: {headroom['extreme_operable']}")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    # =========================================================================
    # 7. Cross-Site Comparison Rankings
    # =========================================================================
    section_header("Cross-Site Comparison Rankings")
    
    try:
        comparison = site_comparison_report()
        rankings = comparison['rankings']
        
        print(f"\n  Best Cooling Efficiency (most free cooling hours):")
        for i, code in enumerate(rankings['best_cooling_efficiency'][:5], 1):
            site = get_site(code)
            hours = site.ashrae.free_cooling_hours()
            print(f"    {i}. {code:<12} {site.campus:<16} {hours} hours")
        
        print(f"\n  Lowest Carbon Intensity:")
        for i, code in enumerate(rankings['lowest_carbon_intensity'][:5], 1):
            site = get_site(code)
            ci = site.grid.energy_mix.carbon_intensity_kg_mwh
            print(f"    {i}. {code:<12} {site.campus:<16} {ci:.0f} kg CO₂/MWh")
        
        print(f"\n  Best Thermal Conditions (lowest worst-case ambient):")
        for i, code in enumerate(rankings['best_thermal_conditions'][:5], 1):
            site = get_site(code)
            worst = site.ashrae.cooling_04_db
            print(f"    {i}. {code:<12} {site.campus:<16} {worst:.1f}°C design")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    print(f"\n{'='*80}")
    print(f"  Analysis Complete")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
