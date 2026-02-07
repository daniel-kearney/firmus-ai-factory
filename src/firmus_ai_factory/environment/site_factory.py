"""Site-Aware Factory Builder for Firmus AI Factory.

Creates FirmusAIFactory instances that are configured with site-specific
ASHRAE environmental conditions, local grid energy mix, and ambient
temperature profiles. This bridges the environment module with the
factory configuration module.

Usage:
    from firmus_ai_factory.environment.site_factory import (
        create_site_factory, site_comparison_report
    )
    
    # Create factory for a specific site
    factory = create_site_factory("RT1")
    report = factory.generate_full_report()
    
    # Compare all sites
    comparison = site_comparison_report()
"""

from typing import Dict, List, Optional, Tuple
import math

from firmus_ai_factory.factory_config import (
    FactoryConfig, FirmusAIFactory,
    GPUPlatform, CoolingType,
)
from firmus_ai_factory.grid.regional_grids import GridRegion
from firmus_ai_factory.environment.site_conditions import (
    SiteConfig, ALL_SITES, get_site, portfolio_summary,
    get_sites_by_region, get_sites_by_gpu,
)


# =============================================================================
# GPU Series to Platform Mapping
# =============================================================================

def _resolve_platform(site: SiteConfig) -> GPUPlatform:
    """Map site GPU series and NV code to GPUPlatform enum.
    
    Args:
        site: SiteConfig with gpu_series and nv_code
    
    Returns:
        GPUPlatform enum value
    """
    series = site.gpu_series.upper()
    
    if series in ("H100",):
        return GPUPlatform.HGX_H100
    elif series in ("H200",):
        return GPUPlatform.HGX_H200
    elif series in ("GB300",):
        return GPUPlatform.GB300_NVL72
    elif series in ("VR", "VERA RUBIN"):
        # Default to Max P for Vera Rubin
        return GPUPlatform.VR_NVL72_MAX_P
    else:
        raise ValueError(f"Unknown GPU series: {series}")


def _resolve_cooling(platform: GPUPlatform) -> CoolingType:
    """Resolve cooling type from platform.
    
    Enforces canonical mapping:
        HGX H100/H200 → Immersion
        GB300/VR NVL72 → Benmax HCU2500
    """
    if platform in (GPUPlatform.HGX_H100, GPUPlatform.HGX_H200):
        return CoolingType.IMMERSION
    else:
        return CoolingType.BENMAX_HCU2500


def _resolve_grid_region(site: SiteConfig) -> GridRegion:
    """Resolve grid region from site location.
    
    Enforces canonical mapping:
        HGX H100/H200 → Singapore
        GB300/VR NVL72 → Australia NEM
    """
    platform = _resolve_platform(site)
    if platform in (GPUPlatform.HGX_H100, GPUPlatform.HGX_H200):
        return GridRegion.SINGAPORE
    else:
        return GridRegion.AUSTRALIA_NEM


# =============================================================================
# Site-Aware Factory Creation
# =============================================================================

def create_site_factory(
    dc_code: str,
    vr_max_q: bool = False,
    coolant_inlet_override_c: Optional[float] = None,
) -> FirmusAIFactory:
    """Create a FirmusAIFactory configured for a specific site.
    
    Uses the site's ASHRAE conditions to set ambient temperature
    to the 0.4% cooling design dry bulb (worst-case design condition).
    
    Args:
        dc_code: Data center code (e.g., 'LN2/LN3', 'RT1', 'BK2')
        vr_max_q: If True, use VR Max Q instead of Max P for VR sites
        coolant_inlet_override_c: Override coolant inlet temperature
    
    Returns:
        FirmusAIFactory configured for the site
    
    Raises:
        KeyError: If dc_code not found
        ValueError: If platform-cooling-region mapping is invalid
    """
    site = get_site(dc_code)
    
    # Resolve platform (handle VR Max Q override)
    platform = _resolve_platform(site)
    if vr_max_q and platform == GPUPlatform.VR_NVL72_MAX_P:
        platform = GPUPlatform.VR_NVL72_MAX_Q
    
    cooling = _resolve_cooling(platform)
    grid_region = _resolve_grid_region(site)
    
    # Use ASHRAE 0.4% cooling design temperature as ambient
    design_ambient = site.ashrae.cooling_04_db
    
    # Coolant inlet: approach temperature above ambient
    # Typical CDU approach: 5-8°C above ambient
    approach_temp = 7.0
    if coolant_inlet_override_c is not None:
        coolant_inlet = coolant_inlet_override_c
    else:
        # Cap at NVIDIA max inlet (45°C for VR NVL72)
        coolant_inlet = min(design_ambient + approach_temp, 45.0)
    
    config = FactoryConfig(
        name=f"Firmus {dc_code}",
        platform=platform,
        num_racks=site.num_racks,
        cooling_type=cooling,
        grid_region=grid_region,
        coolant_inlet_temp_c=coolant_inlet,
        ambient_temp_c=design_ambient,
    )
    
    return FirmusAIFactory(config)


# =============================================================================
# Site-Aware Thermal Analysis
# =============================================================================

def site_thermal_analysis(
    dc_code: str,
    vr_max_q: bool = False,
) -> Dict:
    """Perform site-aware thermal analysis using ASHRAE conditions.
    
    Analyzes cooling performance across all 12 months using the
    site's monthly average temperatures.
    
    Args:
        dc_code: Data center code
        vr_max_q: If True, use VR Max Q mode
    
    Returns:
        Dict with monthly thermal analysis and annual summary
    """
    site = get_site(dc_code)
    
    monthly_results = []
    for month in range(1, 13):
        ambient = site.ashrae.get_ambient_temp(month, "avg")
        
        # Create factory with monthly ambient
        approach_temp = 7.0
        coolant_inlet = min(ambient + approach_temp, 45.0)
        
        platform = _resolve_platform(site)
        if vr_max_q and platform == GPUPlatform.VR_NVL72_MAX_P:
            platform = GPUPlatform.VR_NVL72_MAX_Q
        
        cooling = _resolve_cooling(platform)
        grid_region = _resolve_grid_region(site)
        
        config = FactoryConfig(
            name=f"Firmus {dc_code} M{month:02d}",
            platform=platform,
            num_racks=site.num_racks,
            cooling_type=cooling,
            grid_region=grid_region,
            coolant_inlet_temp_c=coolant_inlet,
            ambient_temp_c=ambient,
        )
        
        factory = FirmusAIFactory(config)
        thermal = factory.thermal_summary()
        
        # Extract key metrics
        result = {
            'month': month,
            'ambient_temp_c': ambient,
            'coolant_inlet_c': coolant_inlet,
        }
        
        if cooling == CoolingType.BENMAX_HCU2500:
            result.update({
                'outlet_temp_c': thermal.get('per_rack', {}).get('outlet_temp_c'),
                'delta_t_c': thermal.get('per_rack', {}).get('delta_t_c'),
                'flowrate_lpm': thermal.get('per_rack', {}).get('flowrate_lpm'),
                'within_limits': thermal.get('per_rack', {}).get('within_limits'),
                'cooling_capacity_kw': thermal.get('hypercube', {}).get('cooling_capacity_kw'),
                'capacity_margin_pct': thermal.get('hypercube', {}).get('capacity_margin_pct'),
                'pPUE': thermal.get('hypercube', {}).get('pPUE'),
                'nvidia_compliant': thermal.get('all_compliant'),
            })
        else:
            result.update({
                'cooling_type': 'immersion',
                'note': 'Immersion cooling — ambient-independent',
            })
        
        monthly_results.append(result)
    
    # Annual summary
    if monthly_results and 'within_limits' in monthly_results[0]:
        months_within_limits = sum(
            1 for r in monthly_results if r.get('within_limits', False))
        months_nvidia_compliant = sum(
            1 for r in monthly_results if r.get('nvidia_compliant', False))
        worst_month = max(monthly_results, key=lambda r: r['ambient_temp_c'])
        best_month = min(monthly_results, key=lambda r: r['ambient_temp_c'])
        avg_ppue = sum(
            r.get('pPUE', 0) for r in monthly_results) / 12
    else:
        months_within_limits = 12
        months_nvidia_compliant = 12
        worst_month = max(monthly_results, key=lambda r: r['ambient_temp_c'])
        best_month = min(monthly_results, key=lambda r: r['ambient_temp_c'])
        avg_ppue = 1.05  # Immersion estimate
    
    return {
        'site': dc_code,
        'campus': site.campus,
        'region': site.region,
        'climate_zone': site.ashrae.climate_zone.value,
        'gpu_series': site.gpu_series,
        'num_racks': site.num_racks,
        'monthly': monthly_results,
        'annual_summary': {
            'months_within_thermal_limits': months_within_limits,
            'months_nvidia_compliant': months_nvidia_compliant,
            'worst_month': worst_month['month'],
            'worst_ambient_c': worst_month['ambient_temp_c'],
            'best_month': best_month['month'],
            'best_ambient_c': best_month['ambient_temp_c'],
            'annual_avg_ambient_c': site.ashrae.annual_avg_db,
            'design_cooling_c': site.ashrae.cooling_04_db,
            'extreme_max_c': site.ashrae.extreme_max_db,
            'avg_pPUE': avg_ppue,
            'free_cooling_hours': site.ashrae.free_cooling_hours(),
        },
    }


# =============================================================================
# Multi-Site Comparison
# =============================================================================

def site_comparison_report(
    dc_codes: Optional[List[str]] = None,
) -> Dict:
    """Generate comparison report across multiple sites.
    
    Args:
        dc_codes: List of site codes to compare. If None, compare all.
    
    Returns:
        Dict with per-site analysis and cross-site comparison
    """
    if dc_codes is None:
        dc_codes = list(ALL_SITES.keys())
    
    site_reports = {}
    for code in dc_codes:
        try:
            site = get_site(code)
            env_report = site.generate_site_report()
            thermal_report = site_thermal_analysis(code)
            
            site_reports[code] = {
                'environment': env_report,
                'thermal': thermal_report,
            }
        except Exception as e:
            site_reports[code] = {'error': str(e)}
    
    # Cross-site rankings
    valid_sites = {k: v for k, v in site_reports.items() if 'error' not in v}
    
    if valid_sites:
        # Rank by cooling efficiency (free cooling hours)
        cooling_ranking = sorted(
            valid_sites.keys(),
            key=lambda k: valid_sites[k]['thermal']['annual_summary'].get(
                'free_cooling_hours', 0),
            reverse=True,
        )
        
        # Rank by carbon intensity (lower is better)
        carbon_ranking = sorted(
            valid_sites.keys(),
            key=lambda k: valid_sites[k]['environment']['grid'][
                'carbon_intensity_kg_mwh'],
        )
        
        # Rank by thermal margin (worst month ambient, lower is better)
        thermal_ranking = sorted(
            valid_sites.keys(),
            key=lambda k: valid_sites[k]['thermal']['annual_summary'].get(
                'worst_ambient_c', 50),
        )
    else:
        cooling_ranking = []
        carbon_ranking = []
        thermal_ranking = []
    
    return {
        'sites': site_reports,
        'rankings': {
            'best_cooling_efficiency': cooling_ranking,
            'lowest_carbon_intensity': carbon_ranking,
            'best_thermal_conditions': thermal_ranking,
        },
        'portfolio': portfolio_summary(),
    }


# =============================================================================
# Extreme Condition Analysis
# =============================================================================

def extreme_condition_analysis(dc_code: str) -> Dict:
    """Analyze factory performance under extreme ASHRAE conditions.
    
    Tests the factory at three temperature scenarios:
    1. Annual average (typical operation)
    2. 0.4% cooling design (design condition)
    3. Extreme maximum (worst-case)
    
    Args:
        dc_code: Data center code
    
    Returns:
        Dict with performance at each condition
    """
    site = get_site(dc_code)
    
    scenarios = {
        'annual_average': site.ashrae.annual_avg_db,
        'design_0_4_pct': site.ashrae.cooling_04_db,
        'extreme_maximum': site.ashrae.extreme_max_db,
    }
    
    results = {}
    for scenario_name, ambient_temp in scenarios.items():
        approach_temp = 7.0
        coolant_inlet = min(ambient_temp + approach_temp, 45.0)
        
        try:
            factory = create_site_factory(
                dc_code, coolant_inlet_override_c=coolant_inlet)
            factory.config.ambient_temp_c = ambient_temp
            
            report = factory.generate_full_report()
            
            results[scenario_name] = {
                'ambient_temp_c': ambient_temp,
                'coolant_inlet_c': coolant_inlet,
                'pue': report['power']['pue'],
                'total_power_mw': report['power']['total_facility_power_mw'],
                'thermal_ok': True,
                'report': report,
            }
        except Exception as e:
            results[scenario_name] = {
                'ambient_temp_c': ambient_temp,
                'coolant_inlet_c': coolant_inlet,
                'thermal_ok': False,
                'error': str(e),
            }
    
    # Thermal headroom at extreme
    extreme = results.get('extreme_maximum', {})
    design = results.get('design_0_4_pct', {})
    
    return {
        'site': dc_code,
        'campus': site.campus,
        'region': site.region,
        'scenarios': results,
        'thermal_headroom': {
            'nvidia_max_inlet_c': 45.0,
            'extreme_coolant_inlet_c': extreme.get('coolant_inlet_c', 0),
            'headroom_c': 45.0 - extreme.get('coolant_inlet_c', 45.0),
            'extreme_operable': extreme.get('thermal_ok', False),
        },
    }


# =============================================================================
# Main — Site Portfolio Analysis
# =============================================================================

if __name__ == "__main__":
    import json
    
    print("=" * 80)
    print("  FIRMUS AI FACTORY — Site-Aware Environmental Analysis")
    print("=" * 80)
    
    # Portfolio overview
    summary = portfolio_summary()
    print(f"\nPortfolio: {summary['total_sites']} sites, "
          f"{summary['total_gross_mw']:.0f} MW, "
          f"{summary['total_gpus']:,} GPUs")
    
    # Per-site analysis
    for code in ALL_SITES:
        print(f"\n{'='*60}")
        print(f"  Analyzing: {code}")
        print(f"{'='*60}")
        
        try:
            thermal = site_thermal_analysis(code)
            annual = thermal['annual_summary']
            
            print(f"  Climate Zone: {thermal['climate_zone']}")
            print(f"  GPU: {thermal['gpu_series']} | Racks: {thermal['num_racks']}")
            print(f"  Annual Avg Ambient: {annual['annual_avg_ambient_c']:.1f}°C")
            print(f"  Design Cooling: {annual['design_cooling_c']:.1f}°C")
            print(f"  Extreme Max: {annual['extreme_max_c']:.1f}°C")
            print(f"  Months Within Limits: {annual['months_within_thermal_limits']}/12")
            print(f"  NVIDIA Compliant Months: {annual['months_nvidia_compliant']}/12")
            print(f"  Avg pPUE: {annual['avg_pPUE']:.3f}")
            print(f"  Free Cooling Hours: {annual['free_cooling_hours']}")
            
            # Extreme condition test
            extreme = extreme_condition_analysis(code)
            headroom = extreme['thermal_headroom']
            print(f"  Extreme Headroom: {headroom['headroom_c']:.1f}°C")
            print(f"  Extreme Operable: {headroom['extreme_operable']}")
            
        except Exception as e:
            print(f"  Error: {e}")
