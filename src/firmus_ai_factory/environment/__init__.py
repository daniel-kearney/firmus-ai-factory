"""Site-aware environmental conditions for Firmus AI Factory.

Provides ASHRAE climatic design conditions, local grid energy mix,
and site-specific parameters for all Firmus data center locations.
"""

from .site_conditions import (
    ASHRAEClimateZone,
    ASHRAEConditions,
    GridEnergyMix,
    GridConnection,
    NEM_Region,
    SiteConfig,
    SiteStatus,
    # Pre-defined ASHRAE conditions
    ASHRAE_LAUNCESTON,
    ASHRAE_MELBOURNE,
    ASHRAE_CANBERRA,
    ASHRAE_SYDNEY,
    ASHRAE_ROBERTSTOWN,
    ASHRAE_ALICE_SPRINGS,
    ASHRAE_BATAM,
    # Pre-defined grid mixes
    GRID_MIX_TASMANIA,
    GRID_MIX_SOUTH_AUSTRALIA,
    GRID_MIX_VICTORIA,
    GRID_MIX_NSW,
    GRID_MIX_NT_GAS,
    GRID_MIX_BATAM,
    # Pre-defined sites
    ALL_SITES,
    get_site,
    get_sites_by_region,
    get_sites_by_gpu,
    get_sites_by_provider,
    portfolio_summary,
)

__all__ = [
    "ASHRAEClimateZone",
    "ASHRAEConditions",
    "GridEnergyMix",
    "GridConnection",
    "NEM_Region",
    "SiteConfig",
    "SiteStatus",
    "ASHRAE_LAUNCESTON",
    "ASHRAE_MELBOURNE",
    "ASHRAE_CANBERRA",
    "ASHRAE_SYDNEY",
    "ASHRAE_ROBERTSTOWN",
    "ASHRAE_ALICE_SPRINGS",
    "ASHRAE_BATAM",
    "GRID_MIX_TASMANIA",
    "GRID_MIX_SOUTH_AUSTRALIA",
    "GRID_MIX_VICTORIA",
    "GRID_MIX_NSW",
    "GRID_MIX_NT_GAS",
    "GRID_MIX_BATAM",
    "ALL_SITES",
    "get_site",
    "get_sites_by_region",
    "get_sites_by_gpu",
    "get_sites_by_provider",
    "portfolio_summary",
    # Site-aware factory builder
    "create_site_factory",
    "site_thermal_analysis",
    "site_comparison_report",
    "extreme_condition_analysis",
]

from .site_factory import (
    create_site_factory,
    site_thermal_analysis,
    site_comparison_report,
    extreme_condition_analysis,
)
