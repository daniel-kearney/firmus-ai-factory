"""Hydration: turn a validated BoD into runnable digital-twin objects.

The BoD is deliberately decoupled from the runtime classes so that:

    * The BoD schema can evolve independently of the physics models.
    * The physics models don't have to import BoD types.
    * Hydration is the single place where mapping rules live.

If the BoD requests a platform/cooling/region combination that the runtime
``factory_config.PLATFORM_CONFIG`` doesn't support, hydration raises early
with a clear message instead of blowing up deep inside a simulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from firmus_ai_factory.bod.schema import (
    BasisOfDesign,
    CoolingArchitecture,
    GridRegionCode,
    NVIDIAPlatformCode,
    TariffStructure,
)
from firmus_ai_factory.economics.electricity_tariff import ElectricityTariff
from firmus_ai_factory.factory_config import (
    CoolingType,
    FactoryConfig,
    FirmusAIFactory,
    GPUPlatform,
)
from firmus_ai_factory.grid.regional_grids import GridRegion


if TYPE_CHECKING:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Enum bridges (BoD strings <-> runtime enums)
# ---------------------------------------------------------------------------


_PLATFORM_BRIDGE = {
    NVIDIAPlatformCode.HGX_H100.value: GPUPlatform.HGX_H100,
    NVIDIAPlatformCode.HGX_H200.value: GPUPlatform.HGX_H200,
    NVIDIAPlatformCode.GB300_NVL72.value: GPUPlatform.GB300_NVL72,
    NVIDIAPlatformCode.VR_NVL72_MAX_P.value: GPUPlatform.VR_NVL72_MAX_P,
    NVIDIAPlatformCode.VR_NVL72_MAX_Q.value: GPUPlatform.VR_NVL72_MAX_Q,
    # HGX_B200 is defined in the BoD but not yet in PLATFORM_CONFIG.
}

_COOLING_BRIDGE = {
    CoolingArchitecture.IMMERSION_SINGLE_PHASE.value: CoolingType.IMMERSION,
    CoolingArchitecture.IMMERSION_TWO_PHASE.value: CoolingType.IMMERSION,
    CoolingArchitecture.BENMAX_HCU2500.value: CoolingType.BENMAX_HCU2500,
    # DTC / RDHx / air-only are valid BoD choices but not yet runtime-modelled.
}

_GRID_BRIDGE = {
    GridRegionCode.SINGAPORE.value: GridRegion.SINGAPORE,
    GridRegionCode.AUSTRALIA_NEM.value: GridRegion.AUSTRALIA_NEM,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def hydrate_factory_config(bod: BasisOfDesign) -> FactoryConfig:
    """Build a ``FactoryConfig`` from a BoD document.

    Rules:

    * Platform, cooling and region must all be individually representable in
      the runtime enums.
    * The combination must also match ``PLATFORM_CONFIG`` (enforced later by
      ``FirmusAIFactory.__init__``).  Hydration only guarantees the *types*
      round-trip; it does not silently rewrite invalid combinations.
    """

    platform_key = bod.nvidia_platform.platform
    cooling_key = bod.cooling.architecture
    grid_key = bod.grid.region

    if platform_key not in _PLATFORM_BRIDGE:
        raise NotImplementedError(
            f"Runtime does not yet support platform {platform_key!r}. "
            f"Supported: {sorted(_PLATFORM_BRIDGE.keys())}"
        )
    if cooling_key not in _COOLING_BRIDGE:
        raise NotImplementedError(
            f"Runtime does not yet support cooling architecture {cooling_key!r}. "
            f"Supported: {sorted(_COOLING_BRIDGE.keys())}"
        )
    if grid_key not in _GRID_BRIDGE:
        raise NotImplementedError(
            f"Runtime does not yet support grid region {grid_key!r}. "
            f"Supported: {sorted(_GRID_BRIDGE.keys())}"
        )

    # Prefer explicit ambient from climate; fall back to design dry bulb.
    ambient = bod.climate.design_dry_bulb_c

    return FactoryConfig(
        name=f"{bod.metadata.site_id} ({bod.metadata.bod_version})",
        platform=_PLATFORM_BRIDGE[platform_key],
        num_racks=bod.nvidia_platform.num_racks,
        cooling_type=_COOLING_BRIDGE[cooling_key],
        grid_region=_GRID_BRIDGE[grid_key],
        coolant_inlet_temp_c=bod.cooling.inlet_temp_c,
        ambient_temp_c=ambient,
    )


def hydrate_electricity_tariff(bod: BasisOfDesign) -> ElectricityTariff:
    """Build an ``ElectricityTariff`` from the BoD tariff block.

    Maps the string TOU periods straight onto the runtime tariff's ``rates``
    dict.  Negotiated PPAs are collapsed to a flat rate for the TOU model -
    the RoI pack handles the fully-articulated PPA separately.
    """
    structure = bod.tariff.structure
    if structure == TariffStructure.FLAT.value:
        tariff = ElectricityTariff(tariff_type="flat")
        flat_rate = next(iter(bod.tariff.rates_ccy_per_kwh.values()))
        tariff.rates = {
            "on_peak": flat_rate,
            "mid_peak": flat_rate,
            "off_peak": flat_rate,
        }
    elif structure == TariffStructure.TOU.value:
        tariff = ElectricityTariff(tariff_type="TOU")
        rates = dict(bod.tariff.rates_ccy_per_kwh)
        # Ensure the runtime's expected keys exist.
        for key in ("on_peak", "mid_peak", "off_peak"):
            rates.setdefault(key, rates.get("off_peak", 0.10))
        tariff.rates = {k: rates[k] for k in ("on_peak", "mid_peak", "off_peak")}
    elif structure == TariffStructure.RTP.value:
        tariff = ElectricityTariff(tariff_type="RTP")
        # RTP profiles come from the market simulator; seed with declared averages.
        tariff.rates = dict(bod.tariff.rates_ccy_per_kwh) or {
            "on_peak": 0.25,
            "mid_peak": 0.15,
            "off_peak": 0.08,
        }
    else:  # negotiated_ppa
        tariff = ElectricityTariff(tariff_type="flat")
        assert bod.tariff.ppa_ccy_per_mwh is not None  # validated in schema
        flat = bod.tariff.ppa_ccy_per_mwh / 1000.0
        tariff.rates = {"on_peak": flat, "mid_peak": flat, "off_peak": flat}

    tariff.demand_charge_rate = bod.tariff.demand_charge_ccy_per_kw_month
    tariff.monthly_fixed_charge = bod.tariff.fixed_charge_ccy_per_month
    return tariff


def hydrate_factory(bod: BasisOfDesign) -> FirmusAIFactory:
    """One-shot: BoD -> runnable ``FirmusAIFactory``."""
    return FirmusAIFactory(hydrate_factory_config(bod))
