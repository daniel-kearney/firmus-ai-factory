"""Canonical Basis of Design (BoD) schema for a Firmus AI Factory.

Version: 0.1.0

Design principles
-----------------

1.  **One document, nine domains.**  Every AI Factory is described by a single
    ``BasisOfDesign`` document with a fixed set of top-level domains.  Missing
    a domain is a validation error - a partial BoD is not a BoD.

2.  **Units are declared, not inferred.**  Every physical field carries its
    unit in the field name (``rack_power_kw``, ``inlet_temp_c``,
    ``tariff_currency_per_kwh``).  We do not silently convert kW <-> MW.

3.  **Design intent, not runtime state.**  The BoD captures what we intend to
    build.  Live sensor values, telemetry and setpoints live elsewhere.

4.  **Freezable.**  Once a BoD is approved for a gate (JOA G2/G3), the
    ``metadata.approval`` block is populated and the document is treated as
    immutable by the optimizer - the optimizer produces a *new* BoD.

5.  **Optimizer-friendly.**  Every field the optimizer is allowed to vary is
    declared explicitly under ``optimizer.free_variables``.  Everything else
    is a hard constraint.

The schema is a Pydantic model so we get:

    * ``BasisOfDesign.model_validate(...)`` - validate a dict/YAML
    * ``BasisOfDesign.model_json_schema()`` - JSON Schema export
    * ``.model_dump(mode='json')`` - deterministic serialisation
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "0.1.0"


# =============================================================================
# Shared enums (kept string-typed so the JSON schema is human-readable)
# =============================================================================


class CoolingArchitecture(str, Enum):
    """Top-level cooling architecture family."""

    IMMERSION_SINGLE_PHASE = "immersion_single_phase"
    IMMERSION_TWO_PHASE = "immersion_two_phase"
    DIRECT_TO_CHIP = "direct_to_chip"
    REAR_DOOR_HEAT_EXCHANGER = "rear_door_hx"
    AIR_ONLY = "air_only"
    BENMAX_HCU2500 = "benmax_hcu2500"  # Firmus AU reference design


class HeatRejectionType(str, Enum):
    DRY_COOLER = "dry_cooler"
    ADIABATIC = "adiabatic"
    COOLING_TOWER_OPEN = "cooling_tower_open"
    COOLING_TOWER_CLOSED = "cooling_tower_closed"
    CHILLER = "chiller"
    HYBRID = "hybrid"


class UPSTopology(str, Enum):
    DOUBLE_CONVERSION = "double_conversion"
    LINE_INTERACTIVE = "line_interactive"
    DYNAMIC_ROTARY = "dynamic_rotary"
    LITHIUM_ION_BESS = "lithium_ion_bess"
    NONE = "none"


class GridRegionCode(str, Enum):
    """String-typed grid regions used in the BoD.

    Kept independent of ``grid.regional_grids.GridRegion`` so BoD files stay
    stable if the runtime enum shifts.
    """

    SINGAPORE = "singapore"
    AUSTRALIA_NEM = "australia_nem"
    US_ERCOT = "us_ercot"
    US_PJM = "us_pjm"
    EU_ENTSOE = "eu_entsoe"
    OTHER = "other"


class TariffStructure(str, Enum):
    FLAT = "flat"
    TOU = "TOU"
    RTP = "RTP"
    NEGOTIATED_PPA = "negotiated_ppa"


class NVIDIAPlatformCode(str, Enum):
    HGX_H100 = "hgx_h100"
    HGX_H200 = "hgx_h200"
    HGX_B200 = "hgx_b200"
    GB300_NVL72 = "gb300_nvl72"
    VR_NVL72_MAX_P = "vr_nvl72_max_p"
    VR_NVL72_MAX_Q = "vr_nvl72_max_q"


# =============================================================================
# Base model config - forbid extras, use enum values in JSON
# =============================================================================


class _BoDModel(BaseModel):
    """Base for every BoD sub-model.

    We ``forbid`` extras so typos in a BoD YAML fail validation instead of
    being silently ignored.  We serialise enums as their string values so the
    JSON Schema and dumped YAML round-trip cleanly.
    """

    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )


# =============================================================================
# Metadata
# =============================================================================


class Approval(_BoDModel):
    """Gate approval record.  Present only once the BoD is signed off."""

    gate: str = Field(..., description="JOA gate at which the BoD was frozen, e.g. 'G2', 'G3'.")
    approved_by: str = Field(..., description="Name of the approver (e.g. 'Daniel Kearney').")
    approved_on: date = Field(..., description="Approval date (ISO 8601).")
    signature_ref: Optional[str] = Field(
        default=None,
        description="Pointer to the signed document (SharePoint / DocuSign envelope ID).",
    )


class BoDMetadata(_BoDModel):
    """Provenance and version metadata for the BoD document."""

    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="BoD schema version this document validates against.",
    )
    bod_version: str = Field(
        ...,
        description="Human-readable BoD revision, e.g. 'v0.2', 'RevA'.",
    )
    site_id: str = Field(
        ...,
        min_length=1,
        description="IATA-style short code for the site, e.g. 'BT1_2', 'SGT01'.",
    )
    title: str = Field(..., description="Full BoD title as it appears in SharePoint.")
    authors: List[str] = Field(default_factory=list, description="BoD authors.")
    created_on: datetime = Field(default_factory=datetime.utcnow)
    updated_on: Optional[datetime] = Field(default=None)
    approval: Optional[Approval] = Field(
        default=None,
        description="Non-null once the BoD is frozen at a JOA gate.",
    )
    notes: Optional[str] = Field(default=None, description="Free-text drafting notes.")


# =============================================================================
# 1. Site
# =============================================================================


class SiteBoD(_BoDModel):
    """Physical site parameters."""

    site_id: str = Field(..., description="Must match metadata.site_id.")
    country: str = Field(..., description="ISO 3166-1 alpha-2 country code, e.g. 'AU', 'SG'.")
    region: str = Field(..., description="Sub-national region, e.g. 'VIC', 'NSW'.")
    latitude_deg: float = Field(..., ge=-90.0, le=90.0)
    longitude_deg: float = Field(..., ge=-180.0, le=180.0)
    elevation_m: float = Field(default=0.0, ge=-500.0, le=6000.0)
    land_area_m2: float = Field(..., gt=0.0)
    it_hall_area_m2: float = Field(..., gt=0.0)
    design_life_years: int = Field(default=15, ge=1, le=50)
    seismic_zone: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def _check_area(self) -> "SiteBoD":
        if self.it_hall_area_m2 > self.land_area_m2:
            raise ValueError("it_hall_area_m2 cannot exceed land_area_m2")
        return self


# =============================================================================
# 2. Climate
# =============================================================================


class ClimateBoD(_BoDModel):
    """Climate design conditions (ASHRAE / on-site TMY)."""

    ashrae_class: str = Field(
        ...,
        description="ASHRAE thermal class, e.g. 'A1', 'A2', 'W40', 'H1'.",
    )
    design_dry_bulb_c: float = Field(..., description="Design dry-bulb temperature (°C, 0.4% n=20).")
    design_wet_bulb_c: float = Field(..., description="Design wet-bulb temperature (°C, 0.4% n=20).")
    extreme_dry_bulb_c: float = Field(..., description="Extreme dry-bulb (°C, worst 20-year).")
    ambient_temp_range_c: Tuple[float, float] = Field(
        ..., description="(min, max) annual ambient temperature (°C)."
    )
    relative_humidity_range_pct: Tuple[float, float] = Field(
        ..., description="(min, max) annual relative humidity (%)."
    )
    solar_ghi_kwh_m2_yr: Optional[float] = Field(default=None, ge=0.0)
    hdd_18_c: Optional[float] = Field(default=None, ge=0.0)
    cdd_18_c: Optional[float] = Field(default=None, ge=0.0)

    @field_validator("ambient_temp_range_c", "relative_humidity_range_pct")
    @classmethod
    def _check_range(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        lo, hi = v
        if lo > hi:
            raise ValueError(f"range must be (min, max); got ({lo}, {hi})")
        return v


# =============================================================================
# 3. Mechanical
# =============================================================================


class MechanicalBoD(_BoDModel):
    """Non-cooling mechanical (containment, floor loading, seismic bracing)."""

    hot_aisle_containment: bool = Field(default=True)
    cold_aisle_containment: bool = Field(default=False)
    raised_floor: bool = Field(default=False)
    floor_loading_kn_m2: float = Field(..., gt=0.0, description="Slab live load rating (kN/m²).")
    ceiling_height_m: float = Field(..., gt=0.0)
    fire_suppression: str = Field(default="pre_action_sprinkler")
    seismic_bracing: bool = Field(default=False)


# =============================================================================
# 4. Cooling
# =============================================================================


class CoolingLoopBoD(_BoDModel):
    """A single hydraulic loop (primary or secondary)."""

    name: str = Field(..., description="Loop label, e.g. 'TCS primary', 'FWS secondary'.")
    fluid: str = Field(default="PG25", description="Working fluid, e.g. 'water', 'PG25'.")
    supply_temp_c: float
    return_temp_c: float
    flow_lpm: float = Field(..., gt=0.0)
    design_pressure_kpa: float = Field(..., gt=0.0)

    @model_validator(mode="after")
    def _delta_t(self) -> "CoolingLoopBoD":
        if self.return_temp_c <= self.supply_temp_c:
            raise ValueError(
                f"{self.name}: return_temp_c must exceed supply_temp_c"
            )
        return self


class CoolingBoD(_BoDModel):
    """Cooling architecture: liquid + air + heat rejection."""

    architecture: CoolingArchitecture
    heat_rejection: HeatRejectionType
    design_ppue: float = Field(
        ..., gt=1.0, le=2.0, description="Design partial PUE (cooling only), e.g. 1.06."
    )
    design_pue: float = Field(..., gt=1.0, le=3.0)
    design_wue_l_per_kwh: float = Field(
        default=0.0, ge=0.0, description="Design water-usage-effectiveness (L / IT-kWh)."
    )
    inlet_temp_c: float = Field(..., description="Rack inlet coolant temperature (°C).")
    approach_temp_c: float = Field(
        default=5.0, ge=0.0, description="Approach temperature across heat rejection (°C)."
    )
    loops: List[CoolingLoopBoD] = Field(default_factory=list)
    n_plus_redundancy: str = Field(default="N+1", description="Cooling plant redundancy, e.g. 'N+1', '2N'.")


# =============================================================================
# 5. Electrical
# =============================================================================


class TransformerBoD(_BoDModel):
    role: str = Field(..., description="e.g. 'HV/MV incomer', 'MV/LV distribution'.")
    primary_kv: float = Field(..., gt=0.0)
    secondary_v: float = Field(..., gt=0.0)
    rating_mva: float = Field(..., gt=0.0)
    impedance_pct: float = Field(..., gt=0.0, le=30.0)
    vector_group: str = Field(default="Dyn11")


class UPSBoD(_BoDModel):
    topology: UPSTopology
    rating_kva: float = Field(..., gt=0.0)
    autonomy_minutes: float = Field(..., ge=0.0)
    redundancy: str = Field(default="N+1")


class GeneratorBoD(_BoDModel):
    fuel: str = Field(default="diesel", description="'diesel', 'hvo', 'gas', 'none'.")
    rating_kva: float = Field(..., ge=0.0)
    n_units: int = Field(..., ge=0)
    fuel_autonomy_hours: float = Field(..., ge=0.0)


class ElectricalBoD(_BoDModel):
    """MV/LV electrical topology and protection intent."""

    incoming_supply_kv: float = Field(..., gt=0.0)
    n_incomers: int = Field(..., ge=1, le=8)
    design_it_load_mw: float = Field(..., gt=0.0)
    design_facility_load_mw: float = Field(..., gt=0.0)
    transformers: List[TransformerBoD] = Field(default_factory=list)
    ups: UPSBoD
    generators: GeneratorBoD
    tier_rating: str = Field(default="Tier III", description="Uptime Institute or equivalent.")
    protection_philosophy_ref: Optional[str] = Field(
        default=None,
        description="Pointer to the protection philosophy document (SharePoint URL / doc ID).",
    )
    discrimination_targets: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Boundary → time margin (s). Keys use 'upstream>downstream' notation, "
            "e.g. 'MV_incomer>LV_incomer': 0.4."
        ),
    )
    earth_fault_max_current_a: float = Field(..., gt=0.0)
    arc_flash_max_incident_energy_cal_cm2: float = Field(..., gt=0.0)


# =============================================================================
# 6. Grid
# =============================================================================


class GridBoD(_BoDModel):
    """Grid interface and market participation intent."""

    region: GridRegionCode
    operator: str = Field(..., description="Grid operator name, e.g. 'SP PowerGrid', 'AEMO'.")
    poi_capacity_mw: float = Field(..., gt=0.0, description="Point-of-interconnection capacity (MW).")
    nominal_frequency_hz: float = Field(default=50.0)
    frequency_normal_band_hz: float = Field(default=0.2)
    voltage_kv_options: List[float] = Field(default_factory=list)
    power_factor_target: float = Field(default=0.98, gt=0.0, le=1.0)
    grid_emissions_kg_co2_per_kwh: float = Field(
        default=0.5, ge=0.0, le=2.0, description="Design-year marginal grid emissions factor."
    )
    demand_response_programs: List[str] = Field(default_factory=list)
    firming_bess_mwh: float = Field(default=0.0, ge=0.0)


# =============================================================================
# 7. Tariff
# =============================================================================


class TariffBoD(_BoDModel):
    """Electricity tariff assumptions.

    Only one of ``rates_ccy_per_kwh`` or ``ppa_ccy_per_mwh`` is required
    depending on ``structure``.
    """

    structure: TariffStructure
    currency: str = Field(..., description="ISO 4217 code, e.g. 'AUD', 'SGD', 'USD'.")
    rates_ccy_per_kwh: Dict[str, float] = Field(
        default_factory=dict,
        description="TOU rates by period, e.g. {'on_peak': 0.28, 'mid_peak': 0.18, 'off_peak': 0.09}.",
    )
    demand_charge_ccy_per_kw_month: float = Field(default=0.0, ge=0.0)
    fixed_charge_ccy_per_month: float = Field(default=0.0, ge=0.0)
    ppa_ccy_per_mwh: Optional[float] = Field(default=None, ge=0.0)
    ppa_term_years: Optional[int] = Field(default=None, ge=1, le=30)
    escalation_pct_yr: float = Field(default=2.5, ge=-10.0, le=20.0)

    @model_validator(mode="after")
    def _check_structure_matches(self) -> "TariffBoD":
        if self.structure == TariffStructure.NEGOTIATED_PPA and self.ppa_ccy_per_mwh is None:
            raise ValueError("negotiated_ppa requires ppa_ccy_per_mwh")
        if self.structure in {TariffStructure.TOU, TariffStructure.FLAT} and not self.rates_ccy_per_kwh:
            raise ValueError(f"{self.structure} requires rates_ccy_per_kwh")
        return self


# =============================================================================
# 8. Network
# =============================================================================


class NetworkBoD(_BoDModel):
    """Rack-to-fabric and DC-to-DC networking intent."""

    fabric: str = Field(..., description="e.g. 'NVIDIA Quantum-2 IB', 'Spectrum-X Ethernet'.")
    rail_topology: str = Field(default="rail_optimized_fat_tree")
    intra_rack_gbps: float = Field(default=800.0, gt=0.0)
    inter_rack_gbps: float = Field(default=800.0, gt=0.0)
    dc_egress_gbps: float = Field(..., gt=0.0)
    peering_providers: List[str] = Field(default_factory=list)
    latency_target_ms_intra_region: float = Field(default=2.0, gt=0.0)


# =============================================================================
# 9. NVIDIA platform
# =============================================================================


class NVIDIAPlatformBoD(_BoDModel):
    """GPU platform selection and deployment shape."""

    platform: NVIDIAPlatformCode
    num_racks: int = Field(..., gt=0)
    rack_power_kw: Optional[float] = Field(
        default=None,
        gt=0.0,
        description=(
            "Optional override of the platform's default rack power. "
            "If None, hydrate() falls back to PLATFORM_CONFIG."
        ),
    )
    power_mode: str = Field(
        default="max_p",
        description="Vera Rubin power mode ('max_p' | 'max_q'). Ignored for other platforms.",
    )
    coolant_inlet_temp_c: float = Field(default=35.0)
    reference_workload: str = Field(
        default="mixed_training_inference",
        description="Reference workload for RoI ('training_only', 'inference_only', 'mixed_training_inference').",
    )
    utilization_pct: float = Field(
        default=75.0, gt=0.0, le=100.0, description="Assumed annual GPU utilisation (%)."
    )


# =============================================================================
# Economics (financial assumptions for RoI)
# =============================================================================


class EconomicsBoD(_BoDModel):
    """Financial assumptions used by the RoI pack.

    Kept separate from ``TariffBoD`` because tariffs are physics-adjacent
    (they price kWh) while economics are portfolio-level (capex, discount).
    """

    currency: str = Field(..., description="Base currency for RoI outputs.")
    capex_usd_per_kw_it: float = Field(
        ..., gt=0.0, description="Fully-loaded capex per kW of IT capacity."
    )
    opex_fixed_usd_per_kw_it_yr: float = Field(default=60.0, ge=0.0)
    revenue_usd_per_gpu_hour: float = Field(
        ..., ge=0.0, description="Blended revenue per GPU-hour."
    )
    discount_rate_pct: float = Field(default=8.0, ge=0.0, le=30.0)
    horizon_years: int = Field(default=10, ge=1, le=30)
    tax_rate_pct: float = Field(default=25.0, ge=0.0, le=60.0)
    depreciation_years: int = Field(default=7, ge=1, le=30)


# =============================================================================
# Optimizer directives
# =============================================================================


class OptimizerDirectivesBoD(_BoDModel):
    """Which BoD fields the optimizer may vary, and its objectives.

    Free variables are named as dotted paths from the BoD root, e.g.
    ``'nvidia_platform.num_racks'`` or ``'cooling.inlet_temp_c'``.  The
    optimizer refuses to touch anything not listed here.
    """

    objectives: List[str] = Field(
        default_factory=lambda: ["maximize_npv", "minimize_energy"],
        description=(
            "Ordered objectives from: 'maximize_npv', 'minimize_energy', "
            "'minimize_lcoe', 'maximize_pflops_per_mw'."
        ),
    )
    free_variables: Dict[str, Tuple[float, float]] = Field(
        default_factory=dict,
        description="Dotted path → (min, max) allowed range for that variable.",
    )
    fixed_constraints: Dict[str, float] = Field(
        default_factory=dict,
        description="Hard equality constraints, e.g. {'grid.poi_capacity_mw': 50.0}.",
    )


# =============================================================================
# Top-level document
# =============================================================================


class BasisOfDesign(_BoDModel):
    """The canonical BoD document.

    Nine domains + metadata + economics + optimizer directives.  This is what
    the loader validates and what the digital twin ingests.
    """

    metadata: BoDMetadata
    site: SiteBoD
    climate: ClimateBoD
    mechanical: MechanicalBoD
    cooling: CoolingBoD
    electrical: ElectricalBoD
    grid: GridBoD
    tariff: TariffBoD
    network: NetworkBoD
    nvidia_platform: NVIDIAPlatformBoD
    economics: EconomicsBoD
    optimizer: OptimizerDirectivesBoD = Field(default_factory=OptimizerDirectivesBoD)
    high_fidelity: Optional["HighFidelityBoD"] = Field(
        default=None,
        description=(
            "Optional Ansys / ETAP high-fidelity solver hooks. Absent means "
            "the optimizer runs purely on the analytic Digital Twin."
        ),
    )

    # ------------------------------------------------------------------
    # Cross-domain integrity checks
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _cross_domain(self) -> "BasisOfDesign":
        # 1. site_id must match between metadata and site
        if self.metadata.site_id != self.site.site_id:
            raise ValueError(
                f"metadata.site_id ({self.metadata.site_id!r}) must equal "
                f"site.site_id ({self.site.site_id!r})"
            )

        # 2. IT load must be within POI capacity
        if self.electrical.design_facility_load_mw > self.grid.poi_capacity_mw:
            raise ValueError(
                f"design_facility_load_mw ({self.electrical.design_facility_load_mw}) "
                f"exceeds grid.poi_capacity_mw ({self.grid.poi_capacity_mw})"
            )

        # 3. PUE must be internally consistent (design_pue >= design_ppue)
        if self.cooling.design_pue < self.cooling.design_ppue:
            raise ValueError("cooling.design_pue must be >= cooling.design_ppue")

        # 4. Tariff currency should match economics currency for a clean RoI
        if self.tariff.currency != self.economics.currency:
            raise ValueError(
                f"tariff.currency ({self.tariff.currency}) must match "
                f"economics.currency ({self.economics.currency}); mix currencies via FX in a wrapper."
            )
        return self

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def is_frozen(self) -> bool:
        """A BoD is frozen once ``metadata.approval`` is populated."""
        return self.metadata.approval is not None


# Resolve forward reference to HighFidelityBoD (defined in bod.high_fidelity).
from firmus_ai_factory.bod.high_fidelity import HighFidelityBoD  # noqa: E402

BasisOfDesign.model_rebuild()
