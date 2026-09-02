"""AI Factory Optimizer.

Takes a BoD, sweeps the free variables the BoD declares as tunable, and
returns the winning configuration plus a full RoI / energy / sensitivity
pack.  This is the piece the Model-to-Grid diagram calls "optimiser".

Design
------

The optimizer is deliberately model-light and physics-aware:

    * It re-uses ``FirmusAIFactory.generate_full_report()`` for every
      candidate, so results stay consistent with the rest of the digital
      twin.
    * It uses a coarse grid-plus-random-refine search rather than a
      gradient method.  BoD free variables are typically integer racks,
      inlet temperature bands or a small set of platform choices, which
      makes a swept search cheaper and more transparent than MPC.
    * All outputs are pure Python dicts / dataclasses so the pack is easy
      to persist to JSON, feed the dashboards, and diff between runs.

Outputs (as requested)
----------------------

    * ``optimal_bod``      - a new ``BasisOfDesign`` with the winning values
    * ``roi``              - NPV, payback, LCOE, $/token
    * ``energy``           - annual MWh, PUE, WUE, tCO2
    * ``sensitivity``      - tornado of top BoD levers by RoI impact
"""

from __future__ import annotations

import copy
import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from firmus_ai_factory.bod.hydrate import hydrate_factory
from firmus_ai_factory.bod.schema import BasisOfDesign
from firmus_ai_factory.hf import (
    HFResult,
    HFStatus,
    run_electrical,
    run_thermal,
)


# ---------------------------------------------------------------------------
# Helpers to walk dotted paths through the BoD (Pydantic v2)
# ---------------------------------------------------------------------------


def _get_by_path(bod: BasisOfDesign, path: str) -> Any:
    obj: Any = bod
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _set_by_path(bod_dict: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = bod_dict
    for p in parts[:-1]:
        cursor = cursor[p]
    cursor[parts[-1]] = value


def _mutate_bod(bod: BasisOfDesign, overrides: Dict[str, Any]) -> BasisOfDesign:
    """Return a new BoD with the given dotted-path overrides applied.

    We round-trip through the schema so every mutation is re-validated.
    """
    payload = bod.model_dump(mode="json")
    for path, value in overrides.items():
        _set_by_path(payload, path, value)
    return BasisOfDesign.model_validate(payload)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class RoIPack:
    currency: str
    horizon_years: int
    capex_total: float
    annual_revenue: float
    annual_energy_cost: float
    annual_opex_fixed: float
    annual_ebitda: float
    npv: float
    payback_years: float
    lcoe_ccy_per_mwh: float
    revenue_per_gpu_hour: float
    cost_per_token_usd_micro: float  # $/million tokens, illustrative

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class EnergyPack:
    it_load_mw: float
    facility_load_mw: float
    pue: float
    wue_l_per_kwh: float
    annual_energy_mwh: float
    annual_it_energy_mwh: float
    annual_water_m3: float
    grid_emissions_kg_co2_per_kwh: float
    annual_tco2e: float

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class HFPack:
    """Provenance for the high-fidelity contribution to a candidate score.

    Present on every OptimizerResult (empty if neither adapter is wired).
    """

    thermal_status: str = HFStatus.ABSENT.value
    electrical_status: str = HFStatus.ABSENT.value
    thermal_correction: Optional[Dict[str, Any]] = None
    electrical_correction: Optional[Dict[str, Any]] = None
    thermal_reason: str = ""
    electrical_reason: str = ""
    hotspot_violation: bool = False
    arc_flash_violation: bool = False
    discrimination_ok: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class SensitivityEntry:
    variable: str
    low_value: float
    high_value: float
    npv_low: float
    npv_high: float
    swing: float  # abs difference in NPV between low and high


@dataclass
class OptimizerResult:
    optimal_bod: BasisOfDesign
    roi: RoIPack
    energy: EnergyPack
    sensitivity: List[SensitivityEntry] = field(default_factory=list)
    candidates_evaluated: int = 0
    objective_score: float = 0.0
    hf: HFPack = field(default_factory=HFPack)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "optimal_bod": self.optimal_bod.model_dump(mode="json"),
            "roi": self.roi.to_dict(),
            "energy": self.energy.to_dict(),
            "sensitivity": [s.__dict__ for s in self.sensitivity],
            "candidates_evaluated": self.candidates_evaluated,
            "objective_score": self.objective_score,
            "hf": self.hf.to_dict(),
        }


# ---------------------------------------------------------------------------
# Core evaluation: BoD -> (RoI, Energy)
# ---------------------------------------------------------------------------


HOURS_PER_YEAR = 8760.0


def _evaluate_bod(
    bod: BasisOfDesign,
    *,
    use_hf: bool = True,
) -> Tuple[RoIPack, EnergyPack, "HFPack"]:
    """Score a candidate BoD by hydrating it and reading physics + economics.

    When ``use_hf`` is true and the BoD declares Ansys / ETAP hooks, the
    solvers are called and their corrections are folded into PUE, energy,
    and losses before RoI is computed. Fallback is graceful when
    ``bod.high_fidelity.fail_open`` is true (the default).
    """
    factory = hydrate_factory(bod)
    report = factory.generate_full_report()

    it_kw = report["power"]["it_power_kw"]
    facility_kw = report["power"]["total_facility_power_kw"]
    pue = report["power"]["pue"]

    # --- High-fidelity corrections -----------------------------------------
    hf_pack = HFPack()
    if use_hf and bod.high_fidelity is not None and bod.high_fidelity.is_active:
        fail_open = bod.high_fidelity.fail_open
        t_res = run_thermal(bod, fail_open=fail_open)
        e_res = run_electrical(bod, fail_open=fail_open)

        hf_pack.thermal_status = t_res.status.value
        hf_pack.electrical_status = e_res.status.value
        hf_pack.thermal_reason = t_res.reason
        hf_pack.electrical_reason = e_res.reason

        if t_res.correction is not None:
            hf_pack.thermal_correction = t_res.correction.to_dict()
            pue = pue + t_res.correction.pue_bump
            # Facility load scales with the corrected PUE.
            facility_kw = it_kw * pue
            hf_pack.hotspot_violation = t_res.correction.hotspot_violation

        if e_res.correction is not None:
            hf_pack.electrical_correction = e_res.correction.to_dict()
            # Electrical losses add to facility draw (fraction of IT).
            facility_kw = facility_kw * (1.0 + e_res.correction.losses_pct / 100.0)
            hf_pack.arc_flash_violation = e_res.correction.arc_flash_violation
            hf_pack.discrimination_ok = e_res.correction.discrimination_ok

    utilization = bod.nvidia_platform.utilization_pct / 100.0
    hours = HOURS_PER_YEAR * utilization

    annual_it_mwh = it_kw * hours / 1000.0
    annual_facility_mwh = facility_kw * hours / 1000.0

    # Water
    wue = bod.cooling.design_wue_l_per_kwh
    annual_water_l = wue * annual_facility_mwh * 1000.0
    annual_water_m3 = annual_water_l / 1000.0

    # Emissions
    ef = bod.grid.grid_emissions_kg_co2_per_kwh
    annual_tco2e = annual_facility_mwh * ef  # MWh * (kg/kWh) = t

    # Tariff blended rate: for TOU take the simple mean, for PPA use flat.
    tariff = bod.tariff
    if tariff.rates_ccy_per_kwh:
        avg_rate = sum(tariff.rates_ccy_per_kwh.values()) / len(tariff.rates_ccy_per_kwh)
    elif tariff.ppa_ccy_per_mwh is not None:
        avg_rate = tariff.ppa_ccy_per_mwh / 1000.0
    else:
        avg_rate = 0.10  # defensive fallback

    annual_energy_cost = annual_facility_mwh * 1000.0 * avg_rate
    annual_energy_cost += (
        facility_kw * tariff.demand_charge_ccy_per_kw_month * 12.0
    )
    annual_energy_cost += tariff.fixed_charge_ccy_per_month * 12.0

    # Economics
    econ = bod.economics
    total_gpus = report["compute"]["total_gpus"]
    annual_gpu_hours = total_gpus * hours
    annual_revenue = annual_gpu_hours * econ.revenue_usd_per_gpu_hour
    annual_opex_fixed = econ.opex_fixed_usd_per_kw_it_yr * it_kw

    capex_total = econ.capex_usd_per_kw_it * it_kw
    ebitda = annual_revenue - annual_energy_cost - annual_opex_fixed

    r = econ.discount_rate_pct / 100.0
    n = econ.horizon_years
    # NPV of a level annuity minus capex
    if r > 0:
        annuity_factor = (1 - (1 + r) ** -n) / r
    else:
        annuity_factor = float(n)
    npv = -capex_total + ebitda * annuity_factor

    payback = capex_total / ebitda if ebitda > 0 else float("inf")

    # LCOE = discounted lifetime cost / discounted lifetime energy
    lifetime_cost = capex_total + (annual_energy_cost + annual_opex_fixed) * annuity_factor
    lifetime_energy_mwh = annual_facility_mwh * annuity_factor
    lcoe = lifetime_cost / lifetime_energy_mwh if lifetime_energy_mwh > 0 else float("inf")

    # $ per million tokens - illustrative constant tokens/GPU-hour by platform tier.
    tokens_per_gpu_hour = {
        "hgx_h100": 1.8e6,
        "hgx_h200": 2.4e6,
        "gb300_nvl72": 5.5e6,
        "vr_nvl72_max_p": 7.2e6,
        "vr_nvl72_max_q": 6.4e6,
    }.get(bod.nvidia_platform.platform, 2.0e6)
    annual_tokens = tokens_per_gpu_hour * annual_gpu_hours
    cost_per_mtoken = (annual_energy_cost + annual_opex_fixed) / (annual_tokens / 1e6)

    roi = RoIPack(
        currency=econ.currency,
        horizon_years=n,
        capex_total=capex_total,
        annual_revenue=annual_revenue,
        annual_energy_cost=annual_energy_cost,
        annual_opex_fixed=annual_opex_fixed,
        annual_ebitda=ebitda,
        npv=npv,
        payback_years=payback,
        lcoe_ccy_per_mwh=lcoe,
        revenue_per_gpu_hour=econ.revenue_usd_per_gpu_hour,
        cost_per_token_usd_micro=cost_per_mtoken,
    )

    energy = EnergyPack(
        it_load_mw=it_kw / 1000.0,
        facility_load_mw=facility_kw / 1000.0,
        pue=pue,
        wue_l_per_kwh=wue,
        annual_energy_mwh=annual_facility_mwh,
        annual_it_energy_mwh=annual_it_mwh,
        annual_water_m3=annual_water_m3,
        grid_emissions_kg_co2_per_kwh=ef,
        annual_tco2e=annual_tco2e,
    )
    return roi, energy, hf_pack


# ---------------------------------------------------------------------------
# Objective scoring
# ---------------------------------------------------------------------------


def _score(objectives: List[str], roi: RoIPack, energy: EnergyPack) -> float:
    """Blend multiple objectives into a single scalar to rank candidates.

    Higher = better.  For minimisation objectives we negate; for
    maximisation we take the raw value.  Objectives are equally weighted
    - the BoD ``optimizer.objectives`` field expresses priority through
    ordering, not weight, on purpose (keeps intent legible).
    """
    parts: List[float] = []
    for obj in objectives:
        if obj == "maximize_npv":
            parts.append(roi.npv)
        elif obj == "minimize_energy":
            parts.append(-energy.annual_energy_mwh)
        elif obj == "minimize_lcoe":
            parts.append(-roi.lcoe_ccy_per_mwh)
        elif obj == "maximize_pflops_per_mw":
            # Higher IT density = higher score
            parts.append(1.0 / max(energy.facility_load_mw, 1e-6))
        else:
            raise ValueError(f"Unknown objective {obj!r}")

    # Simple normalised sum (equal weight after magnitude normalisation).
    if not parts:
        return 0.0
    max_abs = max(abs(p) for p in parts) or 1.0
    return sum(p / max_abs for p in parts)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def _sample_points(lo: float, hi: float, n: int, integer: bool) -> List[float]:
    if n <= 1 or hi <= lo:
        return [lo]
    step = (hi - lo) / (n - 1)
    pts = [lo + i * step for i in range(n)]
    if integer:
        pts = sorted(set(int(round(p)) for p in pts))
    return pts


def _looks_integer(path: str) -> bool:
    return path.endswith("num_racks") or path.endswith("_years") or path.endswith("n_incomers")


def _generate_candidates(
    bod: BasisOfDesign, samples_per_var: int = 5, max_candidates: int = 400
) -> List[Dict[str, Any]]:
    """Cartesian product of samples across all free variables, capped."""
    directives = bod.optimizer
    if not directives.free_variables:
        return [{}]  # single candidate = the BoD as-is

    axes: List[Tuple[str, List[float]]] = []
    for path, (lo, hi) in directives.free_variables.items():
        integer = _looks_integer(path)
        axes.append((path, _sample_points(lo, hi, samples_per_var, integer)))

    grid = list(itertools.product(*[pts for _, pts in axes]))
    random.Random(42).shuffle(grid)
    if len(grid) > max_candidates:
        grid = grid[:max_candidates]

    return [{path: val for (path, _), val in zip(axes, combo)} for combo in grid]


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def optimize(
    bod: BasisOfDesign,
    *,
    samples_per_var: int = 5,
    max_candidates: int = 400,
    with_sensitivity: bool = True,
    progress: Optional[Callable[[int, int], None]] = None,
) -> OptimizerResult:
    """Run the AI-factory optimizer over the BoD's declared free variables.

    Parameters
    ----------
    bod
        Input BoD document.  Must not be frozen (``metadata.approval is None``)
        unless the caller is deliberately re-scoring a signed design; the
        optimizer will still run but a warning is left in the result payload
        via ``optimal_bod.metadata.notes``.
    samples_per_var
        Grid resolution per free variable before the max-candidates cap.
    max_candidates
        Hard cap on candidates evaluated (protects long runs).
    with_sensitivity
        If True, also compute the one-variable-at-a-time tornado around the
        winning point.
    progress
        Optional callback ``fn(i, total)`` for progress reporting.
    """

    if bod.is_frozen:
        # Non-fatal: emit a note into the winning BoD when we write it back.
        pass

    candidates = _generate_candidates(bod, samples_per_var, max_candidates)

    best: Optional[Tuple[float, BasisOfDesign, RoIPack, EnergyPack, HFPack]] = None
    evaluated = 0
    for i, overrides in enumerate(candidates):
        try:
            candidate = _mutate_bod(bod, overrides) if overrides else bod
            roi, energy, hf_pack = _evaluate_bod(candidate)
        except (ValueError, NotImplementedError):
            # Invalid combination: skip, but still count so progress is honest.
            continue
        evaluated += 1
        # Hard constraints from HF: a candidate that breaches hotspot or
        # arc-flash envelopes is disqualified so it cannot win the sweep.
        if hf_pack.hotspot_violation or hf_pack.arc_flash_violation or not hf_pack.discrimination_ok:
            continue
        score = _score(bod.optimizer.objectives, roi, energy)
        if best is None or score > best[0]:
            best = (score, candidate, roi, energy, hf_pack)
        if progress:
            progress(i + 1, len(candidates))

    if best is None:
        raise RuntimeError("Optimizer evaluated 0 valid candidates - check free_variables ranges.")

    score, winner_bod, roi, energy, winner_hf = best

    sensitivity: List[SensitivityEntry] = []
    if with_sensitivity and bod.optimizer.free_variables:
        sensitivity = _tornado(winner_bod)

    if bod.is_frozen:
        winner_bod = _annotate_frozen(winner_bod)

    return OptimizerResult(
        optimal_bod=winner_bod,
        roi=roi,
        energy=energy,
        sensitivity=sensitivity,
        hf=winner_hf,
        candidates_evaluated=evaluated,
        objective_score=score,
    )


def _annotate_frozen(bod: BasisOfDesign) -> BasisOfDesign:
    payload = bod.model_dump(mode="json")
    note = "Optimizer re-scored a frozen BoD; approval block preserved but result is advisory only."
    existing = payload["metadata"].get("notes") or ""
    payload["metadata"]["notes"] = (existing + "\n" + note).strip()
    return BasisOfDesign.model_validate(payload)


def _tornado(winner: BasisOfDesign, *, use_hf: bool = True) -> List[SensitivityEntry]:
    """One-at-a-time NPV sensitivity around the winning BoD."""
    entries: List[SensitivityEntry] = []
    base_roi, _, _ = _evaluate_bod(winner, use_hf=use_hf)
    for path, (lo, hi) in winner.optimizer.free_variables.items():
        integer = _looks_integer(path)
        lo_val: Any = int(round(lo)) if integer else lo
        hi_val: Any = int(round(hi)) if integer else hi
        try:
            bod_lo = _mutate_bod(winner, {path: lo_val})
            bod_hi = _mutate_bod(winner, {path: hi_val})
            roi_lo, _, _ = _evaluate_bod(bod_lo, use_hf=use_hf)
            roi_hi, _, _ = _evaluate_bod(bod_hi, use_hf=use_hf)
        except (ValueError, NotImplementedError):
            continue
        entries.append(
            SensitivityEntry(
                variable=path,
                low_value=float(lo_val),
                high_value=float(hi_val),
                npv_low=roi_lo.npv,
                npv_high=roi_hi.npv,
                swing=abs(roi_hi.npv - roi_lo.npv),
            )
        )
    entries.sort(key=lambda e: e.swing, reverse=True)
    return entries
