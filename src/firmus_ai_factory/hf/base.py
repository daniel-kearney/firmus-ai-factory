"""Base types shared by every high-fidelity adapter.

Correction records are deliberately narrow. HF solvers return floats and
booleans that the optimizer can fold into the analytic result - not
mesh-level state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class HFStatus(str, Enum):
    """Outcome of an HF call, always reported alongside the correction."""

    OK = "ok"
    FALLBACK = "fallback"        # backend unreachable, analytic result used
    DISABLED = "disabled"         # adapter present but backend='disabled'
    ABSENT = "absent"             # BoD has no adapter for this domain
    REJECTED = "rejected"         # fail_open=false and backend blew up


class HFError(RuntimeError):
    """Raised when fail_open=false and a solver call fails."""


# ---------------------------------------------------------------------------
# Correction records
# ---------------------------------------------------------------------------


@dataclass
class ThermalCorrection:
    """Ansys / CFD corrections folded back into the analytic model.

    All fields are relative or absolute *deltas* — the optimizer applies
    them on top of the analytic report:

        pue_bump             : additive PUE correction (dimensionless)
        rack_hotspot_c       : worst rack outlet air / coolant temperature
        rack_hotspot_margin_c: margin vs. NVIDIA limit (positive = safe)
        cdu_delta_t_c        : simulated CDU ΔT for cross-check
        hotspot_violation    : True if any rack breaches the NVIDIA envelope
    """

    pue_bump: float = 0.0
    rack_hotspot_c: Optional[float] = None
    rack_hotspot_margin_c: Optional[float] = None
    cdu_delta_t_c: Optional[float] = None
    hotspot_violation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ElectricalCorrection:
    """ETAP corrections folded back into the analytic model.

        losses_pct             : distribution + transformer losses (%)
        worst_arc_flash_cal_cm2: worst-case incident energy across LV boards
        arc_flash_violation    : True if worst_arc_flash exceeds BoD budget
        discrimination_ok      : True if all boundary time margins met
        discrimination_violations: list of 'upstream>downstream' offenders
        short_circuit_ka       : max symmetrical fault current on LV bus
    """

    losses_pct: float = 0.0
    worst_arc_flash_cal_cm2: Optional[float] = None
    arc_flash_violation: bool = False
    discrimination_ok: bool = True
    discrimination_violations: list = field(default_factory=list)
    short_circuit_ka: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HFResult:
    """Wrapper carrying status + correction + provenance."""

    status: HFStatus
    correction: Any                # ThermalCorrection | ElectricalCorrection | None
    reason: str = ""
    solver: str = ""
    duration_s: float = 0.0
    cache_hit: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "correction": self.correction.to_dict() if self.correction is not None else None,
            "reason": self.reason,
            "solver": self.solver,
            "duration_s": self.duration_s,
            "cache_hit": self.cache_hit,
        }


# ---------------------------------------------------------------------------
# Content-addressed cache
# ---------------------------------------------------------------------------


def hash_bod_subset(payload: Dict[str, Any]) -> str:
    """Stable SHA1 over a JSON-serialisable BoD subset.

    Used to key HF results so a 200-candidate sweep only pays for each
    unique geometry / topology once.
    """
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


class HFCache:
    """Tiny in-memory + on-disk cache for HF results.

    On-disk path is opt-in: pass a directory and results are also written
    as JSON so runs across sessions can reuse them. The on-disk format is
    intentionally trivial (one file per hash) to make CI reproducibility
    easy.
    """

    def __init__(self, directory: Optional[Path] = None) -> None:
        self._mem: Dict[str, HFResult] = {}
        self._dir = Path(directory) if directory else None
        if self._dir is not None:
            self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[HFResult]:
        if key in self._mem:
            r = self._mem[key]
            return HFResult(**{**r.__dict__, "cache_hit": True})
        if self._dir is not None:
            p = self._dir / f"{key}.json"
            if p.exists():
                try:
                    data = json.loads(p.read_text())
                    r = _rehydrate_result(data)
                    r.cache_hit = True
                    self._mem[key] = r
                    return r
                except (json.JSONDecodeError, KeyError):
                    return None
        return None

    def put(self, key: str, result: HFResult) -> None:
        self._mem[key] = result
        if self._dir is not None:
            p = self._dir / f"{key}.json"
            p.write_text(json.dumps(result.to_dict(), indent=2))


def _rehydrate_result(data: Dict[str, Any]) -> HFResult:
    """Reconstruct an HFResult from its ``to_dict`` form (best-effort)."""
    corr_data = data.get("correction")
    correction: Any = None
    if isinstance(corr_data, dict):
        if "pue_bump" in corr_data:
            correction = ThermalCorrection(**corr_data)
        elif "losses_pct" in corr_data:
            correction = ElectricalCorrection(**corr_data)
    return HFResult(
        status=HFStatus(data["status"]),
        correction=correction,
        reason=data.get("reason", ""),
        solver=data.get("solver", ""),
        duration_s=data.get("duration_s", 0.0),
    )


# Module-level default caches (adapters may override with their own).
_DEFAULT_THERMAL_CACHE = HFCache()
_DEFAULT_ELECTRICAL_CACHE = HFCache()


def get_default_thermal_cache() -> HFCache:
    return _DEFAULT_THERMAL_CACHE


def get_default_electrical_cache() -> HFCache:
    return _DEFAULT_ELECTRICAL_CACHE


def reset_default_caches() -> None:
    """Test helper: wipe both module-level caches."""
    _DEFAULT_THERMAL_CACHE._mem.clear()
    _DEFAULT_ELECTRICAL_CACHE._mem.clear()
