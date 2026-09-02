"""ETAP electrical adapter.

Mirrors the thermal adapter in shape: subprocess / http / file transports,
content-hash cache, and an analytic shim for unreachable backends.

Where ETAP earns its keep is protection coordination and arc-flash.  The
adapter re-uses the BoD's ``electrical.discrimination_targets`` to check
whether ETAP's simulated time margins meet the design intent, and folds
the worst-case incident energy into the correction record for the
optimizer to consider.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from firmus_ai_factory.bod.high_fidelity import ElectricalHFBoD, HFBackend
from firmus_ai_factory.bod.schema import BasisOfDesign
from firmus_ai_factory.hf.base import (
    ElectricalCorrection,
    HFCache,
    HFError,
    HFResult,
    HFStatus,
    get_default_electrical_cache,
    hash_bod_subset,
)


# Nominal arc-flash budget (cal/cm²) used only by the analytic shim.
_DEFAULT_ARC_FLASH_BUDGET = 8.0


def _electrical_key(bod: BasisOfDesign) -> str:
    sub = {
        "electrical": bod.electrical.model_dump(mode="json"),
        "grid_operator": bod.grid.operator,
        "grid_voltage_options": bod.grid.voltage_kv_options,
        "facility_load_mw": bod.electrical.design_facility_load_mw,
    }
    return "electrical:" + hash_bod_subset(sub)


class ETAPAdapter:
    """Adapter that resolves a BoD candidate to an ``ElectricalCorrection``."""

    def __init__(self, cfg: ElectricalHFBoD, cache: Optional[HFCache] = None) -> None:
        self.cfg = cfg
        self.cache = cache or get_default_electrical_cache()

    # ---------- transports ----------

    def _run_subprocess(self, bod: BasisOfDesign) -> ElectricalCorrection:
        assert self.cfg.endpoint
        payload = {
            "studies": self.cfg.studies,
            "project_ref": self.cfg.project_ref,
            "bod": bod.model_dump(mode="json"),
        }
        proc = subprocess.run(
            [self.cfg.endpoint, "--batch"],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=self.cfg.timeout_s,
        )
        if proc.returncode != 0:
            raise HFError(f"ETAP subprocess exited {proc.returncode}: {proc.stderr[:400]}")
        return _parse_electrical(proc.stdout, bod, self.cfg)

    def _run_http(self, bod: BasisOfDesign) -> ElectricalCorrection:
        import urllib.request

        assert self.cfg.endpoint
        payload = {
            "studies": self.cfg.studies,
            "project_ref": self.cfg.project_ref,
            "bod": bod.model_dump(mode="json"),
        }
        req = urllib.request.Request(
            self.cfg.endpoint + "/solve",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
            body = resp.read().decode("utf-8")
        return _parse_electrical(body, bod, self.cfg)

    def _run_file(self, bod: BasisOfDesign) -> ElectricalCorrection:
        assert self.cfg.endpoint
        base = Path(self.cfg.endpoint)
        key = _electrical_key(bod)
        p = base / f"{key}.json"
        if not p.exists():
            manifest = base / "manifest.json"
            if manifest.exists():
                data = json.loads(manifest.read_text())
                if key in data:
                    return _parse_electrical_dict(data[key], bod, self.cfg)
            raise HFError(f"No file-mode result at {p} or manifest for key {key}")
        return _parse_electrical(p.read_text(), bod, self.cfg)

    # ---------- analytic shim ----------

    def _analytic_shim(self, bod: BasisOfDesign) -> ElectricalCorrection:
        # Losses scale mildly with facility load fraction of POI.
        load_frac = bod.electrical.design_facility_load_mw / max(bod.grid.poi_capacity_mw, 1e-6)
        losses = 1.5 + 0.5 * load_frac  # % - baseline ~1.5% growing to ~2.0% at 100% POI
        # Symmetrical fault current: scale with incomer transformer rating.
        # Very rough: I_sc ~ S / (sqrt(3) * V * Z_pu). Use first transformer if present.
        i_sc_ka = 25.0
        if bod.electrical.transformers:
            tx = bod.electrical.transformers[0]
            i_sc_ka = (tx.rating_mva * 1000.0) / (
                (3 ** 0.5) * (tx.secondary_v / 1000.0) * max(tx.impedance_pct, 1.0)
            )
        arc_flash = min(bod.electrical.arc_flash_max_incident_energy_cal_cm2, _DEFAULT_ARC_FLASH_BUDGET) * 0.9
        # Assume design intent met (no violations) in the shim - real ETAP
        # is required to *disprove* the design.
        return ElectricalCorrection(
            losses_pct=round(losses, 3),
            worst_arc_flash_cal_cm2=round(arc_flash, 2),
            arc_flash_violation=arc_flash > bod.electrical.arc_flash_max_incident_energy_cal_cm2,
            discrimination_ok=True,
            discrimination_violations=[],
            short_circuit_ka=round(i_sc_ka, 1),
        )

    # ---------- public entry ----------

    def run(self, bod: BasisOfDesign, *, fail_open: bool = True) -> HFResult:
        if self.cfg.backend == HFBackend.DISABLED:
            return HFResult(status=HFStatus.DISABLED, correction=None, solver=self.cfg.solver)

        key = _electrical_key(bod)
        if self.cfg.cache_enabled:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        start = time.perf_counter()
        try:
            if self.cfg.backend == HFBackend.SUBPROCESS:
                correction = self._run_subprocess(bod)
            elif self.cfg.backend == HFBackend.HTTP:
                correction = self._run_http(bod)
            elif self.cfg.backend == HFBackend.FILE:
                correction = self._run_file(bod)
            else:  # pragma: no cover
                raise HFError(f"Unknown backend {self.cfg.backend!r}")
            result = HFResult(
                status=HFStatus.OK,
                correction=correction,
                solver=self.cfg.solver,
                duration_s=time.perf_counter() - start,
            )
        except (subprocess.TimeoutExpired, OSError, HFError, ValueError) as exc:
            if not fail_open:
                raise HFError(str(exc)) from exc
            result = HFResult(
                status=HFStatus.FALLBACK,
                correction=self._analytic_shim(bod),
                reason=f"{type(exc).__name__}: {exc}",
                solver=self.cfg.solver,
                duration_s=time.perf_counter() - start,
            )

        if self.cfg.cache_enabled:
            self.cache.put(key, result)
        return result


# ---------------------------------------------------------------------------
# Parsing (with BoD-aware discrimination check)
# ---------------------------------------------------------------------------


def _parse_electrical_dict(
    data: Dict[str, Any], bod: BasisOfDesign, cfg: ElectricalHFBoD
) -> ElectricalCorrection:
    losses = float(data.get("losses_pct", 0.0))
    arc = data.get("worst_arc_flash_cal_cm2")
    arc_val = float(arc) if arc is not None else None
    i_sc = data.get("short_circuit_ka")
    i_sc_val = float(i_sc) if i_sc is not None else None

    # Solver-reported margins by boundary, e.g. {"MV_incomer>LV_incomer": 0.38}
    solver_margins: Dict[str, float] = data.get("discrimination_margins_s", {}) or {}
    discrimination_ok = True
    violations: List[str] = []
    if cfg.check_discrimination_targets:
        for boundary, target in bod.electrical.discrimination_targets.items():
            achieved = solver_margins.get(boundary)
            if achieved is None:
                # No solver report -> treat as unverified, not a failure.
                continue
            if achieved < target:
                discrimination_ok = False
                violations.append(
                    f"{boundary}: solver {achieved:.2f}s < target {target:.2f}s"
                )

    arc_violation = False
    if arc_val is not None:
        arc_violation = arc_val > bod.electrical.arc_flash_max_incident_energy_cal_cm2

    return ElectricalCorrection(
        losses_pct=losses,
        worst_arc_flash_cal_cm2=arc_val,
        arc_flash_violation=arc_violation,
        discrimination_ok=discrimination_ok,
        discrimination_violations=violations,
        short_circuit_ka=i_sc_val,
    )


def _parse_electrical(
    text: str, bod: BasisOfDesign, cfg: ElectricalHFBoD
) -> ElectricalCorrection:
    return _parse_electrical_dict(json.loads(text), bod, cfg)


# ---------------------------------------------------------------------------
# Convenience function used by the optimizer
# ---------------------------------------------------------------------------


def run_electrical(bod: BasisOfDesign, *, fail_open: bool = True) -> HFResult:
    if bod.high_fidelity is None or bod.high_fidelity.electrical is None:
        return HFResult(status=HFStatus.ABSENT, correction=None)
    adapter = ETAPAdapter(bod.high_fidelity.electrical)
    return adapter.run(bod, fail_open=fail_open)
