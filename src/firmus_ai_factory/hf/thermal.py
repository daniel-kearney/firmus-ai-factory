"""Ansys thermal adapter.

Three transports:

    * ``subprocess`` - invokes ``ansys_fluent --batch --journal <path>``, reads
      the JSON summary the journal is expected to write to stdout / a
      sibling file.
    * ``http``       - POSTs the BoD payload to a solve service, GETs the
      correction back.
    * ``file``       - looks up a pre-computed correction on disk (key =
      content hash of the thermal-relevant BoD subset).

If the requested backend is unreachable and ``fail_open`` is true, the
adapter returns a deterministic *analytic shim* correction so the
optimizer keeps making progress. The shim is not a stand-in for real
CFD - it just captures the qualitative direction of the coupling
(higher inlet temp -> more pump work -> higher PUE bump) so tornado
plots stay honest.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from firmus_ai_factory.bod.high_fidelity import HFBackend, ThermalHFBoD
from firmus_ai_factory.bod.schema import BasisOfDesign
from firmus_ai_factory.hf.base import (
    HFCache,
    HFError,
    HFResult,
    HFStatus,
    ThermalCorrection,
    get_default_thermal_cache,
    hash_bod_subset,
)


# NVIDIA envelope proxy - used only by the analytic shim for the
# hotspot-violation flag. Real envelope comes from Ansys.
_MAX_COOLANT_C = 45.0


def _thermal_key(bod: BasisOfDesign) -> str:
    """Content hash over the BoD subset that thermal cares about."""
    sub = {
        "platform": bod.nvidia_platform.platform,
        "num_racks": bod.nvidia_platform.num_racks,
        "rack_power_kw": bod.nvidia_platform.rack_power_kw,
        "coolant_inlet_c": bod.cooling.inlet_temp_c,
        "cooling": bod.cooling.model_dump(mode="json"),
        "climate_dry_bulb_c": bod.climate.design_dry_bulb_c,
        "mechanical": bod.mechanical.model_dump(mode="json"),
    }
    return "thermal:" + hash_bod_subset(sub)


class AnsysThermalAdapter:
    """Adapter that resolves a BoD candidate to a ``ThermalCorrection``."""

    def __init__(self, cfg: ThermalHFBoD, cache: Optional[HFCache] = None) -> None:
        self.cfg = cfg
        self.cache = cache or get_default_thermal_cache()

    # ---------- transports ----------

    def _run_subprocess(self, bod: BasisOfDesign) -> ThermalCorrection:
        assert self.cfg.endpoint
        payload = json.dumps(bod.model_dump(mode="json"))
        proc = subprocess.run(
            [self.cfg.endpoint, "--batch", "--journal", "-"],
            input=payload,
            text=True,
            capture_output=True,
            timeout=self.cfg.timeout_s,
        )
        if proc.returncode != 0:
            raise HFError(f"Ansys subprocess exited {proc.returncode}: {proc.stderr[:400]}")
        return _parse_thermal_json(proc.stdout)

    def _run_http(self, bod: BasisOfDesign) -> ThermalCorrection:
        import urllib.request  # stdlib to avoid a new dep

        assert self.cfg.endpoint
        data = json.dumps(bod.model_dump(mode="json")).encode("utf-8")
        req = urllib.request.Request(
            self.cfg.endpoint + "/solve",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
            body = resp.read().decode("utf-8")
        return _parse_thermal_json(body)

    def _run_file(self, bod: BasisOfDesign) -> ThermalCorrection:
        assert self.cfg.endpoint
        base = Path(self.cfg.endpoint)
        key = _thermal_key(bod)
        candidate = base / f"{key}.json"
        if not candidate.exists():
            # Manifest-style fallback: single manifest.json with dict-of-keys.
            manifest = base / "manifest.json"
            if manifest.exists():
                data = json.loads(manifest.read_text())
                if key in data:
                    return _parse_thermal_dict(data[key])
            raise HFError(f"No file-mode result at {candidate} or manifest for key {key}")
        return _parse_thermal_json(candidate.read_text())

    # ---------- analytic shim ----------

    def _analytic_shim(self, bod: BasisOfDesign) -> ThermalCorrection:
        """Deterministic surrogate used when a solver is unreachable.

        Not physics-grade: encodes the sign of the coupling only.
        """
        inlet = bod.cooling.inlet_temp_c
        rack_power = bod.nvidia_platform.rack_power_kw or 150.0
        # Linear PUE bump: hotter inlet + hotter rack -> more pump/fan work.
        bump = max(0.0, (inlet - 30.0) * 0.001) + max(0.0, (rack_power - 150.0) * 0.00005)
        rack_hotspot = inlet + 8.0 + (rack_power - 150.0) * 0.01
        margin = _MAX_COOLANT_C - rack_hotspot
        cdu_dt = 10.0 + max(0.0, (rack_power - 150.0) * 0.02)
        return ThermalCorrection(
            pue_bump=round(bump, 4),
            rack_hotspot_c=round(rack_hotspot, 2),
            rack_hotspot_margin_c=round(margin, 2),
            cdu_delta_t_c=round(cdu_dt, 2),
            hotspot_violation=margin < 0.0,
        )

    # ---------- public entry ----------

    def run(self, bod: BasisOfDesign, *, fail_open: bool = True) -> HFResult:
        # Backend-disabled: report but do not attempt.
        if self.cfg.backend == HFBackend.DISABLED:
            return HFResult(status=HFStatus.DISABLED, correction=None, solver=self.cfg.solver)

        key = _thermal_key(bod)
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
            else:  # pragma: no cover - Enum exhausted
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
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_thermal_dict(data: Dict[str, Any]) -> ThermalCorrection:
    """Coerce a JSON dict from the solver into a ThermalCorrection.

    The contract is deliberately permissive - the solver's schema is
    external and drifts. Unknown fields are ignored; missing fields fall
    back to the ThermalCorrection defaults.
    """
    allowed = {"pue_bump", "rack_hotspot_c", "rack_hotspot_margin_c", "cdu_delta_t_c", "hotspot_violation"}
    kwargs = {k: v for k, v in data.items() if k in allowed}
    return ThermalCorrection(**kwargs)


def _parse_thermal_json(text: str) -> ThermalCorrection:
    return _parse_thermal_dict(json.loads(text))


# ---------------------------------------------------------------------------
# Convenience function used by the optimizer
# ---------------------------------------------------------------------------


def run_thermal(bod: BasisOfDesign, *, fail_open: bool = True) -> HFResult:
    """Resolve the BoD's thermal adapter (if any) and return an HFResult."""
    if bod.high_fidelity is None or bod.high_fidelity.thermal is None:
        return HFResult(status=HFStatus.ABSENT, correction=None)
    adapter = AnsysThermalAdapter(bod.high_fidelity.thermal)
    return adapter.run(bod, fail_open=fail_open)
