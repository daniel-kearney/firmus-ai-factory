"""Tests for the Ansys / ETAP high-fidelity adapters."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from firmus_ai_factory.bod.high_fidelity import (
    ElectricalHFBoD,
    HFBackend,
    HighFidelityBoD,
    ThermalHFBoD,
)
from firmus_ai_factory.bod.loader import load_bod
from firmus_ai_factory.hf import HFStatus, run_electrical, run_thermal
from firmus_ai_factory.hf.base import reset_default_caches
from firmus_ai_factory.hf.electrical import _electrical_key
from firmus_ai_factory.hf.thermal import _thermal_key
from firmus_ai_factory.optimization.factory_optimizer import (
    HFPack,
    _evaluate_bod,
    optimize,
)


BT1_2 = Path("examples/bod/bt1_2.yaml")


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    reset_default_caches()


@pytest.fixture
def bod():
    return load_bod(BT1_2)


def _attach_hf(bod, *, thermal=None, electrical=None, fail_open=True):
    """Return a copy of ``bod`` with a HighFidelity block attached."""
    payload = bod.model_dump(mode="json")
    hf = {"fail_open": fail_open}
    if thermal is not None:
        hf["thermal"] = thermal
    if electrical is not None:
        hf["electrical"] = electrical
    payload["high_fidelity"] = hf
    return bod.model_validate(payload)


# ---------------------------------------------------------------------------
# Schema-level guards on the HF block
# ---------------------------------------------------------------------------


class TestHighFidelitySchema:
    def test_absent_block_is_valid(self, bod) -> None:
        assert bod.high_fidelity is None

    def test_active_flag_reflects_backends(self, bod) -> None:
        hf_bod = _attach_hf(
            bod,
            thermal={"backend": "disabled"},
            electrical={"backend": "disabled"},
        )
        assert hf_bod.high_fidelity.is_active is False

        hf_bod2 = _attach_hf(
            bod,
            thermal={"backend": "subprocess", "endpoint": "/opt/ansys/fluent"},
        )
        assert hf_bod2.high_fidelity.is_active is True

    def test_non_disabled_backend_requires_endpoint(self, bod) -> None:
        with pytest.raises(Exception):
            _attach_hf(bod, thermal={"backend": "http"})

    def test_disabled_backend_needs_no_endpoint(self, bod) -> None:
        hf_bod = _attach_hf(bod, thermal={"backend": "disabled"})
        assert hf_bod.high_fidelity.thermal.backend == HFBackend.DISABLED


# ---------------------------------------------------------------------------
# Runtime status paths
# ---------------------------------------------------------------------------


class TestStatuses:
    def test_absent_returns_absent(self, bod) -> None:
        assert run_thermal(bod).status == HFStatus.ABSENT
        assert run_electrical(bod).status == HFStatus.ABSENT

    def test_disabled_returns_disabled(self, bod) -> None:
        hf_bod = _attach_hf(
            bod,
            thermal={"backend": "disabled"},
            electrical={"backend": "disabled"},
        )
        assert run_thermal(hf_bod).status == HFStatus.DISABLED
        assert run_electrical(hf_bod).status == HFStatus.DISABLED

    def test_unreachable_http_falls_back_to_shim(self, bod) -> None:
        hf_bod = _attach_hf(
            bod,
            thermal={
                "backend": "http",
                "endpoint": "http://127.0.0.1:1/thermal",
                "timeout_s": 0.05,
            },
            electrical={
                "backend": "http",
                "endpoint": "http://127.0.0.1:1/etap",
                "timeout_s": 0.05,
            },
            fail_open=True,
        )
        t = run_thermal(hf_bod)
        e = run_electrical(hf_bod)
        assert t.status == HFStatus.FALLBACK
        assert e.status == HFStatus.FALLBACK
        assert t.correction is not None
        assert e.correction is not None
        assert t.correction.pue_bump >= 0.0
        assert e.correction.losses_pct > 0.0

    def test_fail_closed_raises(self, bod) -> None:
        from firmus_ai_factory.hf.base import HFError

        hf_bod = _attach_hf(
            bod,
            thermal={
                "backend": "http",
                "endpoint": "http://127.0.0.1:1/thermal",
                "timeout_s": 0.05,
            },
            fail_open=False,
        )
        with pytest.raises(HFError):
            run_thermal(hf_bod, fail_open=False)


# ---------------------------------------------------------------------------
# File backend + cache
# ---------------------------------------------------------------------------


class TestFileBackend:
    def test_thermal_file_backend_reads_lookup(self, bod, tmp_path: Path) -> None:
        # Attach adapter first so we can compute the exact key.
        hf_bod = _attach_hf(
            bod,
            thermal={"backend": "file", "endpoint": str(tmp_path)},
        )
        key = _thermal_key(hf_bod)
        (tmp_path / f"{key}.json").write_text(
            json.dumps(
                {
                    "pue_bump": 0.012,
                    "rack_hotspot_c": 42.1,
                    "rack_hotspot_margin_c": 2.9,
                    "cdu_delta_t_c": 11.5,
                    "hotspot_violation": False,
                }
            )
        )
        res = run_thermal(hf_bod)
        assert res.status == HFStatus.OK
        assert res.correction.pue_bump == pytest.approx(0.012)
        assert res.correction.rack_hotspot_c == pytest.approx(42.1)

    def test_electrical_file_backend_flags_arc_flash(self, bod, tmp_path: Path) -> None:
        hf_bod = _attach_hf(
            bod,
            electrical={"backend": "file", "endpoint": str(tmp_path)},
        )
        key = _electrical_key(hf_bod)
        budget = hf_bod.electrical.arc_flash_max_incident_energy_cal_cm2
        (tmp_path / f"{key}.json").write_text(
            json.dumps(
                {
                    "losses_pct": 2.4,
                    "worst_arc_flash_cal_cm2": budget + 1.0,
                    "short_circuit_ka": 38.2,
                    "discrimination_margins_s": {},
                }
            )
        )
        res = run_electrical(hf_bod)
        assert res.status == HFStatus.OK
        assert res.correction.arc_flash_violation is True

    def test_cache_hit_second_call(self, bod, tmp_path: Path) -> None:
        hf_bod = _attach_hf(
            bod,
            thermal={"backend": "file", "endpoint": str(tmp_path)},
        )
        key = _thermal_key(hf_bod)
        (tmp_path / f"{key}.json").write_text(json.dumps({"pue_bump": 0.005}))
        first = run_thermal(hf_bod)
        second = run_thermal(hf_bod)
        assert first.cache_hit is False
        assert second.cache_hit is True


# ---------------------------------------------------------------------------
# End-to-end: optimizer uses HF corrections
# ---------------------------------------------------------------------------


class TestOptimizerHFIntegration:
    def test_evaluate_returns_hf_pack(self, bod) -> None:
        _, _, hf = _evaluate_bod(bod)
        assert isinstance(hf, HFPack)
        # No HF wired -> both statuses are 'absent'.
        assert hf.thermal_status == HFStatus.ABSENT.value
        assert hf.electrical_status == HFStatus.ABSENT.value

    def test_fallback_shim_bumps_pue(self, bod) -> None:
        base_roi, base_energy, _ = _evaluate_bod(bod)
        hf_bod = _attach_hf(
            bod,
            thermal={
                "backend": "http",
                "endpoint": "http://127.0.0.1:1/thermal",
                "timeout_s": 0.05,
            },
            fail_open=True,
        )
        _, hf_energy, hf_pack = _evaluate_bod(hf_bod)
        # Fallback shim adds a positive PUE bump for BT1_2's inlet temp.
        assert hf_energy.pue >= base_energy.pue
        assert hf_pack.thermal_status == HFStatus.FALLBACK.value

    def test_hotspot_violation_disqualifies_candidate(self, bod, tmp_path: Path) -> None:
        # Force every thermal call to report a hotspot breach via a manifest.
        manifest_path = tmp_path / "manifest.json"
        # Create adapter first to sniff keys as the sweep runs.
        hf_bod = _attach_hf(
            bod,
            thermal={"backend": "file", "endpoint": str(tmp_path)},
        )
        # Build a manifest that answers *any* key with a hotspot violation.
        # We do that by writing a shim adapter file that mirrors every key we
        # observe. Easier: pre-populate the base BoD key and let the fallback
        # (missing key -> HFError -> shim) handle the rest, then rely on the
        # shim's own violation logic. For that we push inlet temp very high.
        hot_bod_payload = hf_bod.model_dump(mode="json")
        hot_bod_payload["cooling"]["inlet_temp_c"] = 55.0
        hot_bod_payload["optimizer"]["free_variables"] = {
            "nvidia_platform.num_racks": [16, 32]
        }
        hot_bod = bod.model_validate(hot_bod_payload)
        with pytest.raises(RuntimeError, match="0 valid candidates"):
            optimize(hot_bod, with_sensitivity=False, samples_per_var=2)
