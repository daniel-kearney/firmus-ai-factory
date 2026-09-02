"""Contract tests over the real Ansys / ETAP response fixtures.

These tests exist so that changes to the ThermalCorrection or
ElectricalCorrection parsers are caught the moment they drift from what
solver teams actually produce.  If a fixture stops parsing here, the
external solver contract is broken - fix the code, not the fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from firmus_ai_factory.bod.loader import load_bod
from firmus_ai_factory.hf import HFStatus, run_electrical, run_thermal
from firmus_ai_factory.hf.base import reset_default_caches
from firmus_ai_factory.hf.electrical import _parse_electrical_dict
from firmus_ai_factory.hf.thermal import _parse_thermal_dict


REPO_ROOT = Path(__file__).resolve().parents[2]
ANSYS_DIR = REPO_ROOT / "examples" / "hf" / "ansys"
ETAP_DIR = REPO_ROOT / "examples" / "hf" / "etap"
FILE_BACKEND_BOD = REPO_ROOT / "examples" / "bod" / "bt1_2_with_hf_file_backend.yaml"


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    reset_default_caches()


# ---------------------------------------------------------------------------
# Ansys fixtures
# ---------------------------------------------------------------------------


class TestAnsysFixtures:
    def test_fluent_nominal_parses(self) -> None:
        data = json.loads((ANSYS_DIR / "fluent_nominal.json").read_text())
        corr = _parse_thermal_dict(data)
        assert corr.pue_bump == pytest.approx(0.008)
        assert corr.rack_hotspot_c == pytest.approx(41.3)
        assert corr.rack_hotspot_margin_c == pytest.approx(3.7)
        assert corr.cdu_delta_t_c == pytest.approx(11.2)
        assert corr.hotspot_violation is False

    def test_icepak_hotspot_parses_and_flags_violation(self) -> None:
        data = json.loads((ANSYS_DIR / "icepak_hotspot.json").read_text())
        corr = _parse_thermal_dict(data)
        assert corr.rack_hotspot_c == pytest.approx(46.8)
        assert corr.rack_hotspot_margin_c == pytest.approx(-1.8)
        assert corr.hotspot_violation is True

    def test_extra_solver_fields_are_ignored(self) -> None:
        """Provenance blocks (mesh_cells, loop_summary, ...) must not blow up
        the parser - the contract is defined by the top-level fields only.
        """
        data = json.loads((ANSYS_DIR / "fluent_nominal.json").read_text())
        assert "provenance" in data
        assert "per_rack" in data
        assert "loop_summary" in data
        corr = _parse_thermal_dict(data)
        assert corr.pue_bump == pytest.approx(0.008)


# ---------------------------------------------------------------------------
# ETAP fixtures
# ---------------------------------------------------------------------------


class TestETAPFixtures:
    def test_nominal_parses_and_passes_discrimination(self) -> None:
        bod = load_bod(FILE_BACKEND_BOD)
        data = json.loads((ETAP_DIR / "etap_nominal.json").read_text())
        # Strip $comment field the parser doesn't expect.
        data["discrimination_margins_s"] = {
            k: v for k, v in data["discrimination_margins_s"].items() if k != "$comment"
        }
        corr = _parse_electrical_dict(data, bod, bod.high_fidelity.electrical)
        assert corr.losses_pct == pytest.approx(2.1)
        assert corr.worst_arc_flash_cal_cm2 == pytest.approx(6.4)
        assert corr.short_circuit_ka == pytest.approx(42.5)
        assert corr.arc_flash_violation is False
        assert corr.discrimination_ok is True
        assert corr.discrimination_violations == []

    def test_discrimination_miss_flags_specific_boundary(self) -> None:
        bod = load_bod(FILE_BACKEND_BOD)
        data = json.loads((ETAP_DIR / "etap_discrimination_miss.json").read_text())
        corr = _parse_electrical_dict(data, bod, bod.high_fidelity.electrical)
        assert corr.discrimination_ok is False
        assert len(corr.discrimination_violations) == 1
        assert "MV_incomer>LV_incomer" in corr.discrimination_violations[0]
        assert "0.22" in corr.discrimination_violations[0]
        assert "0.40" in corr.discrimination_violations[0]

    def test_arc_flash_budget_check_uses_bod_value(self) -> None:
        bod = load_bod(FILE_BACKEND_BOD)
        # Sanity: the fixture is well below the BoD budget so no violation.
        budget = bod.electrical.arc_flash_max_incident_energy_cal_cm2
        data = json.loads((ETAP_DIR / "etap_nominal.json").read_text())
        data["discrimination_margins_s"] = {
            k: v for k, v in data["discrimination_margins_s"].items() if k != "$comment"
        }
        corr = _parse_electrical_dict(data, bod, bod.high_fidelity.electrical)
        assert corr.worst_arc_flash_cal_cm2 < budget
        assert corr.arc_flash_violation is False

    def test_check_discrimination_targets_off_disables_the_check(self) -> None:
        bod = load_bod(FILE_BACKEND_BOD)
        cfg = bod.high_fidelity.electrical.model_copy(
            update={"check_discrimination_targets": False}
        )
        data = json.loads((ETAP_DIR / "etap_discrimination_miss.json").read_text())
        corr = _parse_electrical_dict(data, bod, cfg)
        # Even with a solver-reported miss, the adapter reports OK when the
        # BoD-aware check is disabled.
        assert corr.discrimination_ok is True
        assert corr.discrimination_violations == []


# ---------------------------------------------------------------------------
# End-to-end via the file backend + shipped manifest
# ---------------------------------------------------------------------------


class TestFileBackendEndToEnd:
    def test_file_backend_bod_loads(self) -> None:
        bod = load_bod(FILE_BACKEND_BOD)
        assert bod.high_fidelity is not None
        assert bod.high_fidelity.is_active
        assert bod.high_fidelity.thermal.backend == "file"
        assert bod.high_fidelity.electrical.backend == "file"

    def test_thermal_manifest_serves_ok_for_base_bod(self) -> None:
        bod = load_bod(FILE_BACKEND_BOD)
        res = run_thermal(bod)
        assert res.status == HFStatus.OK
        assert res.correction is not None
        assert res.correction.pue_bump == pytest.approx(0.008)

    def test_electrical_manifest_serves_ok_for_base_bod(self) -> None:
        bod = load_bod(FILE_BACKEND_BOD)
        res = run_electrical(bod)
        assert res.status == HFStatus.OK
        assert res.correction is not None
        assert res.correction.losses_pct == pytest.approx(2.1)
        assert res.correction.discrimination_ok is True

    def test_optimizer_runs_end_to_end_on_shipped_fixtures(self) -> None:
        """Full sweep against the file-backed BoD. The winning candidate must
        pick up the manifest-served electrical correction (2.1% losses).
        """
        from firmus_ai_factory.optimization import optimize

        bod = load_bod(FILE_BACKEND_BOD)
        res = optimize(bod, samples_per_var=3, with_sensitivity=False)
        assert res.candidates_evaluated > 0
        # Winner sits at the base BoD's electrical key (mutations don't
        # touch electrical) so the manifest hit is exact.
        assert res.hf.electrical_status == HFStatus.OK.value
        assert res.hf.electrical_correction["losses_pct"] == pytest.approx(2.1)
        # NPV is finite and positive.
        assert res.roi.npv > 0
