"""Unit tests for the AI Factory optimizer."""

from __future__ import annotations

from pathlib import Path

import pytest

from firmus_ai_factory.bod import load_bod
from firmus_ai_factory.optimization import optimize
from firmus_ai_factory.optimization.factory_optimizer import (
    EnergyPack,
    OptimizerResult,
    RoIPack,
    _evaluate_bod,
    _mutate_bod,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BT1_2 = REPO_ROOT / "examples" / "bod" / "bt1_2.yaml"


class TestEvaluateBoD:
    def test_evaluate_returns_positive_energy(self) -> None:
        bod = load_bod(BT1_2)
        roi, energy = _evaluate_bod(bod)
        assert isinstance(roi, RoIPack)
        assert isinstance(energy, EnergyPack)
        assert energy.annual_energy_mwh > 0.0
        assert energy.pue > 1.0
        assert roi.capex_total > 0.0
        assert roi.annual_revenue > 0.0

    def test_scaling_num_racks_increases_energy(self) -> None:
        bod = load_bod(BT1_2)
        bod_small = _mutate_bod(bod, {"nvidia_platform.num_racks": 16})
        bod_big = _mutate_bod(bod, {"nvidia_platform.num_racks": 40})
        _, e_small = _evaluate_bod(bod_small)
        _, e_big = _evaluate_bod(bod_big)
        assert e_big.annual_energy_mwh > e_small.annual_energy_mwh


class TestOptimize:
    def test_returns_valid_result(self) -> None:
        bod = load_bod(BT1_2)
        result = optimize(bod, samples_per_var=3, max_candidates=27)
        assert isinstance(result, OptimizerResult)
        assert result.candidates_evaluated > 0
        assert result.roi.npv == result.roi.npv  # not NaN
        assert result.energy.annual_energy_mwh > 0.0

    def test_optimal_bod_is_valid_schema(self) -> None:
        bod = load_bod(BT1_2)
        result = optimize(bod, samples_per_var=3, max_candidates=27)
        # Round-trip through the schema by re-serialising.
        payload = result.optimal_bod.model_dump(mode="json")
        from firmus_ai_factory.bod import load_bod_dict

        reloaded = load_bod_dict(payload)
        assert reloaded.metadata.site_id == bod.metadata.site_id

    def test_sensitivity_populated(self) -> None:
        bod = load_bod(BT1_2)
        result = optimize(bod, samples_per_var=3, max_candidates=27)
        assert len(result.sensitivity) >= 1
        # Sensitivity is sorted by swing descending.
        swings = [e.swing for e in result.sensitivity]
        assert swings == sorted(swings, reverse=True)

    def test_to_dict_is_jsonable(self) -> None:
        bod = load_bod(BT1_2)
        result = optimize(bod, samples_per_var=3, max_candidates=27)
        import json

        payload = json.dumps(result.to_dict(), default=str)
        assert '"optimal_bod"' in payload
        assert '"roi"' in payload
        assert '"energy"' in payload
        assert '"sensitivity"' in payload

    def test_no_free_variables_still_runs(self) -> None:
        bod = load_bod(BT1_2)
        payload = bod.model_dump(mode="json")
        payload["optimizer"]["free_variables"] = {}
        from firmus_ai_factory.bod import load_bod_dict

        stripped = load_bod_dict(payload)
        result = optimize(stripped, samples_per_var=3, max_candidates=5)
        assert result.candidates_evaluated == 1
        assert result.sensitivity == []
