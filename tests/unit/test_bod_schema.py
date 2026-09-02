"""Unit tests for the Basis of Design (BoD) schema and loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from firmus_ai_factory.bod import (
    BasisOfDesign,
    dump_bod,
    export_json_schema,
    load_bod,
    load_bod_dict,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BT1_2_PATH = REPO_ROOT / "examples" / "bod" / "bt1_2.yaml"
SOUTHGATE_PATH = REPO_ROOT / "examples" / "bod" / "southgate_01.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bt1_2() -> BasisOfDesign:
    return load_bod(BT1_2_PATH)


@pytest.fixture
def southgate() -> BasisOfDesign:
    return load_bod(SOUTHGATE_PATH)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestBoDLoadRoundtrip:
    def test_bt1_2_loads(self, bt1_2: BasisOfDesign) -> None:
        assert bt1_2.metadata.site_id == "BT1_2"
        assert bt1_2.nvidia_platform.platform == "vr_nvl72_max_p"
        assert bt1_2.nvidia_platform.num_racks == 32
        assert bt1_2.cooling.architecture == "benmax_hcu2500"
        assert bt1_2.grid.region == "australia_nem"

    def test_southgate_loads(self, southgate: BasisOfDesign) -> None:
        assert southgate.metadata.site_id == "southgate_01"
        assert southgate.nvidia_platform.platform == "gb300_nvl72"

    def test_roundtrip_json(self, bt1_2: BasisOfDesign, tmp_path: Path) -> None:
        out = tmp_path / "bod.json"
        dump_bod(bt1_2, out)
        reloaded = load_bod(out)
        assert reloaded.model_dump(mode="json") == bt1_2.model_dump(mode="json")

    def test_roundtrip_yaml(self, bt1_2: BasisOfDesign, tmp_path: Path) -> None:
        yaml = pytest.importorskip("yaml")  # noqa: F841
        out = tmp_path / "bod.yaml"
        dump_bod(bt1_2, out)
        reloaded = load_bod(out)
        assert reloaded.model_dump(mode="json") == bt1_2.model_dump(mode="json")

    def test_json_schema_exports(self, tmp_path: Path) -> None:
        out = export_json_schema(tmp_path / "bod.schema.json")
        schema = json.loads(out.read_text())
        assert schema["title"] == "BasisOfDesign"
        # All nine domains + metadata + economics + optimizer are required.
        required = set(schema["required"])
        for key in (
            "metadata",
            "site",
            "climate",
            "mechanical",
            "cooling",
            "electrical",
            "grid",
            "tariff",
            "network",
            "nvidia_platform",
            "economics",
        ):
            assert key in required, f"missing required domain {key!r}"


# ---------------------------------------------------------------------------
# Validation guardrails
# ---------------------------------------------------------------------------


class TestBoDValidation:
    def test_forbids_unknown_top_level_field(self, bt1_2: BasisOfDesign) -> None:
        payload = bt1_2.model_dump(mode="json")
        payload["extra_field"] = 123
        with pytest.raises(ValidationError):
            load_bod_dict(payload)

    def test_site_id_mismatch_rejected(self, bt1_2: BasisOfDesign) -> None:
        payload = bt1_2.model_dump(mode="json")
        payload["site"]["site_id"] = "OTHER"
        with pytest.raises(ValidationError, match="site_id"):
            load_bod_dict(payload)

    def test_facility_load_over_poi_rejected(self, bt1_2: BasisOfDesign) -> None:
        payload = bt1_2.model_dump(mode="json")
        payload["electrical"]["design_facility_load_mw"] = 999.0
        with pytest.raises(ValidationError, match="poi_capacity_mw"):
            load_bod_dict(payload)

    def test_pue_below_ppue_rejected(self, bt1_2: BasisOfDesign) -> None:
        payload = bt1_2.model_dump(mode="json")
        payload["cooling"]["design_pue"] = 1.02
        payload["cooling"]["design_ppue"] = 1.10
        with pytest.raises(ValidationError, match="design_pue"):
            load_bod_dict(payload)

    def test_currency_mismatch_rejected(self, bt1_2: BasisOfDesign) -> None:
        payload = bt1_2.model_dump(mode="json")
        payload["tariff"]["currency"] = "USD"
        # economics still AUD -> reject
        with pytest.raises(ValidationError, match="currency"):
            load_bod_dict(payload)

    def test_cooling_loop_delta_t_rejected(self, bt1_2: BasisOfDesign) -> None:
        payload = bt1_2.model_dump(mode="json")
        payload["cooling"]["loops"][0]["return_temp_c"] = payload["cooling"]["loops"][0][
            "supply_temp_c"
        ]
        with pytest.raises(ValidationError, match="return_temp_c"):
            load_bod_dict(payload)

    def test_ambient_range_reversed_rejected(self, bt1_2: BasisOfDesign) -> None:
        payload = bt1_2.model_dump(mode="json")
        payload["climate"]["ambient_temp_range_c"] = [40.0, -5.0]
        with pytest.raises(ValidationError, match="range"):
            load_bod_dict(payload)

    def test_ppa_requires_price(self, bt1_2: BasisOfDesign) -> None:
        payload = bt1_2.model_dump(mode="json")
        payload["tariff"]["structure"] = "negotiated_ppa"
        payload["tariff"]["ppa_ccy_per_mwh"] = None
        with pytest.raises(ValidationError, match="ppa"):
            load_bod_dict(payload)

    def test_is_frozen_reflects_approval(self, bt1_2: BasisOfDesign) -> None:
        assert bt1_2.is_frozen is False
        payload = bt1_2.model_dump(mode="json")
        payload["metadata"]["approval"] = {
            "gate": "G2",
            "approved_by": "Daniel Kearney",
            "approved_on": "2026-09-01",
        }
        frozen = load_bod_dict(payload)
        assert frozen.is_frozen is True
