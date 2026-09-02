"""Unit tests for BoD -> runtime hydration."""

from __future__ import annotations

from pathlib import Path

import pytest

from firmus_ai_factory.bod import (
    hydrate_electricity_tariff,
    hydrate_factory,
    hydrate_factory_config,
    load_bod,
)
from firmus_ai_factory.factory_config import CoolingType, FactoryConfig, GPUPlatform
from firmus_ai_factory.grid.regional_grids import GridRegion


REPO_ROOT = Path(__file__).resolve().parents[2]
BT1_2 = REPO_ROOT / "examples" / "bod" / "bt1_2.yaml"


class TestHydrateFactoryConfig:
    def test_hydrates_bt1_2(self) -> None:
        bod = load_bod(BT1_2)
        cfg = hydrate_factory_config(bod)
        assert isinstance(cfg, FactoryConfig)
        assert cfg.platform == GPUPlatform.VR_NVL72_MAX_P
        assert cfg.cooling_type == CoolingType.BENMAX_HCU2500
        assert cfg.grid_region == GridRegion.AUSTRALIA_NEM
        assert cfg.num_racks == 32
        assert cfg.coolant_inlet_temp_c == 35.0

    def test_hydrate_factory_end_to_end(self) -> None:
        bod = load_bod(BT1_2)
        factory = hydrate_factory(bod)
        report = factory.generate_full_report()
        assert report["factory"]["num_racks"] == 32
        assert report["power"]["pue"] > 1.0
        assert report["power"]["it_power_mw"] > 0.0

    def test_rejects_unsupported_platform(self) -> None:
        bod = load_bod(BT1_2)
        payload = bod.model_dump(mode="json")
        payload["nvidia_platform"]["platform"] = "hgx_b200"  # in BoD, not in runtime
        # Also switch cooling to satisfy platform-cooling mapping in the runtime
        from firmus_ai_factory.bod import load_bod_dict

        bod2 = load_bod_dict(payload)
        with pytest.raises(NotImplementedError, match="platform"):
            hydrate_factory_config(bod2)


class TestHydrateTariff:
    def test_tou_maps_correctly(self) -> None:
        bod = load_bod(BT1_2)
        tariff = hydrate_electricity_tariff(bod)
        assert tariff.tariff_type == "TOU"
        assert set(tariff.rates.keys()) == {"on_peak", "mid_peak", "off_peak"}
        assert tariff.rates["on_peak"] == 0.22
        assert tariff.demand_charge_rate == 12.0
        assert tariff.monthly_fixed_charge == 800.0
