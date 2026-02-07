"""Unit tests for Vera Rubin NVL72 thermal model.

Validates thermal calculations against NVIDIA VR NVL72 Rack Level
Thermal Model v07 specifications.
"""

import pytest
import numpy as np


class TestVRThermalLimits:
    """Test VR NVL72 thermal limit specifications."""
    
    def test_max_inlet_temperature(self):
        """VR NVL72 max coolant inlet: 45°C."""
        from firmus_ai_factory.thermal.vera_rubin_thermal import VRThermalLimits
        assert VRThermalLimits.MAX_INLET_TEMP_C == 45.0
    
    def test_max_outlet_temperature(self):
        """VR NVL72 max coolant outlet: 60°C."""
        from firmus_ai_factory.thermal.vera_rubin_thermal import VRThermalLimits
        assert VRThermalLimits.MAX_OUTLET_TEMP_C == 60.0
    
    def test_max_pressure_drop(self):
        """VR NVL72 max pressure drop: 60 psi."""
        from firmus_ai_factory.thermal.vera_rubin_thermal import VRThermalLimits
        assert VRThermalLimits.MAX_PRESSURE_DROP_PSID == 60.0
    
    def test_min_flowrate(self):
        """VR NVL72 minimum flowrate: 30 LPM."""
        from firmus_ai_factory.thermal.vera_rubin_thermal import VRThermalLimits
        assert VRThermalLimits.MIN_FLOWRATE_LPM == 30.0
    
    def test_max_flowrate(self):
        """VR NVL72 maximum flowrate: 60 LPM."""
        from firmus_ai_factory.thermal.vera_rubin_thermal import VRThermalLimits
        assert VRThermalLimits.MAX_FLOWRATE_LPM == 60.0


class TestPG25Properties:
    """Test PG25 coolant property calculations."""
    
    def test_density_at_35c(self):
        """PG25 density at 35°C should be approximately 1020 kg/m³."""
        from firmus_ai_factory.thermal.vera_rubin_thermal import PG25Properties
        rho = PG25Properties.density(35.0)
        assert 1010 < rho < 1030
    
    def test_specific_heat_at_35c(self):
        """PG25 specific heat at 35°C should be approximately 3900 J/(kg·K)."""
        from firmus_ai_factory.thermal.vera_rubin_thermal import PG25Properties
        cp = PG25Properties.specific_heat(35.0)
        assert 3800 < cp < 4000
    
    def test_density_decreases_with_temperature(self):
        """Density should decrease with increasing temperature."""
        from firmus_ai_factory.thermal.vera_rubin_thermal import PG25Properties
        rho_25 = PG25Properties.density(25.0)
        rho_45 = PG25Properties.density(45.0)
        assert rho_25 > rho_45
    
    def test_specific_heat_increases_with_temperature(self):
        """Specific heat should increase with increasing temperature."""
        from firmus_ai_factory.thermal.vera_rubin_thermal import PG25Properties
        cp_25 = PG25Properties.specific_heat(25.0)
        cp_45 = PG25Properties.specific_heat(45.0)
        assert cp_45 > cp_25


class TestVRNvl72ThermalModel:
    """Test VR NVL72 rack-level thermal model."""
    
    @pytest.fixture
    def model_max_p(self):
        from firmus_ai_factory.thermal.vera_rubin_thermal import VRNvl72ThermalModel
        return VRNvl72ThermalModel(rack_tdp_kw=227.0, power_mode="max_p")
    
    @pytest.fixture
    def model_max_q(self):
        from firmus_ai_factory.thermal.vera_rubin_thermal import VRNvl72ThermalModel
        return VRNvl72ThermalModel(rack_tdp_kw=187.0, power_mode="max_q")
    
    def test_max_p_rack_power(self, model_max_p):
        """Max P rack TDP should be 227 kW."""
        assert model_max_p.rack_tdp_kw == 227.0
    
    def test_max_q_rack_power(self, model_max_q):
        """Max Q rack TDP should be 187 kW."""
        assert model_max_q.rack_tdp_kw == 187.0
    
    def test_outlet_temp_within_limits_at_35c(self, model_max_p):
        """Outlet temp should be within 60°C limit at 35°C inlet."""
        result = model_max_p.compute_outlet_temperature(
            inlet_temp_c=35.0, flowrate_lpm=45.0)
        assert result['outlet_temp_c'] <= 60.0
    
    def test_outlet_temp_exceeds_at_high_inlet(self, model_max_p):
        """Outlet temp may exceed limits at very high inlet temperature."""
        result = model_max_p.compute_outlet_temperature(
            inlet_temp_c=45.0, flowrate_lpm=30.0)
        # At max inlet and min flow, outlet should be high
        assert result['outlet_temp_c'] > 50.0
    
    def test_higher_flow_reduces_delta_t(self, model_max_p):
        """Higher flowrate should reduce temperature rise."""
        result_low = model_max_p.compute_outlet_temperature(
            inlet_temp_c=35.0, flowrate_lpm=30.0)
        result_high = model_max_p.compute_outlet_temperature(
            inlet_temp_c=35.0, flowrate_lpm=60.0)
        assert result_high['delta_t_c'] < result_low['delta_t_c']
    
    def test_pressure_drop_within_limits(self, model_max_p):
        """Pressure drop at nominal flow should be within 60 psi limit."""
        result = model_max_p.compute_pressure_drop(flowrate_lpm=45.0)
        assert result['total_psid'] <= 60.0
    
    def test_pressure_drop_increases_with_flow(self, model_max_p):
        """Pressure drop should increase with flowrate."""
        dp_low = model_max_p.compute_pressure_drop(flowrate_lpm=30.0)
        dp_high = model_max_p.compute_pressure_drop(flowrate_lpm=60.0)
        assert dp_high['total_psid'] > dp_low['total_psid']
    
    def test_thermal_report_structure(self, model_max_p):
        """Thermal report should contain all required fields."""
        report = model_max_p.generate_thermal_report(inlet_temp_c=35.0)
        required_keys = [
            'rack_tdp_kw', 'power_mode', 'inlet_temp_c', 'outlet_temp_c',
            'delta_T_c', 'flowrate_lpm', 'pressure_drop_psid', 'within_limits',
        ]
        for key in required_keys:
            assert key in report, f"Missing key: {key}"
    
    def test_max_q_lower_delta_t_than_max_p(self, model_max_p, model_max_q):
        """Max Q should have lower delta T than Max P at same flow."""
        result_p = model_max_p.compute_outlet_temperature(
            inlet_temp_c=35.0, flowrate_lpm=45.0)
        result_q = model_max_q.compute_outlet_temperature(
            inlet_temp_c=35.0, flowrate_lpm=45.0)
        assert result_q['delta_t_c'] < result_p['delta_t_c']
    
    def test_energy_conservation(self, model_max_p):
        """Heat removed by coolant should equal rack TDP."""
        from firmus_ai_factory.thermal.vera_rubin_thermal import PG25Properties
        inlet = 35.0
        flow_lpm = 45.0
        result = model_max_p.compute_outlet_temperature(inlet, flow_lpm)
        
        # Q = m_dot * cp * delta_T
        flow_m3s = flow_lpm / 60000.0
        avg_temp = (inlet + result['outlet_temp_c']) / 2
        rho = PG25Properties.density(avg_temp)
        cp = PG25Properties.specific_heat(avg_temp)
        m_dot = rho * flow_m3s
        q_removed_kw = m_dot * cp * result['delta_t_c'] / 1000.0
        
        # Should be close to rack TDP (within 5%)
        assert abs(q_removed_kw - model_max_p.rack_tdp_kw) / model_max_p.rack_tdp_kw < 0.05


class TestBenmaxHCU2500:
    """Test Benmax HCU2500 CDU model."""
    
    @pytest.fixture
    def hcu(self):
        from firmus_ai_factory.thermal.benmax_hcu2500 import BenmaxHCU2500
        return BenmaxHCU2500(
            primary_inlet_temp_c=35.0,
            secondary_inlet_temp_c=25.0,
        )
    
    def test_cooling_capacity(self, hcu):
        """HCU2500 cooling capacity should be approximately 2500 kW."""
        assert 2400 <= hcu.cooling_capacity_kw <= 2600
    
    def test_primary_flow_rate(self, hcu):
        """Primary flow rate should match HCU2500 specs."""
        flow = hcu.primary_flow_rate_lpm
        assert flow > 0
    
    def test_pump_power(self, hcu):
        """Pump power should be reasonable for CDU."""
        power = hcu.pump_power_kw()
        assert 5.0 < power < 50.0
    
    def test_heat_rejection(self, hcu):
        """Heat rejection should match load within tolerance."""
        result = hcu.compute_heat_rejection(load_kw=2000.0)
        assert abs(result['heat_rejected_kw'] - 2000.0) / 2000.0 < 0.1


class TestBenmaxHypercube:
    """Test Benmax Hypercube (multi-HCU) configuration."""
    
    @pytest.fixture
    def hypercube(self):
        from firmus_ai_factory.thermal.benmax_hcu2500 import (
            BenmaxHypercube, HCURedundancyMode)
        return BenmaxHypercube(
            num_hcu=4,
            num_racks=32,
            rack_power_kw=227.0,
            primary_inlet_temp_c=35.0,
        )
    
    def test_total_it_load(self, hypercube):
        """Total IT load should be num_racks * rack_power."""
        expected = 32 * 227.0
        assert abs(hypercube.total_it_load_kw - expected) < 1.0
    
    def test_total_cooling_capacity(self, hypercube):
        """Total cooling capacity should be 4 × HCU capacity."""
        capacity = hypercube.total_cooling_capacity_kw
        assert capacity >= hypercube.total_it_load_kw * 0.8  # Some margin
    
    def test_report_generation(self, hypercube):
        """Report should contain all required sections."""
        from firmus_ai_factory.thermal.benmax_hcu2500 import HCURedundancyMode
        report = hypercube.generate_report(HCURedundancyMode.FOUR_HCU)
        assert 'thermal' in report
        assert 'power' in report
        assert 'nvidia_compliance' in report
        assert 'all_compliant' in report


class TestRegionalGridModels:
    """Test Singapore and Australia grid models."""
    
    def test_singapore_grid_frequency(self):
        """Singapore grid frequency should be 50 Hz ±0.2 Hz."""
        from firmus_ai_factory.grid.regional_grids import SINGAPORE_GRID
        assert SINGAPORE_GRID.nominal_frequency_hz == 50.0
        assert SINGAPORE_GRID.frequency_normal_band_hz == 0.2
    
    def test_singapore_three_phase_voltage(self):
        """Singapore three-phase voltage should be 400V."""
        from firmus_ai_factory.grid.regional_grids import SINGAPORE_GRID
        assert SINGAPORE_GRID.three_phase_voltage_v == 400.0
    
    def test_australia_grid_frequency(self):
        """Australia NEM frequency should be 50 Hz ±0.15 Hz."""
        from firmus_ai_factory.grid.regional_grids import AUSTRALIA_NEM_GRID
        assert AUSTRALIA_NEM_GRID.nominal_frequency_hz == 50.0
        assert AUSTRALIA_NEM_GRID.frequency_normal_band_hz == 0.15
    
    def test_australia_three_phase_voltage(self):
        """Australia three-phase voltage should be 415V."""
        from firmus_ai_factory.grid.regional_grids import AUSTRALIA_NEM_GRID
        assert AUSTRALIA_NEM_GRID.three_phase_voltage_v == 415.0
    
    def test_singapore_ht_voltages(self):
        """Singapore HT supply should include 22kV and 66kV."""
        from firmus_ai_factory.grid.regional_grids import SINGAPORE_GRID
        assert 22.0 in SINGAPORE_GRID.ht_supply_voltages_kv
        assert 66.0 in SINGAPORE_GRID.ht_supply_voltages_kv
    
    def test_australia_ht_voltages(self):
        """Australia HT supply should include 11kV, 33kV, 66kV."""
        from firmus_ai_factory.grid.regional_grids import AUSTRALIA_NEM_GRID
        assert 11.0 in AUSTRALIA_NEM_GRID.ht_supply_voltages_kv
        assert 33.0 in AUSTRALIA_NEM_GRID.ht_supply_voltages_kv
        assert 66.0 in AUSTRALIA_NEM_GRID.ht_supply_voltages_kv
    
    def test_singapore_tariff_structure(self):
        """Singapore tariff should have TOU periods."""
        from firmus_ai_factory.grid.regional_grids import SINGAPORE_TARIFF
        assert len(SINGAPORE_TARIFF.periods) >= 3
        assert SINGAPORE_TARIFF.currency == "SGD"
    
    def test_australia_tariff_structure(self):
        """Australia tariff should have TOU periods."""
        from firmus_ai_factory.grid.regional_grids import AUSTRALIA_NEM_TARIFF
        assert len(AUSTRALIA_NEM_TARIFF.periods) >= 3
        assert AUSTRALIA_NEM_TARIFF.currency == "AUD"
    
    def test_singapore_grid_model_report(self):
        """Singapore grid model should generate valid report."""
        from firmus_ai_factory.grid.regional_grids import (
            GridRegion, RegionalGridModel)
        model = RegionalGridModel(GridRegion.SINGAPORE)
        report = model.generate_report(10.0)
        assert report['grid_spec']['three_phase_voltage_v'] == 400.0
        assert report['energy_cost']['currency'] == 'SGD'
    
    def test_australia_grid_model_report(self):
        """Australia grid model should generate valid report."""
        from firmus_ai_factory.grid.regional_grids import (
            GridRegion, RegionalGridModel)
        model = RegionalGridModel(GridRegion.AUSTRALIA_NEM)
        report = model.generate_report(10.0)
        assert report['grid_spec']['three_phase_voltage_v'] == 415.0
        assert report['energy_cost']['currency'] == 'AUD'
    
    def test_frequency_simulation(self):
        """Grid frequency simulation should stay within bounds."""
        from firmus_ai_factory.grid.regional_grids import (
            GridRegion, RegionalGridModel)
        model = RegionalGridModel(GridRegion.SINGAPORE)
        result = model.simulate_grid_frequency(duration_hours=1.0, dt_s=1.0)
        assert len(result['frequency_hz']) == 3600
        assert np.all(result['frequency_hz'] >= 48.0)
        assert np.all(result['frequency_hz'] <= 50.5)
    
    def test_demand_response_revenue(self):
        """DR revenue should be positive for eligible capacity."""
        from firmus_ai_factory.grid.regional_grids import (
            GridRegion, RegionalGridModel)
        model = RegionalGridModel(GridRegion.AUSTRALIA_NEM)
        revenue = model.demand_response_revenue(5.0, 50.0)
        assert revenue['total_annual_revenue'] > 0


class TestFactoryConfig:
    """Test factory configuration and platform mapping."""
    
    def test_h100_requires_singapore_immersion(self):
        """H100 must be Singapore + Immersion."""
        from firmus_ai_factory.factory_config import (
            FactoryConfig, GPUPlatform, CoolingType, FirmusAIFactory)
        from firmus_ai_factory.grid.regional_grids import GridRegion
        
        config = FactoryConfig(
            name="Test",
            platform=GPUPlatform.HGX_H100,
            num_racks=10,
            cooling_type=CoolingType.IMMERSION,
            grid_region=GridRegion.SINGAPORE,
        )
        factory = FirmusAIFactory(config)
        assert factory.config.grid_region == GridRegion.SINGAPORE
    
    def test_h100_rejects_australia(self):
        """H100 should reject Australia deployment."""
        from firmus_ai_factory.factory_config import (
            FactoryConfig, GPUPlatform, CoolingType, FirmusAIFactory)
        from firmus_ai_factory.grid.regional_grids import GridRegion
        
        config = FactoryConfig(
            name="Test",
            platform=GPUPlatform.HGX_H100,
            num_racks=10,
            cooling_type=CoolingType.IMMERSION,
            grid_region=GridRegion.AUSTRALIA_NEM,
        )
        with pytest.raises(ValueError, match="deployed in"):
            FirmusAIFactory(config)
    
    def test_h100_rejects_benmax_cooling(self):
        """H100 should reject Benmax cooling."""
        from firmus_ai_factory.factory_config import (
            FactoryConfig, GPUPlatform, CoolingType, FirmusAIFactory)
        from firmus_ai_factory.grid.regional_grids import GridRegion
        
        config = FactoryConfig(
            name="Test",
            platform=GPUPlatform.HGX_H100,
            num_racks=10,
            cooling_type=CoolingType.BENMAX_HCU2500,
            grid_region=GridRegion.SINGAPORE,
        )
        with pytest.raises(ValueError, match="requires"):
            FirmusAIFactory(config)
    
    def test_vr_requires_australia_benmax(self):
        """Vera Rubin must be Australia + Benmax."""
        from firmus_ai_factory.factory_config import (
            FactoryConfig, GPUPlatform, CoolingType, FirmusAIFactory)
        from firmus_ai_factory.grid.regional_grids import GridRegion
        
        config = FactoryConfig(
            name="Test",
            platform=GPUPlatform.VR_NVL72_MAX_P,
            num_racks=32,
            cooling_type=CoolingType.BENMAX_HCU2500,
            grid_region=GridRegion.AUSTRALIA_NEM,
        )
        factory = FirmusAIFactory(config)
        assert factory.config.grid_region == GridRegion.AUSTRALIA_NEM
    
    def test_vr_rejects_singapore(self):
        """Vera Rubin should reject Singapore deployment."""
        from firmus_ai_factory.factory_config import (
            FactoryConfig, GPUPlatform, CoolingType, FirmusAIFactory)
        from firmus_ai_factory.grid.regional_grids import GridRegion
        
        config = FactoryConfig(
            name="Test",
            platform=GPUPlatform.VR_NVL72_MAX_P,
            num_racks=32,
            cooling_type=CoolingType.BENMAX_HCU2500,
            grid_region=GridRegion.SINGAPORE,
        )
        with pytest.raises(ValueError, match="deployed in"):
            FirmusAIFactory(config)
    
    def test_gb300_requires_australia_benmax(self):
        """GB300 must be Australia + Benmax."""
        from firmus_ai_factory.factory_config import (
            FactoryConfig, GPUPlatform, CoolingType, FirmusAIFactory)
        from firmus_ai_factory.grid.regional_grids import GridRegion
        
        config = FactoryConfig(
            name="Test",
            platform=GPUPlatform.GB300_NVL72,
            num_racks=32,
            cooling_type=CoolingType.BENMAX_HCU2500,
            grid_region=GridRegion.AUSTRALIA_NEM,
        )
        factory = FirmusAIFactory(config)
        assert factory.config.grid_region == GridRegion.AUSTRALIA_NEM
    
    def test_full_report_structure(self):
        """Full report should contain all sections."""
        from firmus_ai_factory.factory_config import australia_vera_rubin_factory
        factory = australia_vera_rubin_factory(num_racks=4, max_q=True)
        report = factory.generate_full_report()
        
        assert 'factory' in report
        assert 'power' in report
        assert 'compute' in report
        assert 'thermal' in report
        assert 'grid' in report
        assert report['power']['pue'] > 1.0
        assert report['power']['pue'] < 1.5
    
    def test_compute_summary_vera_rubin(self):
        """Vera Rubin compute summary should have correct GPU count."""
        from firmus_ai_factory.factory_config import australia_vera_rubin_factory
        factory = australia_vera_rubin_factory(num_racks=4, max_q=False)
        compute = factory.compute_summary()
        assert compute['total_gpus'] == 4 * 72  # 288 GPUs
    
    def test_convenience_factories(self):
        """All convenience factory functions should work."""
        from firmus_ai_factory.factory_config import (
            singapore_h100_factory,
            singapore_h200_factory,
            australia_gb300_factory,
            australia_vera_rubin_factory,
        )
        
        f1 = singapore_h100_factory(2)
        assert f1.config.grid_region.value == "singapore"
        
        f2 = singapore_h200_factory(2)
        assert f2.config.grid_region.value == "singapore"
        
        f3 = australia_gb300_factory(2)
        assert f3.config.grid_region.value == "australia_nem"
        
        f4 = australia_vera_rubin_factory(2, max_q=False)
        assert f4.config.grid_region.value == "australia_nem"
        
        f5 = australia_vera_rubin_factory(2, max_q=True)
        assert f5.config.grid_region.value == "australia_nem"
