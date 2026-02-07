"""Unit tests for site-aware environmental conditions.

Tests ASHRAE climatic data, grid energy mix, site configurations,
site-aware factory creation, and thermal analysis.
"""

import pytest
import math
from firmus_ai_factory.environment.site_conditions import (
    ASHRAEClimateZone,
    ASHRAEConditions,
    GridEnergyMix,
    GridConnection,
    NEM_Region,
    SiteConfig,
    SiteStatus,
    ASHRAE_LAUNCESTON,
    ASHRAE_MELBOURNE,
    ASHRAE_CANBERRA,
    ASHRAE_SYDNEY,
    ASHRAE_ROBERTSTOWN,
    ASHRAE_ALICE_SPRINGS,
    ASHRAE_BATAM,
    GRID_MIX_TASMANIA,
    GRID_MIX_SOUTH_AUSTRALIA,
    GRID_MIX_VICTORIA,
    GRID_MIX_NSW,
    GRID_MIX_NT_GAS,
    GRID_MIX_BATAM,
    ALL_SITES,
    get_site,
    get_sites_by_region,
    get_sites_by_gpu,
    get_sites_by_provider,
    portfolio_summary,
)


# =============================================================================
# ASHRAE Conditions Tests
# =============================================================================

class TestASHRAEConditions:
    """Test ASHRAE climatic design conditions."""
    
    def test_launceston_coordinates(self):
        """Verify Launceston coordinates and elevation."""
        assert ASHRAE_LAUNCESTON.latitude == pytest.approx(-41.53, abs=0.1)
        assert ASHRAE_LAUNCESTON.longitude == pytest.approx(147.20, abs=0.1)
        assert ASHRAE_LAUNCESTON.elevation_m == pytest.approx(172, abs=10)
    
    def test_launceston_cooling_design(self):
        """Verify Launceston 0.4% cooling design conditions."""
        assert ASHRAE_LAUNCESTON.cooling_04_db == pytest.approx(29.8, abs=0.5)
        assert ASHRAE_LAUNCESTON.cooling_04_mcwb == pytest.approx(18.4, abs=0.5)
    
    def test_canberra_cooling_design(self):
        """Verify Canberra 0.4% cooling design conditions."""
        assert ASHRAE_CANBERRA.cooling_04_db == pytest.approx(34.5, abs=0.5)
        assert ASHRAE_CANBERRA.cooling_04_mcwb == pytest.approx(18.1, abs=0.5)
    
    def test_robertstown_extreme_conditions(self):
        """Verify Robertstown extreme conditions (hot dry climate)."""
        assert ASHRAE_ROBERTSTOWN.extreme_max_db == pytest.approx(42.5, abs=1.0)
        assert ASHRAE_ROBERTSTOWN.extreme_min_db == pytest.approx(0.6, abs=1.0)
    
    def test_alice_springs_extreme_heat(self):
        """Verify Alice Springs extreme heat (hottest site)."""
        assert ASHRAE_ALICE_SPRINGS.extreme_max_db == pytest.approx(45.6, abs=1.0)
        assert ASHRAE_ALICE_SPRINGS.cooling_04_db == pytest.approx(40.5, abs=1.0)
    
    def test_batam_tropical_conditions(self):
        """Verify Batam tropical conditions (high humidity)."""
        assert ASHRAE_BATAM.climate_zone == ASHRAEClimateZone.ZONE_1A_VERY_HOT_HUMID
        assert ASHRAE_BATAM.annual_avg_db == pytest.approx(27.5, abs=0.5)
        # Tropical: minimal temperature variation
        temp_range = max(ASHRAE_BATAM.monthly_avg_db) - min(ASHRAE_BATAM.monthly_avg_db)
        assert temp_range < 3.0  # Less than 3°C annual variation
    
    def test_monthly_avg_db_length(self):
        """All ASHRAE conditions must have 12 monthly values."""
        for ashrae in [ASHRAE_LAUNCESTON, ASHRAE_MELBOURNE, ASHRAE_CANBERRA,
                       ASHRAE_SYDNEY, ASHRAE_ROBERTSTOWN, ASHRAE_ALICE_SPRINGS,
                       ASHRAE_BATAM]:
            assert len(ashrae.monthly_avg_db) == 12
    
    def test_extreme_bounds(self):
        """Extreme max must exceed cooling design; extreme min below heating."""
        for ashrae in [ASHRAE_LAUNCESTON, ASHRAE_MELBOURNE, ASHRAE_CANBERRA,
                       ASHRAE_SYDNEY, ASHRAE_ROBERTSTOWN, ASHRAE_ALICE_SPRINGS]:
            assert ashrae.extreme_max_db >= ashrae.cooling_04_db
            assert ashrae.extreme_min_db <= ashrae.heating_99_db
    
    def test_get_ambient_temp_avg(self):
        """get_ambient_temp('avg') returns monthly average."""
        jan_temp = ASHRAE_CANBERRA.get_ambient_temp(1, "avg")
        assert jan_temp == pytest.approx(21.5, abs=0.1)
    
    def test_get_ambient_temp_design(self):
        """get_ambient_temp('design') returns 0.4% cooling DB."""
        design_temp = ASHRAE_CANBERRA.get_ambient_temp(1, "design")
        assert design_temp == pytest.approx(34.5, abs=0.5)
    
    def test_get_ambient_temp_extreme(self):
        """get_ambient_temp('extreme') returns extreme max DB."""
        extreme_temp = ASHRAE_CANBERRA.get_ambient_temp(1, "extreme")
        assert extreme_temp == pytest.approx(37.9, abs=0.5)
    
    def test_annual_temperature_profile_length(self):
        """Annual temperature profile must have 8760 hourly values."""
        profile = ASHRAE_LAUNCESTON.annual_temperature_profile()
        assert len(profile) == 8760
    
    def test_annual_temperature_profile_range(self):
        """Profile temperatures must be within physical bounds."""
        profile = ASHRAE_CANBERRA.annual_temperature_profile()
        assert min(profile) > -15  # Not unreasonably cold
        assert max(profile) < 50   # Not unreasonably hot
    
    def test_free_cooling_hours_launceston(self):
        """Launceston (cool climate) should have many free cooling hours."""
        hours = ASHRAE_LAUNCESTON.free_cooling_hours(threshold_c=27.0)
        assert hours > 7000  # Most of the year
    
    def test_free_cooling_hours_alice_springs(self):
        """Alice Springs (hot climate) should have fewer free cooling hours."""
        hours = ASHRAE_ALICE_SPRINGS.free_cooling_hours(threshold_c=27.0)
        assert hours < 6000  # Less than Launceston
    
    def test_cooling_energy_indicator(self):
        """Hotter sites should have higher cooling energy indicators."""
        launceston_cdh = ASHRAE_LAUNCESTON.cooling_energy_indicator(27.0)
        alice_cdh = ASHRAE_ALICE_SPRINGS.cooling_energy_indicator(27.0)
        assert alice_cdh > launceston_cdh


# =============================================================================
# Grid Energy Mix Tests
# =============================================================================

class TestGridEnergyMix:
    """Test grid energy mix configurations."""
    
    def test_tasmania_renewable_fraction(self):
        """Tasmania should have >90% renewable fraction."""
        assert GRID_MIX_TASMANIA.renewable_fraction > 0.90
    
    def test_tasmania_low_carbon(self):
        """Tasmania should have very low carbon intensity."""
        assert GRID_MIX_TASMANIA.carbon_intensity_kg_mwh < 50
    
    def test_south_australia_renewable(self):
        """South Australia should have significant renewable fraction."""
        assert GRID_MIX_SOUTH_AUSTRALIA.renewable_fraction > 0.60
    
    def test_nsw_coal_heavy(self):
        """NSW should still have significant coal percentage."""
        assert GRID_MIX_NSW.coal_pct > 40
    
    def test_nt_gas_dominant(self):
        """Northern Territory should be gas-dominated."""
        assert GRID_MIX_NT_GAS.gas_pct > 90
        assert GRID_MIX_NT_GAS.nem_region is None  # NT not in NEM
    
    def test_annual_carbon_calculation(self):
        """Verify annual carbon calculation."""
        carbon = GRID_MIX_TASMANIA.annual_carbon_tonnes(10.0)  # 10 MW
        expected = 10.0 * 8760 * 30.0 / 1000  # MW * hours * kg/MWh / 1000
        assert carbon == pytest.approx(expected, rel=0.01)
    
    def test_energy_mix_sums(self):
        """Energy mix percentages should sum to approximately 100%."""
        for mix in [GRID_MIX_TASMANIA, GRID_MIX_SOUTH_AUSTRALIA,
                    GRID_MIX_VICTORIA, GRID_MIX_NSW, GRID_MIX_NT_GAS,
                    GRID_MIX_BATAM]:
            total = (mix.hydro_pct + mix.wind_pct + mix.solar_pct +
                    mix.gas_pct + mix.coal_pct + mix.other_renewable_pct)
            assert total == pytest.approx(100.0, abs=2.0)


# =============================================================================
# Site Configuration Tests
# =============================================================================

class TestSiteConfig:
    """Test site configurations from Firmus-Southgate Master Plan."""
    
    def test_all_sites_registered(self):
        """Verify all expected sites are in the registry."""
        expected_codes = [
            "LN2/LN3", "WV1", "GT1-2", "RT1", "BT1-3",
            "PG", "BK2", "HU4", "HU5", "HU6", "BE1", "MP1",
        ]
        for code in expected_codes:
            assert code in ALL_SITES, f"Site {code} not found"
    
    def test_get_site_valid(self):
        """get_site returns correct site for valid code."""
        site = get_site("RT1")
        assert site.dc_code == "RT1"
        assert site.campus == "Robertstown"
    
    def test_get_site_invalid(self):
        """get_site raises KeyError for invalid code."""
        with pytest.raises(KeyError):
            get_site("INVALID")
    
    def test_sites_by_region_tasmania(self):
        """Should find multiple sites in Tasmania."""
        tas_sites = get_sites_by_region("Tasmania")
        assert len(tas_sites) >= 3  # LN2/LN3, WV1, GT1-2
    
    def test_sites_by_gpu_h200(self):
        """Should find H200 sites (Singapore immersion)."""
        h200_sites = get_sites_by_gpu("H200")
        assert len(h200_sites) >= 2  # LN2/LN3, BK2
    
    def test_sites_by_provider_firmus(self):
        """Should find multiple Firmus-operated sites."""
        firmus_sites = get_sites_by_provider("Firmus")
        assert len(firmus_sites) >= 5
    
    def test_sites_by_provider_cdc(self):
        """Should find CDC-operated sites."""
        cdc_sites = get_sites_by_provider("CDC")
        assert len(cdc_sites) >= 4  # BK2, HU4, HU5, HU6, BE1, MP1
    
    def test_site_it_power_calculation(self):
        """IT power should account for infrastructure headroom."""
        site = get_site("RT1")
        expected_it = site.gross_mw / (1 + site.infra_headroom_pct / 100)
        assert site.it_power_mw == pytest.approx(expected_it, rel=0.01)
    
    def test_site_cooling_load(self):
        """Cooling load should be positive and proportional to PUE."""
        site = get_site("BK2")
        assert site.cooling_load_kw > 0
        # Cooling = IT * (PUE - 1)
        expected = site.it_power_mw * 1000 * (site.pue - 1)
        assert site.cooling_load_kw == pytest.approx(expected, rel=0.01)
    
    def test_ambient_impact_on_cooling(self):
        """Ambient impact analysis should return valid results."""
        site = get_site("RT1")
        impact = site.ambient_impact_on_cooling(1)  # January (summer)
        
        assert 'ambient_temp_c' in impact
        assert 'coolant_supply_temp_c' in impact
        assert 'thermal_margin_c' in impact
        assert 'free_cooling_available' in impact
        assert 'cop_estimate' in impact
        assert impact['ambient_temp_c'] > 0
    
    def test_annual_cooling_analysis(self):
        """Annual cooling analysis should cover all 12 months."""
        site = get_site("LN2/LN3")
        analysis = site.annual_cooling_analysis()
        
        assert len(analysis['monthly_analysis']) == 12
        assert analysis['free_cooling_months'] >= 0
        assert analysis['free_cooling_months'] <= 12
        assert analysis['avg_annual_cop'] > 0
    
    def test_site_report_structure(self):
        """Site report should contain all required sections."""
        site = get_site("RT1")
        report = site.generate_site_report()
        
        assert 'site' in report
        assert 'capacity' in report
        assert 'environment' in report
        assert 'cooling' in report
        assert 'grid' in report
        
        assert report['site']['dc_code'] == "RT1"
        assert report['capacity']['gross_mw'] > 0
        assert report['environment']['annual_avg_temp_c'] > 0


# =============================================================================
# Portfolio Summary Tests
# =============================================================================

class TestPortfolioSummary:
    """Test portfolio-level summary calculations."""
    
    def test_portfolio_total_sites(self):
        """Portfolio should include all registered sites."""
        summary = portfolio_summary()
        assert summary['total_sites'] == len(ALL_SITES)
    
    def test_portfolio_total_mw(self):
        """Total MW should be sum of all site gross_mw."""
        summary = portfolio_summary()
        expected = sum(s.gross_mw for s in ALL_SITES.values())
        assert summary['total_gross_mw'] == pytest.approx(expected, rel=0.01)
    
    def test_portfolio_total_gpus(self):
        """Total GPUs should be sum of all site num_gpus."""
        summary = portfolio_summary()
        expected = sum(s.num_gpus for s in ALL_SITES.values())
        assert summary['total_gpus'] == expected
    
    def test_portfolio_weighted_pue(self):
        """Weighted PUE should be between 1.0 and 2.0."""
        summary = portfolio_summary()
        assert 1.0 < summary['weighted_avg_pue'] < 2.0
    
    def test_portfolio_gpu_breakdown(self):
        """GPU breakdown should cover all GPU series."""
        summary = portfolio_summary()
        assert 'gpu_breakdown' in summary
        assert len(summary['gpu_breakdown']) >= 2  # At least H200 and GB300
    
    def test_portfolio_region_breakdown(self):
        """Region breakdown should cover all regions."""
        summary = portfolio_summary()
        assert 'region_breakdown' in summary
        assert len(summary['region_breakdown']) >= 3  # Tasmania, SA, Canberra, etc.


# =============================================================================
# Climate Zone Ranking Tests
# =============================================================================

class TestClimateZoneRanking:
    """Test that sites rank correctly by thermal conditions."""
    
    def test_launceston_cooler_than_robertstown(self):
        """Launceston (marine) should be cooler than Robertstown (dry)."""
        assert (ASHRAE_LAUNCESTON.annual_avg_db < 
                ASHRAE_ROBERTSTOWN.annual_avg_db)
    
    def test_alice_springs_hottest_australian(self):
        """Alice Springs should be the hottest Australian site."""
        aus_sites = [ASHRAE_LAUNCESTON, ASHRAE_MELBOURNE, ASHRAE_CANBERRA,
                     ASHRAE_SYDNEY, ASHRAE_ROBERTSTOWN, ASHRAE_ALICE_SPRINGS]
        hottest = max(aus_sites, key=lambda a: a.cooling_04_db)
        assert hottest is ASHRAE_ALICE_SPRINGS
    
    def test_launceston_most_free_cooling(self):
        """Launceston should have the most free cooling hours among AU sites."""
        aus_ashrae = [ASHRAE_LAUNCESTON, ASHRAE_MELBOURNE, ASHRAE_CANBERRA,
                      ASHRAE_SYDNEY, ASHRAE_ROBERTSTOWN, ASHRAE_ALICE_SPRINGS]
        hours = [(a.station_name, a.free_cooling_hours()) for a in aus_ashrae]
        best = max(hours, key=lambda x: x[1])
        assert best[0] == ASHRAE_LAUNCESTON.station_name
    
    def test_southern_hemisphere_seasons(self):
        """Australian sites should be hottest in Jan/Feb (southern summer)."""
        for ashrae in [ASHRAE_LAUNCESTON, ASHRAE_MELBOURNE, ASHRAE_CANBERRA,
                       ASHRAE_SYDNEY, ASHRAE_ROBERTSTOWN, ASHRAE_ALICE_SPRINGS]:
            hottest_idx = ashrae.monthly_avg_db.index(max(ashrae.monthly_avg_db))
            # Hottest month should be Dec(11), Jan(0), or Feb(1)
            assert hottest_idx in [0, 1, 11], (
                f"{ashrae.station_name}: hottest month index {hottest_idx}")
    
    def test_batam_no_seasons(self):
        """Batam (tropical) should have minimal seasonal variation."""
        variation = max(ASHRAE_BATAM.monthly_avg_db) - min(ASHRAE_BATAM.monthly_avg_db)
        assert variation < 3.0  # Less than 3°C
