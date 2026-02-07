"""Site-Aware Environmental Conditions for Firmus AI Factory.

Defines ASHRAE climatic design conditions, local grid mix, and
site-specific parameters for each Firmus data center location.

Data sources:
    - ASHRAE 2021 Handbook of Fundamentals (Climatic Design Conditions)
    - Firmus-Southgate Master Capacity Planning spreadsheet
    - Bureau of Meteorology (BOM) climate zone classifications
    - AEMO NEM generation mix data
    - EMA Singapore energy statistics

Each site includes:
    - Geographic coordinates and elevation
    - ASHRAE cooling/heating design temperatures
    - Monthly average dry bulb temperatures
    - Extreme temperature conditions
    - Local grid energy mix (renewable fraction, carbon intensity)
    - Data center specifications (PUE, rack count, GPU platform)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math


# =============================================================================
# ASHRAE Climate Classification
# =============================================================================

class ASHRAEClimateZone(Enum):
    """ASHRAE climate zone classifications relevant to Firmus sites."""
    ZONE_1A_VERY_HOT_HUMID = "1A"      # Batam
    ZONE_3B_WARM_DRY = "3B"            # Robertstown, Pine Gap
    ZONE_3C_MARINE = "3C"              # Launceston/Tasmania
    ZONE_4A_MIXED_HUMID = "4A"         # Canberra
    ZONE_5A_WARM_TEMPERATE = "5A"      # Sydney (Marsden Park)
    ZONE_TEMPERATE = "Temperate"       # Melbourne


class NEM_Region(Enum):
    """Australian NEM regions for grid dispatch and pricing."""
    TAS = "TAS1"     # Tasmania
    SA = "SA1"       # South Australia
    VIC = "VIC1"     # Victoria
    NSW = "NSW1"     # New South Wales
    QLD = "QLD1"     # Queensland


class GridConnection(Enum):
    """Grid connection type."""
    GRID_HYDRO = "grid_hydro"
    GRID_MIXED = "grid_mixed"
    GRID_GAS_RENEWABLE = "grid_gas_renewable"
    GAS_GENERATION = "gas_generation"
    GRID_STANDARD = "grid_standard"


class SiteStatus(Enum):
    """Site development status."""
    OPERATIONAL = "operational"
    UNDER_CONSTRUCTION = "under_construction"
    IN_BUILD = "in_build"
    CONTRACTING = "contracting"
    PLANNED = "planned"
    GAS_SOURCING = "gas_sourcing"
    ROFR = "rofr"  # Right of First Refusal


# =============================================================================
# ASHRAE Climatic Design Conditions
# =============================================================================

@dataclass
class ASHRAEConditions:
    """ASHRAE 2021 climatic design conditions for a site.
    
    All temperatures in degrees Celsius.
    Based on ASHRAE 2021 Handbook of Fundamentals, Chapter 14.
    
    Attributes:
        climate_zone: ASHRAE climate zone classification
        wmo_station: WMO weather station identifier
        station_name: Weather station name
        latitude: Latitude in decimal degrees (negative for South)
        longitude: Longitude in decimal degrees (positive for East)
        elevation_m: Elevation above sea level in meters
        std_pressure_kpa: Standard atmospheric pressure at elevation
        
        cooling_04_db: 0.4% annual cooling design dry bulb (°C)
        cooling_04_mcwb: Mean coincident wet bulb at 0.4% cooling DB (°C)
        cooling_1_db: 1% annual cooling design dry bulb (°C)
        cooling_1_mcwb: Mean coincident wet bulb at 1% cooling DB (°C)
        
        heating_996_db: 99.6% annual heating design dry bulb (°C)
        heating_99_db: 99% annual heating design dry bulb (°C)
        
        extreme_max_db: Mean extreme maximum dry bulb (°C)
        extreme_min_db: Mean extreme minimum dry bulb (°C)
        extreme_max_wb: Extreme maximum wet bulb (°C)
        
        annual_avg_db: Annual average dry bulb temperature (°C)
        monthly_avg_db: Monthly average dry bulb temperatures [Jan..Dec] (°C)
        
        hdd_18_3: Annual heating degree days (base 18.3°C)
        cdd_18_3: Annual cooling degree days (base 18.3°C)
        
        hottest_month: Hottest month (1=Jan, 12=Dec)
        hottest_month_db_range: Diurnal temperature range in hottest month (°C)
    """
    climate_zone: ASHRAEClimateZone
    wmo_station: str
    station_name: str
    latitude: float
    longitude: float
    elevation_m: float
    std_pressure_kpa: float
    
    # Cooling design conditions
    cooling_04_db: float
    cooling_04_mcwb: float
    cooling_1_db: float
    cooling_1_mcwb: float
    
    # Heating design conditions
    heating_996_db: float
    heating_99_db: float
    
    # Extreme conditions
    extreme_max_db: float
    extreme_min_db: float
    extreme_max_wb: float
    
    # Average conditions
    annual_avg_db: float
    monthly_avg_db: List[float]  # 12 values, Jan-Dec
    
    # Degree days
    hdd_18_3: float
    cdd_18_3: float
    
    # Hottest month
    hottest_month: int
    hottest_month_db_range: float
    
    def get_ambient_temp(self, month: int, percentile: str = "avg") -> float:
        """Get ambient temperature for a given month.
        
        Args:
            month: Month number (1-12)
            percentile: 'avg' for monthly average, 'design' for 0.4% cooling,
                       'extreme' for extreme max
        
        Returns:
            Temperature in °C
        """
        if percentile == "avg":
            return self.monthly_avg_db[month - 1]
        elif percentile == "design":
            return self.cooling_04_db
        elif percentile == "extreme":
            return self.extreme_max_db
        else:
            raise ValueError(f"Unknown percentile: {percentile}")
    
    def annual_temperature_profile(self, hours: int = 8760) -> List[float]:
        """Generate hourly temperature profile using sinusoidal model.
        
        Creates a synthetic hourly temperature profile based on monthly
        averages with diurnal variation. Uses cosine interpolation between
        monthly averages and adds diurnal swing.
        
        Args:
            hours: Number of hours to generate (default 8760 = 1 year)
        
        Returns:
            List of hourly temperatures in °C
        """
        temps = []
        for h in range(hours):
            day_of_year = h // 24
            hour_of_day = h % 24
            
            # Monthly interpolation (cosine for smooth transitions)
            month_frac = day_of_year / 30.44  # Average days per month
            month_idx = int(month_frac) % 12
            next_month_idx = (month_idx + 1) % 12
            frac = month_frac - int(month_frac)
            
            # Cosine interpolation between months
            weight = (1 - math.cos(math.pi * frac)) / 2
            base_temp = (self.monthly_avg_db[month_idx] * (1 - weight) + 
                        self.monthly_avg_db[next_month_idx] * weight)
            
            # Diurnal variation (±half of DB range)
            diurnal_range = self.hottest_month_db_range * 0.7  # Scale for non-peak months
            # Peak at 15:00, minimum at 06:00
            diurnal = (diurnal_range / 2) * math.cos(
                2 * math.pi * (hour_of_day - 15) / 24)
            
            temps.append(base_temp + diurnal)
        
        return temps
    
    def free_cooling_hours(self, threshold_c: float = 27.0) -> int:
        """Estimate annual hours where free cooling is available.
        
        Args:
            threshold_c: Maximum ambient temperature for free cooling (°C)
        
        Returns:
            Estimated number of free cooling hours per year
        """
        profile = self.annual_temperature_profile()
        return sum(1 for t in profile if t <= threshold_c)
    
    def cooling_energy_indicator(self, indoor_setpoint_c: float = 27.0) -> float:
        """Calculate cooling energy indicator (degree-hours above setpoint).
        
        Args:
            indoor_setpoint_c: Indoor temperature setpoint (°C)
        
        Returns:
            Annual cooling degree-hours (°C·h)
        """
        profile = self.annual_temperature_profile()
        return sum(max(0, t - indoor_setpoint_c) for t in profile)


# =============================================================================
# Local Grid Energy Mix
# =============================================================================

@dataclass
class GridEnergyMix:
    """Local grid energy mix and carbon intensity.
    
    Attributes:
        nem_region: NEM region (for Australian sites)
        connection_type: Grid connection type
        hydro_pct: Hydroelectric percentage
        wind_pct: Wind percentage
        solar_pct: Solar percentage
        gas_pct: Gas percentage
        coal_pct: Coal percentage
        other_renewable_pct: Other renewable percentage
        carbon_intensity_kg_mwh: Grid carbon intensity (kg CO2/MWh)
        renewable_fraction: Total renewable fraction (0-1)
        grid_reliability_pct: Grid reliability percentage
        spot_price_avg_aud_mwh: Average spot price (AUD/MWh)
    """
    nem_region: Optional[NEM_Region]
    connection_type: GridConnection
    hydro_pct: float = 0.0
    wind_pct: float = 0.0
    solar_pct: float = 0.0
    gas_pct: float = 0.0
    coal_pct: float = 0.0
    other_renewable_pct: float = 0.0
    carbon_intensity_kg_mwh: float = 0.0
    renewable_fraction: float = 0.0
    grid_reliability_pct: float = 99.99
    spot_price_avg_aud_mwh: float = 0.0
    
    def __post_init__(self):
        """Calculate renewable fraction from components."""
        self.renewable_fraction = (
            self.hydro_pct + self.wind_pct + 
            self.solar_pct + self.other_renewable_pct
        ) / 100.0
    
    def annual_carbon_tonnes(self, power_mw: float) -> float:
        """Calculate annual carbon emissions.
        
        Args:
            power_mw: Average power consumption in MW
        
        Returns:
            Annual CO2 emissions in tonnes
        """
        annual_mwh = power_mw * 8760
        return annual_mwh * self.carbon_intensity_kg_mwh / 1000.0
    
    def annual_energy_cost_aud(self, power_mw: float) -> float:
        """Estimate annual energy cost.
        
        Args:
            power_mw: Average power consumption in MW
        
        Returns:
            Annual energy cost in AUD
        """
        annual_mwh = power_mw * 8760
        return annual_mwh * self.spot_price_avg_aud_mwh


# =============================================================================
# Site Configuration
# =============================================================================

@dataclass
class SiteConfig:
    """Complete site configuration for a Firmus data center.
    
    Combines ASHRAE environmental conditions, grid energy mix,
    and data center specifications.
    
    Attributes:
        dc_code: Data center code (e.g., 'LN2', 'RT1')
        campus: Campus name
        region: Geographic region
        provider: Infrastructure provider ('Firmus', 'CDC', 'DayOne')
        status: Development status
        ashrae: ASHRAE climatic design conditions
        grid_mix: Local grid energy mix
        gpu_series: GPU series ('H200', 'GB300', 'VR')
        nv_code: NVIDIA platform code ('HGX', 'Oberon', 'Kyber')
        gross_mw: Gross power capacity (MW)
        pue: Power Usage Effectiveness
        num_racks: Number of GPU racks
        num_gpus: Total number of GPUs
        rack_power_kw: Power per rack (kW)
        infra_headroom_pct: Infrastructure headroom percentage
    """
    dc_code: str
    campus: str
    region: str
    provider: str
    status: SiteStatus
    ashrae: ASHRAEConditions
    grid_mix: GridEnergyMix
    gpu_series: str
    nv_code: str
    gross_mw: float
    pue: float
    num_racks: int
    num_gpus: int
    rack_power_kw: float
    infra_headroom_pct: float = 14.0
    
    @property
    def it_power_mw(self) -> float:
        """Net IT power after infrastructure headroom."""
        return self.gross_mw / (1 + self.infra_headroom_pct / 100)
    
    @property
    def cooling_load_kw(self) -> float:
        """Estimated cooling load based on PUE."""
        it_kw = self.it_power_mw * 1000
        return it_kw * (self.pue - 1)
    
    def ambient_impact_on_cooling(self, month: int) -> Dict:
        """Analyze ambient temperature impact on cooling for a given month.
        
        Args:
            month: Month number (1-12)
        
        Returns:
            Dict with cooling analysis for the month
        """
        ambient = self.ashrae.get_ambient_temp(month)
        design_ambient = self.ashrae.cooling_04_db
        extreme_ambient = self.ashrae.extreme_max_db
        
        # CDU approach temperature (typical 5-8°C for liquid cooling)
        approach_temp = 7.0
        
        # Coolant supply temperature depends on ambient
        coolant_supply = ambient + approach_temp
        
        # NVIDIA max inlet temperature for liquid cooling
        nvidia_max_inlet = 45.0  # °C per VR NVL72 spec
        
        # Available thermal margin
        thermal_margin = nvidia_max_inlet - coolant_supply
        
        # Free cooling availability (no mechanical chiller needed)
        free_cooling_available = coolant_supply <= nvidia_max_inlet
        
        # Cooling efficiency factor (COP estimate)
        if ambient < 15:
            cop_estimate = 20.0  # Excellent free cooling
        elif ambient < 25:
            cop_estimate = 10.0  # Good free cooling
        elif ambient < 35:
            cop_estimate = 5.0   # Partial mechanical cooling
        else:
            cop_estimate = 3.0   # Full mechanical cooling
        
        return {
            'month': month,
            'ambient_temp_c': ambient,
            'coolant_supply_temp_c': coolant_supply,
            'thermal_margin_c': thermal_margin,
            'free_cooling_available': free_cooling_available,
            'cop_estimate': cop_estimate,
            'design_ambient_c': design_ambient,
            'extreme_ambient_c': extreme_ambient,
            'pue_impact': 1.0 + (1.0 / cop_estimate) if cop_estimate > 0 else 2.0,
        }
    
    def annual_cooling_analysis(self) -> Dict:
        """Generate annual cooling analysis across all months.
        
        Returns:
            Dict with monthly cooling analysis and annual summary
        """
        monthly = [self.ambient_impact_on_cooling(m) for m in range(1, 13)]
        
        free_cooling_months = sum(
            1 for m in monthly if m['free_cooling_available'])
        avg_cop = sum(m['cop_estimate'] for m in monthly) / 12
        min_margin = min(m['thermal_margin_c'] for m in monthly)
        
        # Annual free cooling hours estimate
        free_hours = self.ashrae.free_cooling_hours(
            threshold_c=self.ashrae.cooling_04_db - 7.0)
        
        return {
            'site': self.dc_code,
            'campus': self.campus,
            'monthly_analysis': monthly,
            'free_cooling_months': free_cooling_months,
            'avg_annual_cop': avg_cop,
            'min_thermal_margin_c': min_margin,
            'free_cooling_hours_per_year': free_hours,
            'free_cooling_pct': free_hours / 8760 * 100,
        }
    
    def generate_site_report(self) -> Dict:
        """Generate comprehensive site environmental report.
        
        Returns:
            Dict with complete site analysis
        """
        cooling = self.annual_cooling_analysis()
        carbon = self.grid_mix.annual_carbon_tonnes(self.it_power_mw)
        energy_cost = self.grid_mix.annual_energy_cost_aud(self.gross_mw)
        
        return {
            'site': {
                'dc_code': self.dc_code,
                'campus': self.campus,
                'region': self.region,
                'provider': self.provider,
                'status': self.status.value,
                'gpu_series': self.gpu_series,
                'nv_code': self.nv_code,
            },
            'capacity': {
                'gross_mw': self.gross_mw,
                'it_power_mw': self.it_power_mw,
                'pue': self.pue,
                'num_racks': self.num_racks,
                'num_gpus': self.num_gpus,
                'rack_power_kw': self.rack_power_kw,
            },
            'environment': {
                'climate_zone': self.ashrae.climate_zone.value,
                'annual_avg_temp_c': self.ashrae.annual_avg_db,
                'design_cooling_temp_c': self.ashrae.cooling_04_db,
                'extreme_max_temp_c': self.ashrae.extreme_max_db,
                'extreme_min_temp_c': self.ashrae.extreme_min_db,
                'hdd_18_3': self.ashrae.hdd_18_3,
                'cdd_18_3': self.ashrae.cdd_18_3,
            },
            'cooling': {
                'free_cooling_months': cooling['free_cooling_months'],
                'free_cooling_hours': cooling['free_cooling_hours_per_year'],
                'free_cooling_pct': cooling['free_cooling_pct'],
                'avg_cop': cooling['avg_annual_cop'],
                'min_thermal_margin_c': cooling['min_thermal_margin_c'],
            },
            'grid': {
                'connection_type': self.grid_mix.connection_type.value,
                'renewable_fraction': self.grid_mix.renewable_fraction,
                'carbon_intensity_kg_mwh': self.grid_mix.carbon_intensity_kg_mwh,
                'annual_carbon_tonnes': carbon,
                'annual_energy_cost_aud': energy_cost,
            },
        }


# =============================================================================
# Pre-defined ASHRAE Conditions for Each Site
# =============================================================================

# Launceston, Tasmania (LN2, LN3, WV1, GT1-2)
ASHRAE_LAUNCESTON = ASHRAEConditions(
    climate_zone=ASHRAEClimateZone.ZONE_3C_MARINE,
    wmo_station="949680",
    station_name="Launceston Airport",
    latitude=-41.53,
    longitude=147.20,
    elevation_m=172,
    std_pressure_kpa=99.2,
    cooling_04_db=29.8,
    cooling_04_mcwb=18.4,
    cooling_1_db=27.5,
    cooling_1_mcwb=17.0,
    heating_996_db=-0.2,
    heating_99_db=1.1,
    extreme_max_db=35.1,
    extreme_min_db=-3.4,
    extreme_max_wb=21.5,
    annual_avg_db=12.9,
    monthly_avg_db=[18.1, 17.8, 16.1, 13.2, 10.4, 8.1, 7.6, 8.6, 10.2, 12.3, 14.6, 16.5],
    hdd_18_3=2055,
    cdd_18_3=106,
    hottest_month=1,
    hottest_month_db_range=14.4,
)

# Melbourne Laverton (BK2 Brooklyn)
ASHRAE_MELBOURNE = ASHRAEConditions(
    climate_zone=ASHRAEClimateZone.ZONE_TEMPERATE,
    wmo_station="948700",
    station_name="Melbourne Laverton",
    latitude=-37.98,
    longitude=145.10,
    elevation_m=13,
    std_pressure_kpa=101.2,
    cooling_04_db=34.6,
    cooling_04_mcwb=19.3,
    cooling_1_db=31.9,
    cooling_1_mcwb=18.5,
    heating_996_db=2.7,
    heating_99_db=4.0,
    extreme_max_db=39.8,
    extreme_min_db=0.2,
    extreme_max_wb=26.8,
    annual_avg_db=14.8,
    monthly_avg_db=[19.9, 20.1, 18.3, 15.1, 12.8, 10.5, 10.1, 10.8, 12.3, 14.2, 16.2, 18.0],
    hdd_18_3=1541,
    cdd_18_3=259,
    hottest_month=2,
    hottest_month_db_range=10.1,
)

# Canberra (HU4, HU5, HU6, BE1)
ASHRAE_CANBERRA = ASHRAEConditions(
    climate_zone=ASHRAEClimateZone.ZONE_4A_MIXED_HUMID,
    wmo_station="949260",
    station_name="Canberra Airport",
    latitude=-35.31,
    longitude=149.20,
    elevation_m=578,
    std_pressure_kpa=94.58,
    cooling_04_db=34.5,
    cooling_04_mcwb=18.1,
    cooling_1_db=32.2,
    cooling_1_mcwb=17.6,
    heating_996_db=-3.6,
    heating_99_db=-2.4,
    extreme_max_db=37.9,
    extreme_min_db=-5.8,
    extreme_max_wb=24.1,
    annual_avg_db=13.6,
    monthly_avg_db=[21.5, 20.5, 17.9, 13.7, 9.7, 7.1, 6.2, 7.3, 10.3, 13.4, 16.8, 19.5],
    hdd_18_3=2026,
    cdd_18_3=304,
    hottest_month=1,
    hottest_month_db_range=14.4,
)

# Sydney / Marsden Park (MP1)
ASHRAE_SYDNEY = ASHRAEConditions(
    climate_zone=ASHRAEClimateZone.ZONE_5A_WARM_TEMPERATE,
    wmo_station="947670",
    station_name="Sydney Airport",
    latitude=-33.95,
    longitude=151.17,
    elevation_m=5,
    std_pressure_kpa=101.3,
    cooling_04_db=33.3,
    cooling_04_mcwb=19.5,
    cooling_1_db=30.9,
    cooling_1_mcwb=19.0,
    heating_996_db=6.7,
    heating_99_db=7.6,
    extreme_max_db=39.5,
    extreme_min_db=4.9,
    extreme_max_wb=26.3,
    annual_avg_db=18.6,
    monthly_avg_db=[23.5, 23.2, 22.0, 19.1, 16.3, 13.9, 13.1, 14.1, 16.7, 18.7, 20.3, 22.2],
    hdd_18_3=636,
    cdd_18_3=717,
    hottest_month=1,
    hottest_month_db_range=6.9,
)

# Robertstown, South Australia (RT1-12)
ASHRAE_ROBERTSTOWN = ASHRAEConditions(
    climate_zone=ASHRAEClimateZone.ZONE_3B_WARM_DRY,
    wmo_station="946720",
    station_name="Robertstown (nearest: Clare)",
    latitude=-33.97,
    longitude=139.09,
    elevation_m=245,
    std_pressure_kpa=98.4,
    cooling_04_db=36.2,
    cooling_04_mcwb=19.8,
    cooling_1_db=33.7,
    cooling_1_mcwb=19.0,
    heating_996_db=1.2,
    heating_99_db=2.8,
    extreme_max_db=42.5,
    extreme_min_db=0.6,
    extreme_max_wb=24.5,
    annual_avg_db=15.7,
    monthly_avg_db=[22.1, 22.0, 19.7, 15.8, 12.7, 10.2, 9.4, 10.4, 12.4, 15.1, 18.2, 20.6],
    hdd_18_3=2098,
    cdd_18_3=561,
    hottest_month=1,
    hottest_month_db_range=15.2,
)

# Alice Springs / Pine Gap (PG)
ASHRAE_ALICE_SPRINGS = ASHRAEConditions(
    climate_zone=ASHRAEClimateZone.ZONE_3B_WARM_DRY,
    wmo_station="943260",
    station_name="Alice Springs Airport",
    latitude=-23.80,
    longitude=133.88,
    elevation_m=547,
    std_pressure_kpa=94.9,
    cooling_04_db=40.5,
    cooling_04_mcwb=20.4,
    cooling_1_db=38.9,
    cooling_1_mcwb=20.0,
    heating_996_db=1.8,
    heating_99_db=3.4,
    extreme_max_db=45.6,
    extreme_min_db=-3.5,
    extreme_max_wb=26.1,
    annual_avg_db=21.4,
    monthly_avg_db=[29.2, 28.1, 25.4, 20.7, 15.9, 12.7, 12.4, 15.0, 20.1, 24.4, 26.9, 28.6],
    hdd_18_3=867,
    cdd_18_3=2093,
    hottest_month=1,
    hottest_month_db_range=14.4,
)

# Batam, Indonesia (BT1-3)
ASHRAE_BATAM = ASHRAEConditions(
    climate_zone=ASHRAEClimateZone.ZONE_1A_VERY_HOT_HUMID,
    wmo_station="962490",
    station_name="Hang Nadim Airport, Batam",
    latitude=1.11,
    longitude=104.11,
    elevation_m=24,
    std_pressure_kpa=101.0,
    cooling_04_db=32.2,
    cooling_04_mcwb=27.4,
    cooling_1_db=31.4,
    cooling_1_mcwb=27.0,
    heating_996_db=23.3,
    heating_99_db=23.7,
    extreme_max_db=34.4,
    extreme_min_db=21.8,
    extreme_max_wb=28.1,
    annual_avg_db=27.5,
    monthly_avg_db=[26.7, 27.2, 27.6, 28.0, 28.3, 28.2, 27.8, 27.8, 27.7, 27.7, 27.2, 26.8],
    hdd_18_3=0,
    cdd_18_3=4591,
    hottest_month=5,
    hottest_month_db_range=7.5,
)


# =============================================================================
# Pre-defined Grid Energy Mix for Each Region
# =============================================================================

# Tasmania — predominantly hydroelectric (>90%)
GRID_MIX_TASMANIA = GridEnergyMix(
    nem_region=NEM_Region.TAS,
    connection_type=GridConnection.GRID_HYDRO,
    hydro_pct=82.0,
    wind_pct=14.0,
    solar_pct=2.0,
    gas_pct=1.0,
    coal_pct=0.0,
    other_renewable_pct=1.0,
    carbon_intensity_kg_mwh=30.0,   # Very low due to hydro dominance
    grid_reliability_pct=99.97,
    spot_price_avg_aud_mwh=55.0,
)

# South Australia — high wind/solar, gas peaking
GRID_MIX_SOUTH_AUSTRALIA = GridEnergyMix(
    nem_region=NEM_Region.SA,
    connection_type=GridConnection.GRID_GAS_RENEWABLE,
    hydro_pct=0.0,
    wind_pct=45.0,
    solar_pct=20.0,
    gas_pct=30.0,
    coal_pct=0.0,
    other_renewable_pct=5.0,
    carbon_intensity_kg_mwh=200.0,
    grid_reliability_pct=99.95,
    spot_price_avg_aud_mwh=75.0,
)

# Victoria — transitioning from coal/gas to renewables
GRID_MIX_VICTORIA = GridEnergyMix(
    nem_region=NEM_Region.VIC,
    connection_type=GridConnection.GRID_MIXED,
    hydro_pct=5.0,
    wind_pct=25.0,
    solar_pct=10.0,
    gas_pct=15.0,
    coal_pct=40.0,
    other_renewable_pct=5.0,
    carbon_intensity_kg_mwh=550.0,
    grid_reliability_pct=99.98,
    spot_price_avg_aud_mwh=65.0,
)

# New South Wales — coal-heavy but transitioning
GRID_MIX_NSW = GridEnergyMix(
    nem_region=NEM_Region.NSW,
    connection_type=GridConnection.GRID_MIXED,
    hydro_pct=8.0,
    wind_pct=15.0,
    solar_pct=15.0,
    gas_pct=7.0,
    coal_pct=50.0,
    other_renewable_pct=5.0,
    carbon_intensity_kg_mwh=600.0,
    grid_reliability_pct=99.98,
    spot_price_avg_aud_mwh=70.0,
)

# Northern Territory — gas generation (off-grid for Pine Gap)
GRID_MIX_NT_GAS = GridEnergyMix(
    nem_region=None,  # NT is not part of NEM
    connection_type=GridConnection.GAS_GENERATION,
    hydro_pct=0.0,
    wind_pct=0.0,
    solar_pct=5.0,
    gas_pct=95.0,
    coal_pct=0.0,
    other_renewable_pct=0.0,
    carbon_intensity_kg_mwh=450.0,
    grid_reliability_pct=99.90,
    spot_price_avg_aud_mwh=120.0,
)

# Batam, Indonesia — grid (gas + coal dominant)
GRID_MIX_BATAM = GridEnergyMix(
    nem_region=None,
    connection_type=GridConnection.GRID_STANDARD,
    hydro_pct=5.0,
    wind_pct=0.0,
    solar_pct=3.0,
    gas_pct=52.0,
    coal_pct=37.0,
    other_renewable_pct=3.0,
    carbon_intensity_kg_mwh=700.0,
    grid_reliability_pct=99.80,
    spot_price_avg_aud_mwh=90.0,  # Converted to AUD equivalent
)


# =============================================================================
# Pre-defined Site Configurations (from Firmus-Southgate Master Plan)
# =============================================================================

SITE_LN2_LN3 = SiteConfig(
    dc_code="LN2/LN3",
    campus="Launceston",
    region="Tasmania",
    provider="Firmus",
    status=SiteStatus.UNDER_CONSTRUCTION,
    ashrae=ASHRAE_LAUNCESTON,
    grid_mix=GRID_MIX_TASMANIA,
    gpu_series="GB300",
    nv_code="Oberon",
    gross_mw=46.0,
    pue=1.1,
    num_racks=256,
    num_gpus=18432,
    rack_power_kw=140.0,
)

SITE_WV1 = SiteConfig(
    dc_code="WV1",
    campus="Wesley Vale",
    region="Tasmania",
    provider="Firmus",
    status=SiteStatus.PLANNED,
    ashrae=ASHRAE_LAUNCESTON,  # Same climate region
    grid_mix=GRID_MIX_TASMANIA,
    gpu_series="GB300",
    nv_code="Oberon",
    gross_mw=143.0,
    pue=1.1,
    num_racks=640,
    num_gpus=46080,
    rack_power_kw=140.0,
)

SITE_GT1_2 = SiteConfig(
    dc_code="GT1-2",
    campus="Georgetown",
    region="Tasmania",
    provider="Firmus",
    status=SiteStatus.PLANNED,
    ashrae=ASHRAE_LAUNCESTON,  # Same climate region
    grid_mix=GRID_MIX_TASMANIA,
    gpu_series="GB300",
    nv_code="Oberon",
    gross_mw=143.0,
    pue=1.1,
    num_racks=640,
    num_gpus=46080,
    rack_power_kw=140.0,
)

SITE_RT1 = SiteConfig(
    dc_code="RT1",
    campus="Robertstown",
    region="South Australia",
    provider="Firmus",
    status=SiteStatus.PLANNED,
    ashrae=ASHRAE_ROBERTSTOWN,
    grid_mix=GRID_MIX_SOUTH_AUSTRALIA,
    gpu_series="VR",
    nv_code="Oberon",
    gross_mw=143.0,
    pue=1.1,
    num_racks=384,
    num_gpus=27648,
    rack_power_kw=210.0,
)

# Batam split: first 120 MW is GB300, remaining capacity is VR (Vera Rubin)
SITE_BT1_2 = SiteConfig(
    dc_code="BT1-2",
    campus="Batam",
    region="Batam, Indonesia",
    provider="DayOne",
    status=SiteStatus.CONTRACTING,
    ashrae=ASHRAE_BATAM,
    grid_mix=GRID_MIX_BATAM,
    gpu_series="GB300",
    nv_code="Oberon",
    gross_mw=120.0,
    pue=1.3,
    num_racks=410,
    num_gpus=29520,
    rack_power_kw=140.0,
)

SITE_BT3 = SiteConfig(
    dc_code="BT3",
    campus="Batam",
    region="Batam, Indonesia",
    provider="DayOne",
    status=SiteStatus.CONTRACTING,
    ashrae=ASHRAE_BATAM,
    grid_mix=GRID_MIX_BATAM,
    gpu_series="VR",
    nv_code="Oberon",
    gross_mw=30.0,
    pue=1.3,
    num_racks=102,
    num_gpus=7344,
    rack_power_kw=210.0,
)

SITE_PG = SiteConfig(
    dc_code="PG",
    campus="Pine Gap",
    region="Northern Territory",
    provider="Firmus",
    status=SiteStatus.GAS_SOURCING,
    ashrae=ASHRAE_ALICE_SPRINGS,
    grid_mix=GRID_MIX_NT_GAS,
    gpu_series="VR",
    nv_code="Oberon",
    gross_mw=575.0,
    pue=1.1,
    num_racks=1280,
    num_gpus=92160,
    rack_power_kw=210.0,
)

SITE_BK2 = SiteConfig(
    dc_code="BK2",
    campus="Brooklyn",
    region="Melbourne",
    provider="CDC",
    status=SiteStatus.IN_BUILD,
    ashrae=ASHRAE_MELBOURNE,
    grid_mix=GRID_MIX_VICTORIA,
    gpu_series="GB300",
    nv_code="Oberon",
    gross_mw=55.0,
    pue=1.3,
    num_racks=384,
    num_gpus=27648,
    rack_power_kw=140.0,
)

SITE_HU4 = SiteConfig(
    dc_code="HU4",
    campus="Hume 4",
    region="Canberra",
    provider="CDC",
    status=SiteStatus.ROFR,
    ashrae=ASHRAE_CANBERRA,
    grid_mix=GRID_MIX_NSW,  # ACT draws from NSW grid
    gpu_series="GB300",
    nv_code="Oberon",
    gross_mw=3.2,
    pue=1.3,
    num_racks=8,
    num_gpus=576,
    rack_power_kw=140.0,
)

SITE_HU5 = SiteConfig(
    dc_code="HU5",
    campus="Hume 5",
    region="Canberra",
    provider="CDC",
    status=SiteStatus.ROFR,
    ashrae=ASHRAE_CANBERRA,
    grid_mix=GRID_MIX_NSW,
    gpu_series="GB300",
    nv_code="Oberon",
    gross_mw=10.0,
    pue=1.3,
    num_racks=32,
    num_gpus=2304,
    rack_power_kw=140.0,
)

SITE_HU6 = SiteConfig(
    dc_code="HU6",
    campus="Hume 6",
    region="Canberra",
    provider="CDC",
    status=SiteStatus.ROFR,
    ashrae=ASHRAE_CANBERRA,
    grid_mix=GRID_MIX_NSW,
    gpu_series="GB300",
    nv_code="Oberon",
    gross_mw=17.0,
    pue=1.3,
    num_racks=64,
    num_gpus=4608,
    rack_power_kw=140.0,
)

SITE_BE1 = SiteConfig(
    dc_code="BE1",
    campus="Beard 1",
    region="Canberra",
    provider="CDC",
    status=SiteStatus.ROFR,
    ashrae=ASHRAE_CANBERRA,
    grid_mix=GRID_MIX_NSW,
    gpu_series="GB300",
    nv_code="Oberon",
    gross_mw=10.5,
    pue=1.3,
    num_racks=32,
    num_gpus=2304,
    rack_power_kw=140.0,
)

SITE_MP1 = SiteConfig(
    dc_code="MP1",
    campus="Marsden Park",
    region="Sydney",
    provider="CDC",
    status=SiteStatus.ROFR,
    ashrae=ASHRAE_SYDNEY,
    grid_mix=GRID_MIX_NSW,
    gpu_series="VR",
    nv_code="Oberon",
    gross_mw=125.0,
    pue=1.3,
    num_racks=448,
    num_gpus=32256,
    rack_power_kw=210.0,
)


# =============================================================================
# Site Registry
# =============================================================================

ALL_SITES: Dict[str, SiteConfig] = {
    "LN2/LN3": SITE_LN2_LN3,
    "WV1": SITE_WV1,
    "GT1-2": SITE_GT1_2,
    "RT1": SITE_RT1,
    "BT1-2": SITE_BT1_2,
    "BT3": SITE_BT3,
    "PG": SITE_PG,
    "BK2": SITE_BK2,
    "HU4": SITE_HU4,
    "HU5": SITE_HU5,
    "HU6": SITE_HU6,
    "BE1": SITE_BE1,
    "MP1": SITE_MP1,
}


def get_site(dc_code: str) -> SiteConfig:
    """Look up a site by its data center code.
    
    Args:
        dc_code: Data center code (e.g., 'LN2/LN3', 'RT1', 'BK2')
    
    Returns:
        SiteConfig for the specified site
    
    Raises:
        KeyError: If dc_code not found
    """
    if dc_code not in ALL_SITES:
        available = ", ".join(ALL_SITES.keys())
        raise KeyError(
            f"Site '{dc_code}' not found. Available sites: {available}")
    return ALL_SITES[dc_code]


def get_sites_by_region(region: str) -> List[SiteConfig]:
    """Get all sites in a given region.
    
    Args:
        region: Region name (e.g., 'Tasmania', 'Canberra', 'Melbourne')
    
    Returns:
        List of SiteConfig objects in the region
    """
    return [s for s in ALL_SITES.values() 
            if s.region.lower() == region.lower()]


def get_sites_by_gpu(gpu_series: str) -> List[SiteConfig]:
    """Get all sites using a specific GPU series.
    
    Args:
        gpu_series: GPU series name (e.g., 'H200', 'GB300', 'VR')
    
    Returns:
        List of SiteConfig objects using the GPU series
    """
    return [s for s in ALL_SITES.values() 
            if s.gpu_series == gpu_series]


def get_sites_by_provider(provider: str) -> List[SiteConfig]:
    """Get all sites from a specific provider.
    
    Args:
        provider: Provider name ('Firmus', 'CDC', 'DayOne')
    
    Returns:
        List of SiteConfig objects from the provider
    """
    return [s for s in ALL_SITES.values() 
            if s.provider.lower() == provider.lower()]


def portfolio_summary() -> Dict:
    """Generate summary of entire Firmus site portfolio.
    
    Returns:
        Dict with portfolio-level statistics
    """
    sites = list(ALL_SITES.values())
    
    total_mw = sum(s.gross_mw for s in sites)
    total_gpus = sum(s.num_gpus for s in sites)
    total_racks = sum(s.num_racks for s in sites)
    
    # Weighted average PUE
    weighted_pue = sum(s.pue * s.gross_mw for s in sites) / total_mw
    
    # Weighted average carbon intensity
    weighted_carbon = sum(
        s.grid_mix.carbon_intensity_kg_mwh * s.gross_mw 
        for s in sites) / total_mw
    
    # Weighted average renewable fraction
    weighted_renewable = sum(
        s.grid_mix.renewable_fraction * s.gross_mw 
        for s in sites) / total_mw
    
    # Total annual carbon
    total_carbon = sum(
        s.grid_mix.annual_carbon_tonnes(s.it_power_mw) for s in sites)
    
    # By GPU series
    gpu_breakdown = {}
    for s in sites:
        if s.gpu_series not in gpu_breakdown:
            gpu_breakdown[s.gpu_series] = {'sites': 0, 'gpus': 0, 'mw': 0}
        gpu_breakdown[s.gpu_series]['sites'] += 1
        gpu_breakdown[s.gpu_series]['gpus'] += s.num_gpus
        gpu_breakdown[s.gpu_series]['mw'] += s.gross_mw
    
    # By region
    region_breakdown = {}
    for s in sites:
        if s.region not in region_breakdown:
            region_breakdown[s.region] = {'sites': 0, 'gpus': 0, 'mw': 0}
        region_breakdown[s.region]['sites'] += 1
        region_breakdown[s.region]['gpus'] += s.num_gpus
        region_breakdown[s.region]['mw'] += s.gross_mw
    
    return {
        'total_sites': len(sites),
        'total_gross_mw': total_mw,
        'total_gpus': total_gpus,
        'total_racks': total_racks,
        'weighted_avg_pue': round(weighted_pue, 3),
        'weighted_avg_carbon_intensity': round(weighted_carbon, 1),
        'weighted_avg_renewable_fraction': round(weighted_renewable, 3),
        'total_annual_carbon_tonnes': round(total_carbon, 0),
        'gpu_breakdown': gpu_breakdown,
        'region_breakdown': region_breakdown,
    }


# =============================================================================
# Main — Portfolio Overview
# =============================================================================

if __name__ == "__main__":
    import json
    
    print("=" * 80)
    print("  FIRMUS AI FACTORY — Site Portfolio Environmental Analysis")
    print("=" * 80)
    
    # Portfolio summary
    summary = portfolio_summary()
    print(f"\nTotal Sites: {summary['total_sites']}")
    print(f"Total Capacity: {summary['total_gross_mw']:.0f} MW")
    print(f"Total GPUs: {summary['total_gpus']:,}")
    print(f"Weighted Avg PUE: {summary['weighted_avg_pue']:.3f}")
    print(f"Weighted Renewable Fraction: "
          f"{summary['weighted_avg_renewable_fraction']:.1%}")
    print(f"Total Annual Carbon: "
          f"{summary['total_annual_carbon_tonnes']:,.0f} tonnes CO2")
    
    print("\n" + "-" * 80)
    print("  Site-by-Site Environmental Conditions")
    print("-" * 80)
    
    for code, site in ALL_SITES.items():
        report = site.generate_site_report()
        env = report['environment']
        cooling = report['cooling']
        grid = report['grid']
        
        print(f"\n{'='*60}")
        print(f"  {code} — {site.campus}, {site.region}")
        print(f"  Provider: {site.provider} | Status: {site.status.value}")
        print(f"  GPU: {site.gpu_series} ({site.nv_code}) | "
              f"Capacity: {site.gross_mw} MW | PUE: {site.pue}")
        print(f"{'='*60}")
        print(f"  Climate Zone: {env['climate_zone']}")
        print(f"  Annual Avg Temp: {env['annual_avg_temp_c']:.1f}°C")
        print(f"  Design Cooling: {env['design_cooling_temp_c']:.1f}°C")
        print(f"  Extreme Max: {env['extreme_max_temp_c']:.1f}°C")
        print(f"  Extreme Min: {env['extreme_min_temp_c']:.1f}°C")
        print(f"  Free Cooling: {cooling['free_cooling_pct']:.0f}% of year")
        print(f"  Min Thermal Margin: {cooling['min_thermal_margin_c']:.1f}°C")
        print(f"  Grid Renewable: {grid['renewable_fraction']:.0%}")
        print(f"  Carbon Intensity: {grid['carbon_intensity_kg_mwh']} kg/MWh")
        print(f"  Annual Carbon: {grid['annual_carbon_tonnes']:,.0f} t CO2")
