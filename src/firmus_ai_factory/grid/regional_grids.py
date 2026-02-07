"""Regional Grid Models for Singapore and Australia.

Implements grid-specific configurations for:
    - Singapore: EMA/SP PowerGrid (HGX H100/H200 + Immersion Cooling)
    - Australia: AEMO/NEM (GB300/Vera Rubin + Benmax HCU2500)

Key differences modeled:
    - Voltage levels (400V SG vs 415V AU three-phase)
    - Frequency regulation bands (±0.2 Hz SG vs ±0.15 Hz AU)
    - HT supply voltages (22kV/66kV SG vs 11kV/33kV/66kV AU)
    - Demand response mechanisms
    - Climate impact on cooling (tropical SG vs variable AU)
    - Tariff structures and market participation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum


class GridRegion(Enum):
    """Supported grid regions."""
    SINGAPORE = "singapore"
    AUSTRALIA_NEM = "australia_nem"


class DemandResponseType(Enum):
    """Types of demand response programs."""
    INTERRUPTIBLE_LOAD = "interruptible_load"
    DEMAND_SIDE_MANAGEMENT = "demand_side_management"
    FREQUENCY_REGULATION = "frequency_regulation"
    CONTINGENCY_RESERVE = "contingency_reserve"
    WHOLESALE_PRICE_RESPONSE = "wholesale_price_response"


# =============================================================================
# Grid Specifications
# =============================================================================

@dataclass
class GridSpecification:
    """Electrical grid specification for a region.
    
    Attributes:
        region: Grid region identifier
        nominal_frequency_hz: Nominal grid frequency (Hz)
        frequency_normal_band_hz: Normal operating frequency band (±Hz)
        frequency_tolerance_band_hz: Operational tolerance band (±Hz)
        frequency_extreme_min_hz: Extreme minimum frequency (Hz)
        frequency_extreme_max_hz: Extreme maximum frequency (Hz)
        single_phase_voltage_v: Single-phase voltage (V)
        three_phase_voltage_v: Three-phase line-to-line voltage (V)
        voltage_tolerance_upper_pct: Upper voltage tolerance (%)
        voltage_tolerance_lower_pct: Lower voltage tolerance (%)
        ht_supply_voltages_kv: Available HT supply voltages (kV)
        grid_operator: Grid operator name
        market_operator: Market operator name
        power_factor_requirement: Minimum power factor
        ambient_temp_range_c: (min, max) ambient temperature range (°C)
        relative_humidity_range_pct: (min, max) relative humidity range (%)
    """
    region: GridRegion
    nominal_frequency_hz: float
    frequency_normal_band_hz: float
    frequency_tolerance_band_hz: float
    frequency_extreme_min_hz: float
    frequency_extreme_max_hz: float
    single_phase_voltage_v: float
    three_phase_voltage_v: float
    voltage_tolerance_upper_pct: float
    voltage_tolerance_lower_pct: float
    ht_supply_voltages_kv: List[float]
    grid_operator: str
    market_operator: str
    power_factor_requirement: float
    ambient_temp_range_c: Tuple[float, float]
    relative_humidity_range_pct: Tuple[float, float]
    
    @property
    def frequency_normal_min_hz(self) -> float:
        return self.nominal_frequency_hz - self.frequency_normal_band_hz
    
    @property
    def frequency_normal_max_hz(self) -> float:
        return self.nominal_frequency_hz + self.frequency_normal_band_hz
    
    @property
    def voltage_min_v(self) -> float:
        return self.three_phase_voltage_v * (1 - self.voltage_tolerance_lower_pct / 100)
    
    @property
    def voltage_max_v(self) -> float:
        return self.three_phase_voltage_v * (1 + self.voltage_tolerance_upper_pct / 100)


# Pre-defined grid specifications

SINGAPORE_GRID = GridSpecification(
    region=GridRegion.SINGAPORE,
    nominal_frequency_hz=50.0,
    frequency_normal_band_hz=0.2,
    frequency_tolerance_band_hz=0.5,
    frequency_extreme_min_hz=48.0,
    frequency_extreme_max_hz=50.5,
    single_phase_voltage_v=230.0,
    three_phase_voltage_v=400.0,
    voltage_tolerance_upper_pct=10.0,
    voltage_tolerance_lower_pct=6.0,
    ht_supply_voltages_kv=[22.0, 66.0],
    grid_operator="SP PowerGrid",
    market_operator="Energy Market Company (EMC)",
    power_factor_requirement=0.85,
    ambient_temp_range_c=(24.0, 35.0),
    relative_humidity_range_pct=(60.0, 95.0),
)

AUSTRALIA_NEM_GRID = GridSpecification(
    region=GridRegion.AUSTRALIA_NEM,
    nominal_frequency_hz=50.0,
    frequency_normal_band_hz=0.15,
    frequency_tolerance_band_hz=0.5,
    frequency_extreme_min_hz=47.0,
    frequency_extreme_max_hz=52.0,
    single_phase_voltage_v=230.0,
    three_phase_voltage_v=415.0,
    voltage_tolerance_upper_pct=10.0,
    voltage_tolerance_lower_pct=6.0,
    ht_supply_voltages_kv=[11.0, 33.0, 66.0],
    grid_operator="AEMO",
    market_operator="AEMO (National Electricity Market)",
    power_factor_requirement=0.90,
    ambient_temp_range_c=(5.0, 45.0),
    relative_humidity_range_pct=(10.0, 85.0),
)


# =============================================================================
# Tariff Models
# =============================================================================

@dataclass
class TariffPeriod:
    """A time-of-use tariff period."""
    name: str
    start_hour: int
    end_hour: int
    rate_per_kwh: float  # Local currency per kWh
    demand_charge_per_kw: float = 0.0  # Per kW of peak demand


@dataclass
class GridTariff:
    """Grid tariff structure for a region.
    
    Attributes:
        region: Grid region
        currency: Currency code
        periods: List of TOU periods
        network_charge_per_kwh: Network/transmission charge
        market_charge_per_kwh: Market/ancillary charge
        carbon_charge_per_kwh: Carbon tax/levy per kWh
    """
    region: GridRegion
    currency: str
    periods: List[TariffPeriod]
    network_charge_per_kwh: float
    market_charge_per_kwh: float
    carbon_charge_per_kwh: float = 0.0
    
    def energy_rate(self, hour: int) -> float:
        """Get energy rate for a given hour of day."""
        for period in self.periods:
            if period.start_hour <= hour < period.end_hour:
                return period.rate_per_kwh
        return self.periods[0].rate_per_kwh  # Default
    
    def total_rate(self, hour: int) -> float:
        """Get total rate including all charges."""
        return (self.energy_rate(hour) + 
                self.network_charge_per_kwh + 
                self.market_charge_per_kwh +
                self.carbon_charge_per_kwh)
    
    def daily_cost(self, hourly_power_kw: np.ndarray) -> float:
        """Calculate daily electricity cost.
        
        Args:
            hourly_power_kw: 24-element array of hourly power consumption (kW)
            
        Returns:
            Daily cost in local currency
        """
        assert len(hourly_power_kw) == 24
        cost = 0.0
        for hour in range(24):
            cost += hourly_power_kw[hour] * self.total_rate(hour)
        return cost


# Singapore tariff (SP Group Large Industrial)
SINGAPORE_TARIFF = GridTariff(
    region=GridRegion.SINGAPORE,
    currency="SGD",
    periods=[
        TariffPeriod("Off-Peak", 0, 7, 0.18),
        TariffPeriod("Standard", 7, 9, 0.24),
        TariffPeriod("Peak", 9, 18, 0.30),
        TariffPeriod("Standard", 18, 22, 0.24),
        TariffPeriod("Off-Peak", 22, 24, 0.18),
    ],
    network_charge_per_kwh=0.04,
    market_charge_per_kwh=0.02,
    carbon_charge_per_kwh=0.005,  # Singapore carbon tax
)

# Australia NEM tariff (Large Industrial TOU)
AUSTRALIA_NEM_TARIFF = GridTariff(
    region=GridRegion.AUSTRALIA_NEM,
    currency="AUD",
    periods=[
        TariffPeriod("Off-Peak", 0, 7, 0.08, demand_charge_per_kw=0.0),
        TariffPeriod("Shoulder", 7, 14, 0.14, demand_charge_per_kw=8.0),
        TariffPeriod("Peak", 14, 20, 0.22, demand_charge_per_kw=15.0),
        TariffPeriod("Shoulder", 20, 22, 0.14, demand_charge_per_kw=8.0),
        TariffPeriod("Off-Peak", 22, 24, 0.08, demand_charge_per_kw=0.0),
    ],
    network_charge_per_kwh=0.06,
    market_charge_per_kwh=0.015,
    carbon_charge_per_kwh=0.0,  # No explicit carbon tax on electricity
)


# =============================================================================
# Demand Response Programs
# =============================================================================

@dataclass
class DemandResponseProgram:
    """Demand response program specification.
    
    Attributes:
        name: Program name
        region: Grid region
        dr_type: Type of demand response
        min_capacity_mw: Minimum participation capacity (MW)
        max_response_time_s: Maximum response time (seconds)
        min_duration_hours: Minimum event duration (hours)
        max_events_per_year: Maximum events per year
        payment_per_mw_hour: Payment rate (local currency/MW/hour)
        availability_payment_per_mw_month: Monthly availability payment
        penalty_per_mw_hour: Non-compliance penalty rate
    """
    name: str
    region: GridRegion
    dr_type: DemandResponseType
    min_capacity_mw: float
    max_response_time_s: float
    min_duration_hours: float
    max_events_per_year: int
    payment_per_mw_hour: float
    availability_payment_per_mw_month: float = 0.0
    penalty_per_mw_hour: float = 0.0


# Singapore demand response programs
SG_INTERRUPTIBLE_LOAD = DemandResponseProgram(
    name="Singapore Interruptible Load Scheme",
    region=GridRegion.SINGAPORE,
    dr_type=DemandResponseType.INTERRUPTIBLE_LOAD,
    min_capacity_mw=0.1,
    max_response_time_s=10.0,
    min_duration_hours=0.5,
    max_events_per_year=50,
    payment_per_mw_hour=4500.0,  # SGD/MWh (vesting contract price cap)
    availability_payment_per_mw_month=5000.0,
    penalty_per_mw_hour=9000.0,
)

SG_DEMAND_RESPONSE = DemandResponseProgram(
    name="Singapore Demand Response Programme",
    region=GridRegion.SINGAPORE,
    dr_type=DemandResponseType.DEMAND_SIDE_MANAGEMENT,
    min_capacity_mw=0.1,
    max_response_time_s=300.0,
    min_duration_hours=1.0,
    max_events_per_year=100,
    payment_per_mw_hour=300.0,  # SGD/MWh average
    availability_payment_per_mw_month=2000.0,
)

# Australia NEM demand response programs
AU_WHOLESALE_DR = DemandResponseProgram(
    name="AEMO Wholesale Demand Response Mechanism",
    region=GridRegion.AUSTRALIA_NEM,
    dr_type=DemandResponseType.WHOLESALE_PRICE_RESPONSE,
    min_capacity_mw=0.1,
    max_response_time_s=300.0,
    min_duration_hours=0.5,
    max_events_per_year=200,
    payment_per_mw_hour=300.0,  # AUD/MWh (spot price based)
)

AU_FREQUENCY_REGULATION = DemandResponseProgram(
    name="AEMO Frequency Control Ancillary Services (FCAS)",
    region=GridRegion.AUSTRALIA_NEM,
    dr_type=DemandResponseType.FREQUENCY_REGULATION,
    min_capacity_mw=1.0,
    max_response_time_s=6.0,   # Fast raise/lower: 6 seconds
    min_duration_hours=0.0,     # Continuous
    max_events_per_year=0,      # Continuous service
    payment_per_mw_hour=50.0,   # AUD/MW/hour (varies by market)
    availability_payment_per_mw_month=3000.0,
)

AU_CONTINGENCY_RESERVE = DemandResponseProgram(
    name="AEMO Emergency Frequency Control Scheme (EFCS)",
    region=GridRegion.AUSTRALIA_NEM,
    dr_type=DemandResponseType.CONTINGENCY_RESERVE,
    min_capacity_mw=5.0,
    max_response_time_s=2.0,   # Under-frequency load shedding
    min_duration_hours=0.5,
    max_events_per_year=10,
    payment_per_mw_hour=15000.0,  # AUD/MWh (market price cap)
    penalty_per_mw_hour=30000.0,
)


# =============================================================================
# Regional Grid Model
# =============================================================================

class RegionalGridModel:
    """Complete regional grid model for AI factory integration.
    
    Combines grid specifications, tariff structure, and demand response
    programs for a specific region.
    
    Attributes:
        grid_spec: Grid electrical specifications
        tariff: Tariff structure
        dr_programs: Available demand response programs
    """
    
    def __init__(self, region: GridRegion):
        """Initialize regional grid model.
        
        Args:
            region: Grid region to model
        """
        self.region = region
        
        if region == GridRegion.SINGAPORE:
            self.grid_spec = SINGAPORE_GRID
            self.tariff = SINGAPORE_TARIFF
            self.dr_programs = [SG_INTERRUPTIBLE_LOAD, SG_DEMAND_RESPONSE]
        elif region == GridRegion.AUSTRALIA_NEM:
            self.grid_spec = AUSTRALIA_NEM_GRID
            self.tariff = AUSTRALIA_NEM_TARIFF
            self.dr_programs = [AU_WHOLESALE_DR, AU_FREQUENCY_REGULATION, AU_CONTINGENCY_RESERVE]
        else:
            raise ValueError(f"Unsupported region: {region}")
    
    def transformer_config(self, 
                           facility_power_mw: float) -> Dict:
        """Determine transformer configuration for facility.
        
        Args:
            facility_power_mw: Total facility power (MW)
            
        Returns:
            Dict with transformer configuration
        """
        # Select HT voltage based on facility size
        if facility_power_mw <= 5.0:
            ht_kv = self.grid_spec.ht_supply_voltages_kv[0]
        elif facility_power_mw <= 20.0:
            ht_kv = self.grid_spec.ht_supply_voltages_kv[min(1, len(self.grid_spec.ht_supply_voltages_kv)-1)]
        else:
            ht_kv = self.grid_spec.ht_supply_voltages_kv[-1]
        
        lv_v = self.grid_spec.three_phase_voltage_v
        
        # Number of transformers (N+1 redundancy)
        transformer_rating_mva = min(facility_power_mw * 1.2, 10.0)  # Max 10 MVA per unit
        num_transformers = int(np.ceil(facility_power_mw / transformer_rating_mva)) + 1
        
        return {
            'ht_voltage_kv': ht_kv,
            'lv_voltage_v': lv_v,
            'transformer_ratio': ht_kv * 1000 / lv_v,
            'transformer_rating_mva': transformer_rating_mva,
            'num_transformers': num_transformers,
            'redundancy': f"{num_transformers-1}+1",
            'total_capacity_mva': num_transformers * transformer_rating_mva,
        }
    
    def pdu_config(self) -> Dict:
        """Get PDU configuration for the region.
        
        Returns:
            Dict with PDU voltage and configuration
        """
        return {
            'input_voltage_v': self.grid_spec.three_phase_voltage_v,
            'output_voltage_vdc': 54.0,  # NVL72 busbar voltage
            'input_phases': 3,
            'voltage_range_v': (self.grid_spec.voltage_min_v, self.grid_spec.voltage_max_v),
        }
    
    def frequency_response_capacity(self,
                                      total_it_load_mw: float,
                                      min_load_fraction: float = 0.50
                                      ) -> Dict:
        """Calculate available frequency response capacity.
        
        AI workloads can modulate power consumption for grid services.
        
        Args:
            total_it_load_mw: Total IT load (MW)
            min_load_fraction: Minimum operational load as fraction of total
            
        Returns:
            Dict with frequency response capabilities
        """
        max_reduction_mw = total_it_load_mw * (1 - min_load_fraction)
        
        # Response time depends on workload type
        # Training: can reduce within seconds (checkpoint and pause)
        # Inference: more constrained (SLA requirements)
        
        return {
            'max_reduction_mw': max_reduction_mw,
            'max_increase_mw': 0.0,  # Cannot exceed TDP
            'response_time_s': 5.0,  # GPU power capping response
            'ramp_rate_mw_per_s': max_reduction_mw / 5.0,
            'min_duration_hours': 0.5,
            'available_programs': [p.name for p in self.dr_programs],
        }
    
    def simulate_grid_frequency(self,
                                 duration_hours: float = 24.0,
                                 dt_s: float = 1.0) -> Dict[str, np.ndarray]:
        """Simulate grid frequency variations.
        
        Uses Ornstein-Uhlenbeck process calibrated to regional parameters.
        
        Args:
            duration_hours: Simulation duration (hours)
            dt_s: Time step (seconds)
            
        Returns:
            Dict with 'time_s' and 'frequency_hz' arrays
        """
        n_steps = int(duration_hours * 3600 / dt_s)
        time_s = np.arange(n_steps) * dt_s
        
        f0 = self.grid_spec.nominal_frequency_hz
        band = self.grid_spec.frequency_normal_band_hz
        
        # OU process parameters
        theta = 0.1  # Mean reversion rate
        sigma = band / 3.0  # Volatility (99.7% within normal band)
        
        freq = np.zeros(n_steps)
        freq[0] = f0
        
        for i in range(1, n_steps):
            dW = np.random.normal(0, np.sqrt(dt_s))
            freq[i] = freq[i-1] + theta * (f0 - freq[i-1]) * dt_s + sigma * dW
        
        # Clip to extreme limits
        freq = np.clip(freq, 
                       self.grid_spec.frequency_extreme_min_hz,
                       self.grid_spec.frequency_extreme_max_hz)
        
        return {'time_s': time_s, 'frequency_hz': freq}
    
    def simulate_electricity_price(self,
                                    duration_hours: float = 24.0,
                                    dt_hours: float = 0.5
                                    ) -> Dict[str, np.ndarray]:
        """Simulate electricity spot prices.
        
        Combines TOU base rates with stochastic wholesale price variations.
        
        Args:
            duration_hours: Simulation duration (hours)
            dt_hours: Time step (hours)
            
        Returns:
            Dict with 'time_hours' and 'price_per_kwh' arrays
        """
        n_steps = int(duration_hours / dt_hours)
        time_hours = np.arange(n_steps) * dt_hours
        
        prices = np.zeros(n_steps)
        for i in range(n_steps):
            hour = int(time_hours[i]) % 24
            base_rate = self.tariff.total_rate(hour)
            
            # Add wholesale price volatility
            volatility = 0.15 * base_rate
            noise = np.random.normal(0, volatility)
            
            # Occasional price spikes (1% probability)
            if np.random.random() < 0.01:
                noise += base_rate * np.random.uniform(2.0, 10.0)
            
            prices[i] = max(base_rate + noise, 0.0)
        
        return {'time_hours': time_hours, 'price_per_kwh': prices}
    
    def annual_energy_cost(self,
                            constant_power_mw: float,
                            pue: float = 1.01) -> Dict:
        """Estimate annual energy cost at constant power.
        
        Args:
            constant_power_mw: Constant IT power consumption (MW)
            pue: Power Usage Effectiveness
            
        Returns:
            Dict with annual cost breakdown
        """
        total_power_kw = constant_power_mw * 1000.0 * pue
        hourly_power = np.full(24, total_power_kw)
        
        daily_cost = self.tariff.daily_cost(hourly_power)
        annual_cost = daily_cost * 365
        
        return {
            'currency': self.tariff.currency,
            'daily_cost': daily_cost,
            'monthly_cost': annual_cost / 12,
            'annual_cost': annual_cost,
            'cost_per_kwh_avg': annual_cost / (total_power_kw * 8760),
            'total_energy_mwh_year': total_power_kw * 8760 / 1000,
        }
    
    def demand_response_revenue(self,
                                 available_capacity_mw: float,
                                 utilization_pct: float = 50.0
                                 ) -> Dict:
        """Estimate annual demand response revenue.
        
        Args:
            available_capacity_mw: Capacity available for DR (MW)
            utilization_pct: Expected utilization of DR capacity (%)
            
        Returns:
            Dict with revenue estimates per program
        """
        revenues = {}
        total_annual = 0.0
        
        for program in self.dr_programs:
            if available_capacity_mw < program.min_capacity_mw:
                continue
            
            # Availability payment
            availability_annual = program.availability_payment_per_mw_month * 12 * available_capacity_mw
            
            # Energy payment (based on utilization)
            if program.max_events_per_year > 0:
                hours_per_event = program.min_duration_hours
                events = min(program.max_events_per_year, 
                            int(program.max_events_per_year * utilization_pct / 100))
                energy_annual = (program.payment_per_mw_hour * 
                               available_capacity_mw * 
                               hours_per_event * events)
            else:
                # Continuous service (e.g., frequency regulation)
                energy_annual = (program.payment_per_mw_hour * 
                               available_capacity_mw * 
                               8760 * utilization_pct / 100)
            
            program_annual = availability_annual + energy_annual
            revenues[program.name] = {
                'availability_annual': availability_annual,
                'energy_annual': energy_annual,
                'total_annual': program_annual,
            }
            total_annual += program_annual
        
        return {
            'currency': self.tariff.currency,
            'programs': revenues,
            'total_annual_revenue': total_annual,
        }
    
    def generate_report(self, facility_power_mw: float = 10.0) -> Dict:
        """Generate comprehensive regional grid report.
        
        Args:
            facility_power_mw: Facility IT power (MW)
            
        Returns:
            Dict with complete grid analysis
        """
        transformer = self.transformer_config(facility_power_mw)
        pdu = self.pdu_config()
        fr_capacity = self.frequency_response_capacity(facility_power_mw)
        energy_cost = self.annual_energy_cost(facility_power_mw)
        dr_revenue = self.demand_response_revenue(
            fr_capacity['max_reduction_mw'], 50.0)
        
        return {
            'region': self.region.value,
            'grid_spec': {
                'frequency_hz': self.grid_spec.nominal_frequency_hz,
                'frequency_band_hz': f"±{self.grid_spec.frequency_normal_band_hz}",
                'three_phase_voltage_v': self.grid_spec.three_phase_voltage_v,
                'voltage_range_v': f"{self.grid_spec.voltage_min_v:.1f}-{self.grid_spec.voltage_max_v:.1f}",
                'operator': self.grid_spec.grid_operator,
            },
            'transformer': transformer,
            'pdu': pdu,
            'frequency_response': fr_capacity,
            'energy_cost': energy_cost,
            'demand_response_revenue': dr_revenue,
            'net_annual_cost': (energy_cost['annual_cost'] - 
                               dr_revenue['total_annual_revenue']),
        }


if __name__ == "__main__":
    # Compare Singapore and Australia grid configurations
    for region in [GridRegion.SINGAPORE, GridRegion.AUSTRALIA_NEM]:
        model = RegionalGridModel(region)
        report = model.generate_report(facility_power_mw=10.0)
        
        print(f"\n{'='*60}")
        print(f"  {region.value.upper()} Grid Analysis (10 MW Facility)")
        print(f"{'='*60}")
        print(f"  Frequency: {report['grid_spec']['frequency_hz']} Hz "
              f"({report['grid_spec']['frequency_band_hz']})")
        print(f"  3-Phase Voltage: {report['grid_spec']['three_phase_voltage_v']} V")
        print(f"  HT Supply: {report['transformer']['ht_voltage_kv']} kV")
        print(f"  Transformers: {report['transformer']['redundancy']} × "
              f"{report['transformer']['transformer_rating_mva']:.1f} MVA")
        print(f"  Annual Energy Cost: {report['energy_cost']['currency']} "
              f"{report['energy_cost']['annual_cost']:,.0f}")
        print(f"  DR Revenue: {report['demand_response_revenue']['currency']} "
              f"{report['demand_response_revenue']['total_annual_revenue']:,.0f}")
        print(f"  Net Annual Cost: {report['energy_cost']['currency']} "
              f"{report['net_annual_cost']:,.0f}")
"""
