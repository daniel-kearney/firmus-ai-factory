"""Electricity tariff models.

This module implements various electricity rate structures including
time-of-use, real-time pricing, and demand charges.
"""

import numpy as np
from datetime import datetime
from typing import Tuple, Dict


class ElectricityTariff:
    """Electricity rate structure model.
    
    Supports:
    - Flat rates
    - Time-of-use (TOU) rates
    - Real-time pricing (RTP)
    - Demand charges
    - Fixed monthly charges
    """
    
    def __init__(self, tariff_type: str = "TOU"):
        """Initialize tariff model.
        
        Args:
            tariff_type: 'flat', 'TOU', or 'RTP'
        """
        self.tariff_type = tariff_type
        
        # Example TOU rates ($/kWh)
        self.rates = {
            'on_peak': 0.25,      # 12pm-6pm weekdays
            'mid_peak': 0.15,     # 7am-12pm, 6pm-11pm weekdays
            'off_peak': 0.08,     # 11pm-7am, weekends
        }
        
        # Demand charge ($/kW/month)
        self.demand_charge_rate = 15.0
        
        # Fixed charges
        self.monthly_fixed_charge = 500.0
    
    def calculate_cost(self, power_profile: np.ndarray, 
                      timestamps: np.ndarray) -> Tuple[float, Dict]:
        """Calculate electricity cost for power profile.
        
        Args:
            power_profile: Power consumption (W)
            timestamps: Unix timestamps
            
        Returns:
            total_cost: Total electricity cost ($)
            breakdown: Dict with cost components
        """
        # Convert to kW
        power_kw = power_profile / 1000
        
        # Energy charge
        energy_cost = 0.0
        for i in range(len(timestamps) - 1):
            dt = timestamps[i+1] - timestamps[i]
            period = self._get_tou_period(timestamps[i])
            energy_cost += (power_kw[i] * dt / 3600) * self.rates[period]
        
        # Demand charge
        peak_demand_kw = np.max(power_kw)
        demand_cost = peak_demand_kw * self.demand_charge_rate
        
        # Total cost
        total_cost = energy_cost + demand_cost + self.monthly_fixed_charge
        
        breakdown = {
            'energy_cost': energy_cost,
            'demand_cost': demand_cost,
            'fixed_cost': self.monthly_fixed_charge,
            'total_cost': total_cost
        }
        
        return total_cost, breakdown
    
    def _get_tou_period(self, timestamp: float) -> str:
        """Determine TOU period for timestamp.
        
        Args:
            timestamp: Unix timestamp
            
        Returns:
            period: 'on_peak', 'mid_peak', or 'off_peak'
        """
        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour
        is_weekday = dt.weekday() < 5
        
        if not is_weekday:
            return 'off_peak'
        
        if 12 <= hour < 18:
            return 'on_peak'
        elif (7 <= hour < 12) or (18 <= hour < 23):
            return 'mid_peak'
        else:
            return 'off_peak'
