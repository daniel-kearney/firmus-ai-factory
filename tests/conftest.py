"""
Pytest configuration and shared fixtures for Firmus AI Factory tests.
"""

import sys
import os
import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def default_config():
    """Standard digital twin configuration for testing"""
    return {
        "gpu": {"name": "H100", "TDP": 700, "count": 8},
        "thermal": {
            "cooling_type": "immersion",
            "coolant": "EC-100",
            "flow_rate": 2.5,
            "inlet_temp": 35.0,
        },
        "power": {
            "transformer_rating_kva": 2000,
            "voltage_primary": 13800,
            "voltage_secondary": 480,
        },
        "grid": {
            "nominal_voltage": 480,
            "nominal_frequency": 60,
            "rated_power_kw": 1000,
        },
        "economics": {
            "tariff_type": "TOU",
            "off_peak_rate": 50.0,
            "mid_peak_rate": 100.0,
            "on_peak_rate": 200.0,
            "demand_charge": 15.0,
        },
    }


@pytest.fixture
def tou_prices():
    """24-hour TOU price profile"""
    prices = np.zeros(24)
    for h in range(24):
        if h < 7 or h >= 23:
            prices[h] = 50.0   # Off-peak
        elif 14 <= h < 18:
            prices[h] = 200.0  # On-peak
        else:
            prices[h] = 100.0  # Mid-peak
    return prices


@pytest.fixture
def frequency_event_profile():
    """Grid frequency event profile (5-minute duration)"""
    dt = 1.0
    duration = 300
    freq = np.ones(duration) * 60.0
    freq[60:180] = 59.92   # 80 mHz dip
    freq[180:240] = np.linspace(59.92, 60.0, 60)  # Recovery
    return freq
