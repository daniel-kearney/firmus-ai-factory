"""
Unit tests for Energy Storage (Battery/UPS) module
"""

import unittest
import numpy as np
from firmus_ai_factory.storage import (
    LithiumIonBattery,
    BATTERY_TESLA_MEGAPACK,
)


class TestLithiumIonBattery(unittest.TestCase):
    """Test lithium-ion battery electrochemical model"""

    def setUp(self):
        """Set up test fixtures"""
        self.battery = LithiumIonBattery(BATTERY_TESLA_MEGAPACK)

    def test_initialization(self):
        """Test battery initialization with Tesla Megapack specs"""
        self.assertGreater(self.battery.capacity_kwh, 0)
        self.assertGreater(self.battery.max_power_kw, 0)
        self.assertGreaterEqual(self.battery.soc, 0)
        self.assertLessEqual(self.battery.soc, 1.0)

    def test_soc_limits(self):
        """Test state-of-charge remains within safe limits"""
        # Attempt full discharge
        for _ in range(200):
            self.battery.discharge(self.battery.max_power_kw, dt=60)

        # SOC should not go below minimum
        self.assertGreaterEqual(self.battery.soc, self.battery.soc_min)

        # Reset and attempt full charge
        self.battery.soc = 0.5
        for _ in range(200):
            self.battery.charge(self.battery.max_power_kw, dt=60)

        # SOC should not exceed maximum
        self.assertLessEqual(self.battery.soc, self.battery.soc_max)

    def test_charge_discharge_cycle(self):
        """Test charge/discharge energy balance"""
        self.battery.soc = 0.5
        initial_soc = self.battery.soc

        # Charge for 1 hour at half power
        P_charge = self.battery.max_power_kw * 0.5
        dt = 3600  # 1 hour in seconds
        energy_in = self.battery.charge(P_charge, dt)

        soc_after_charge = self.battery.soc
        self.assertGreater(soc_after_charge, initial_soc)

        # Discharge same energy
        energy_out = self.battery.discharge(P_charge, dt)

        soc_after_discharge = self.battery.soc

        # Round-trip efficiency should be 85-95%
        if energy_in > 0:
            roundtrip_eff = energy_out / energy_in
            self.assertGreater(roundtrip_eff, 0.80)
            self.assertLess(roundtrip_eff, 1.0)

    def test_power_limits(self):
        """Test power limit enforcement"""
        self.battery.soc = 0.5

        # Request power exceeding maximum
        P_request = self.battery.max_power_kw * 2.0
        energy = self.battery.discharge(P_request, dt=60)

        # Actual power should be clamped to maximum
        actual_power = energy / (60 / 3600)  # Convert back to kW
        self.assertLessEqual(actual_power, self.battery.max_power_kw * 1.01)

    def test_thermal_model(self):
        """Test battery thermal dynamics"""
        self.battery.soc = 0.5
        T_initial = self.battery.temperature

        # Heavy cycling should increase temperature
        for _ in range(20):
            self.battery.discharge(self.battery.max_power_kw, dt=60)
            self.battery.charge(self.battery.max_power_kw, dt=60)

        T_final = self.battery.temperature
        self.assertGreater(T_final, T_initial)

    def test_degradation_model(self):
        """Test capacity degradation over cycles"""
        initial_capacity = self.battery.capacity_kwh
        self.battery.soc = 0.5

        # Simulate many cycles
        for _ in range(100):
            self.battery.discharge(self.battery.max_power_kw * 0.5, dt=3600)
            self.battery.charge(self.battery.max_power_kw * 0.5, dt=3600)

        # Capacity should degrade slightly
        final_capacity = self.battery.get_effective_capacity()
        self.assertLessEqual(final_capacity, initial_capacity)

    def test_open_circuit_voltage(self):
        """Test open circuit voltage as function of SOC"""
        # OCV should increase with SOC
        soc_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        ocv_values = []

        for soc in soc_values:
            self.battery.soc = soc
            ocv = self.battery.get_open_circuit_voltage()
            ocv_values.append(ocv)
            self.assertGreater(ocv, 0)

        # OCV should be monotonically increasing
        for i in range(len(ocv_values) - 1):
            self.assertGreater(ocv_values[i + 1], ocv_values[i])

    def test_internal_resistance(self):
        """Test internal resistance model"""
        # Resistance should be positive
        R_int = self.battery.get_internal_resistance()
        self.assertGreater(R_int, 0)

        # Resistance should increase at low SOC
        self.battery.soc = 0.1
        R_low_soc = self.battery.get_internal_resistance()

        self.battery.soc = 0.5
        R_mid_soc = self.battery.get_internal_resistance()

        self.assertGreater(R_low_soc, R_mid_soc)

    def test_energy_accounting(self):
        """Test energy accounting accuracy"""
        self.battery.soc = 0.8
        total_energy_out = 0.0

        # Discharge in steps
        for _ in range(10):
            energy = self.battery.discharge(100.0, dt=360)  # 100 kW for 6 min
            total_energy_out += energy

        # Energy should match SOC change
        soc_change = 0.8 - self.battery.soc
        expected_energy = soc_change * self.battery.capacity_kwh
        # Allow for efficiency losses
        self.assertAlmostEqual(
            total_energy_out, expected_energy, delta=expected_energy * 0.15
        )


class TestBatteryConfigurations(unittest.TestCase):
    """Test different battery configurations"""

    def test_tesla_megapack(self):
        """Test Tesla Megapack configuration"""
        battery = LithiumIonBattery(BATTERY_TESLA_MEGAPACK)
        self.assertGreater(battery.capacity_kwh, 3000)  # >3 MWh
        self.assertGreater(battery.max_power_kw, 1000)  # >1 MW

    def test_custom_battery(self):
        """Test custom battery configuration"""
        custom_specs = {
            "capacity_kwh": 500.0,
            "max_power_kw": 250.0,
            "voltage_nominal": 800.0,
            "soc_min": 0.15,
            "soc_max": 0.90,
            "roundtrip_efficiency": 0.92,
            "thermal_capacity": 50000.0,
            "thermal_resistance": 0.5,
        }

        battery = LithiumIonBattery(custom_specs)
        self.assertEqual(battery.capacity_kwh, 500.0)
        self.assertEqual(battery.max_power_kw, 250.0)


class TestUPSFunctionality(unittest.TestCase):
    """Test UPS (Uninterruptible Power Supply) functionality"""

    def setUp(self):
        """Set up UPS test fixtures"""
        self.ups = LithiumIonBattery(BATTERY_TESLA_MEGAPACK)
        self.ups.soc = 1.0  # Fully charged for UPS mode

    def test_backup_duration(self):
        """Test UPS backup duration calculation"""
        critical_load_kw = 500.0  # 500 kW critical load

        backup_hours = self.ups.calculate_backup_duration(critical_load_kw)

        # Should provide reasonable backup time
        self.assertGreater(backup_hours, 0)

        # Backup time should decrease with higher load
        backup_high = self.ups.calculate_backup_duration(1000.0)
        self.assertLess(backup_high, backup_hours)

    def test_grid_failure_response(self):
        """Test response to grid failure event"""
        self.ups.soc = 0.95
        critical_load_kw = 300.0
        duration_seconds = 600  # 10-minute outage

        # Simulate grid failure
        energy_supplied = 0.0
        for t in range(0, duration_seconds, 10):
            energy = self.ups.discharge(critical_load_kw, dt=10)
            energy_supplied += energy

        # Should have supplied energy throughout outage
        self.assertGreater(energy_supplied, 0)

        # SOC should have decreased
        self.assertLess(self.ups.soc, 0.95)


if __name__ == "__main__":
    unittest.main()
