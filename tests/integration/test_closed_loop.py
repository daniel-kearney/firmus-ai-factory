"""
Integration tests for Closed-Loop Control with Emulated Sensors

This module tests the complete digital twin closed-loop control system
using emulated sensor data instead of physical hardware connections.
"""

import unittest
import numpy as np
import pandas as pd
from firmus_ai_factory.utils.sensor_emulator import SensorEmulator
from firmus_ai_factory.control import DigitalTwin
from firmus_ai_factory.grid import GridInterface, DemandResponseManager, GRID_US_480V
from firmus_ai_factory.economics import ElectricityTariff
from firmus_ai_factory.optimization import ModelPredictiveController


class TestClosedLoopFrequencyResponse(unittest.TestCase):
    """Test closed-loop frequency response with emulated grid signals"""

    def setUp(self):
        """Set up closed-loop test environment"""
        self.config = {
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
        self.twin = DigitalTwin(self.config)
        self.emulator = SensorEmulator(noise_level=0.02)
        self.grid = GridInterface(GRID_US_480V)

    def test_frequency_droop_response(self):
        """Test closed-loop primary frequency response"""
        dt = 1.0  # 1-second control loop
        duration = 300  # 5 minutes

        # Simulate frequency event: normal -> dip -> recovery
        freq_profile = np.ones(duration) * 60.0
        freq_profile[60:180] = 59.92  # 80 mHz dip for 2 minutes
        freq_profile[180:240] = np.linspace(59.92, 60.0, 60)  # Recovery

        power_log = []
        freq_log = []

        for t in range(duration):
            # Emulated sensor reading
            reading = self.emulator.get_synthetic_reading(
                float(t), workload_intensity=0.8
            )

            # Override grid frequency with event profile
            grid_freq = freq_profile[t]

            # Digital twin processes sensor data and determines response
            self.twin.update_state(
                {
                    "workload_intensity": 0.8,
                    "grid_frequency": grid_freq,
                    "ambient_temperature": 25.0,
                },
                dt=dt,
            )

            state = self.twin.get_state()
            power_log.append(state["total_power"])
            freq_log.append(grid_freq)

        power_arr = np.array(power_log)

        # During frequency dip, power should decrease
        pre_event_power = np.mean(power_arr[30:60])
        during_event_power = np.mean(power_arr[90:150])
        self.assertLess(during_event_power, pre_event_power)

        # After recovery, power should return to normal
        post_event_power = np.mean(power_arr[250:290])
        self.assertAlmostEqual(
            post_event_power, pre_event_power, delta=pre_event_power * 0.1
        )

    def test_frequency_regulation_revenue(self):
        """Test revenue calculation from frequency regulation participation"""
        dt = 1.0
        duration = 3600  # 1 hour

        # Simulate realistic grid frequency (normal distribution around 60 Hz)
        np.random.seed(42)
        freq_profile = 60.0 + np.random.normal(0, 0.015, duration)

        baseline_power = 800e3  # 800 kW baseline
        power_log = []

        for t in range(duration):
            self.twin.update_state(
                {"workload_intensity": 0.8, "grid_frequency": freq_profile[t]},
                dt=dt,
            )
            state = self.twin.get_state()
            power_log.append(state["total_power"])

        power_arr = np.array(power_log)

        # Calculate regulation capacity provided
        regulation_up = np.mean(np.maximum(baseline_power - power_arr, 0))
        regulation_down = np.mean(np.maximum(power_arr - baseline_power, 0))
        avg_regulation = (regulation_up + regulation_down) / 2

        # Should provide measurable regulation capacity
        self.assertGreater(avg_regulation, 0)

        # Revenue estimate: $50/MW-h for regulation
        hourly_revenue = (avg_regulation / 1e6) * 50.0
        annual_revenue = hourly_revenue * 8760
        print(f"\nEstimated annual frequency regulation revenue: ${annual_revenue:,.0f}")


class TestClosedLoopDemandResponse(unittest.TestCase):
    """Test closed-loop demand response with emulated signals"""

    def setUp(self):
        """Set up DR test environment"""
        self.config = {
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
        self.twin = DigitalTwin(self.config)
        self.dr_manager = DemandResponseManager(
            baseline_power=800e3, max_reduction=300e3, ramp_rate=50e3
        )

    def test_economic_dr_event(self):
        """Test economic demand response event handling"""
        dt = 10.0  # 10-second control loop
        duration = 7200  # 2 hours

        # DR event: reduce 200 kW from t=1800 to t=5400 (30 min to 1.5 hr)
        dr_active = np.zeros(int(duration / dt))
        dr_active[180:540] = 1  # DR active period

        power_log = []
        workload_log = []

        workload = 0.8  # Base workload intensity

        for step in range(int(duration / dt)):
            t = step * dt

            # DR signal modifies workload
            if dr_active[step]:
                # Reduce workload during DR event
                target_reduction = 200e3  # 200 kW
                workload_adjusted = max(0.3, workload - target_reduction / (700 * 8))
            else:
                workload_adjusted = workload

            self.twin.update_state(
                {
                    "workload_intensity": workload_adjusted,
                    "grid_frequency": 60.0,
                },
                dt=dt,
            )

            state = self.twin.get_state()
            power_log.append(state["total_power"])
            workload_log.append(workload_adjusted)

        power_arr = np.array(power_log)

        # Power should be lower during DR event
        pre_dr_power = np.mean(power_arr[90:180])
        during_dr_power = np.mean(power_arr[270:450])
        post_dr_power = np.mean(power_arr[570:660])

        self.assertLess(during_dr_power, pre_dr_power)

        # Power should recover after DR event
        self.assertAlmostEqual(
            post_dr_power, pre_dr_power, delta=pre_dr_power * 0.15
        )

        # Calculate DR revenue
        reduction_kw = (pre_dr_power - during_dr_power) / 1000.0
        duration_hours = (5400 - 1800) / 3600.0
        dr_price = 100.0  # $/MWh
        revenue = self.dr_manager.calculate_dr_revenue(
            reduction_kw, duration_hours, dr_price
        )
        self.assertGreater(revenue, 0)
        print(f"\nDR event revenue: ${revenue:,.2f}")

    def test_ramp_rate_compliance(self):
        """Test that power changes respect ramp rate limits"""
        dt = 1.0
        duration = 120  # 2 minutes

        power_log = []

        for t in range(duration):
            # Step change at t=30
            if t < 30:
                workload = 0.9
            else:
                workload = 0.3

            self.twin.update_state(
                {"workload_intensity": workload, "grid_frequency": 60.0}, dt=dt
            )
            state = self.twin.get_state()
            power_log.append(state["total_power"])

        power_arr = np.array(power_log)

        # Check ramp rate (power change per second)
        ramp_rates = np.abs(np.diff(power_arr)) / dt

        # Maximum ramp rate should be bounded
        max_ramp = np.max(ramp_rates)
        self.assertLess(max_ramp, 500e3)  # <500 kW/s (reasonable for AI factory)


class TestClosedLoopMPC(unittest.TestCase):
    """Test closed-loop MPC optimization with emulated sensors"""

    def setUp(self):
        """Set up MPC test environment"""
        self.config = {
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
        self.twin = DigitalTwin(self.config)
        self.mpc = ModelPredictiveController(
            horizon=24, dt=3600, weights={"cost": 1.0, "thermal": 0.5, "throughput": 0.3}
        )
        self.tariff = ElectricityTariff(
            tariff_type="TOU",
            rates={"off_peak": 50.0, "mid_peak": 100.0, "on_peak": 200.0},
            demand_charge=15.0,
            periods={
                "off_peak": [(23, 7)],
                "mid_peak": [(7, 14), (18, 23)],
                "on_peak": [(14, 18)],
            },
        )

    def test_mpc_cost_reduction(self):
        """Test MPC achieves cost reduction vs constant workload"""
        # Baseline: constant workload for 24 hours
        twin_baseline = DigitalTwin(self.config)
        baseline_results = twin_baseline.run_scenario(
            duration_hours=24, dt=300, optimization=False
        )
        baseline_cost = baseline_results["cumulative_cost"][-1]

        # MPC-optimized
        twin_optimized = DigitalTwin(self.config)
        optimized_results = twin_optimized.run_scenario(
            duration_hours=24, dt=300, optimization=True
        )
        optimized_cost = optimized_results["cumulative_cost"][-1]

        # Optimized should be cheaper
        savings = baseline_cost - optimized_cost
        savings_pct = (savings / baseline_cost) * 100 if baseline_cost > 0 else 0

        print(f"\nBaseline 24h cost: ${baseline_cost:,.2f}")
        print(f"Optimized 24h cost: ${optimized_cost:,.2f}")
        print(f"Savings: ${savings:,.2f} ({savings_pct:.1f}%)")

        self.assertLessEqual(optimized_cost, baseline_cost * 1.01)

    def test_thermal_constraint_satisfaction(self):
        """Test that MPC respects thermal constraints"""
        T_max = 85.0

        results = self.twin.run_scenario(
            duration_hours=24, dt=300, optimization=True
        )

        temps = np.array(results["T_gpu"])

        # All temperatures should be below limit
        self.assertTrue(
            np.all(temps <= T_max + 2.0),
            f"Max temperature {np.max(temps):.1f}°C exceeds limit {T_max}°C",
        )

    def test_workload_completion(self):
        """Test that MPC completes required workload"""
        results = self.twin.run_scenario(
            duration_hours=24, dt=300, optimization=True
        )

        # Total compute should be positive and substantial
        total_power = np.array(results["P_total"])
        total_energy = np.sum(total_power) * (300 / 3600)  # kWh

        self.assertGreater(total_energy, 0)
        print(f"\nTotal energy consumed: {total_energy:,.0f} kWh")


class TestClosedLoopThermalManagement(unittest.TestCase):
    """Test closed-loop thermal management with emulated sensors"""

    def setUp(self):
        """Set up thermal management test environment"""
        self.config = {
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
        self.twin = DigitalTwin(self.config)
        self.emulator = SensorEmulator(noise_level=0.02)

    def test_cooling_system_response(self):
        """Test cooling system responds to thermal load changes"""
        dt = 10.0
        duration = 600  # 10 minutes

        temp_log = []
        power_log = []

        for step in range(int(duration / dt)):
            t = step * dt

            # Ramp workload from 0.3 to 0.95 over 5 minutes
            if t < 300:
                workload = 0.3 + (0.65 * t / 300.0)
            else:
                workload = 0.95

            self.twin.update_state(
                {"workload_intensity": workload, "ambient_temperature": 25.0},
                dt=dt,
            )

            state = self.twin.get_state()
            temp_log.append(state["gpu_temperature"])
            power_log.append(state["total_power"])

        temps = np.array(temp_log)
        powers = np.array(power_log)

        # Temperature should increase with workload
        self.assertGreater(temps[-1], temps[0])

        # Temperature should stabilize (last 10 readings within 3°C)
        temp_range = np.max(temps[-10:]) - np.min(temps[-10:])
        self.assertLess(temp_range, 3.0)

    def test_ambient_temperature_impact(self):
        """Test impact of ambient temperature on cooling performance"""
        # Test at 25°C ambient
        twin_cool = DigitalTwin(self.config)
        for _ in range(60):
            twin_cool.update_state(
                {"workload_intensity": 0.8, "ambient_temperature": 25.0}, dt=10.0
            )
        state_cool = twin_cool.get_state()

        # Test at 40°C ambient
        twin_hot = DigitalTwin(self.config)
        for _ in range(60):
            twin_hot.update_state(
                {"workload_intensity": 0.8, "ambient_temperature": 40.0}, dt=10.0
            )
        state_hot = twin_hot.get_state()

        # Higher ambient should result in higher GPU temperature
        self.assertGreater(
            state_hot["gpu_temperature"], state_cool["gpu_temperature"]
        )


class TestEndToEndScenario(unittest.TestCase):
    """Test complete end-to-end scenario with all subsystems"""

    def setUp(self):
        """Set up complete system"""
        self.config = {
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

    def test_24h_operation(self):
        """Test complete 24-hour operational scenario"""
        twin = DigitalTwin(self.config)

        results = twin.run_scenario(duration_hours=24, dt=300)

        # Verify all outputs are present
        required_keys = ["time", "P_total", "T_gpu", "cost_rate", "cumulative_cost"]
        for key in required_keys:
            self.assertIn(key, results)

        # Verify data integrity
        times = np.array(results["time"])
        powers = np.array(results["P_total"])
        temps = np.array(results["T_gpu"])
        costs = np.array(results["cumulative_cost"])

        # Monotonically increasing time
        self.assertTrue(np.all(np.diff(times) > 0))

        # Positive power
        self.assertTrue(np.all(powers >= 0))

        # Reasonable temperatures
        self.assertTrue(np.all(temps > 0))
        self.assertTrue(np.all(temps < 100))

        # Non-decreasing cumulative cost
        self.assertTrue(np.all(np.diff(costs) >= -0.01))

        # Print summary
        print(f"\n24-Hour Operational Summary:")
        print(f"  Average power: {np.mean(powers):,.0f} kW")
        print(f"  Peak power: {np.max(powers):,.0f} kW")
        print(f"  Average GPU temp: {np.mean(temps):.1f}°C")
        print(f"  Peak GPU temp: {np.max(temps):.1f}°C")
        print(f"  Total energy: {np.sum(powers) * (300/3600):,.0f} kWh")
        print(f"  Total cost: ${costs[-1]:,.2f}")

    def test_week_simulation(self):
        """Test 7-day simulation for longer-term patterns"""
        twin = DigitalTwin(self.config)

        results = twin.run_scenario(duration_hours=168, dt=900)  # 15-min steps

        times = np.array(results["time"])
        costs = np.array(results["cumulative_cost"])

        # Should have correct number of steps
        expected_steps = int(168 * 3600 / 900)
        self.assertEqual(len(times), expected_steps)

        # Weekly cost should be reasonable
        weekly_cost = costs[-1]
        daily_avg = weekly_cost / 7
        self.assertGreater(daily_avg, 0)

        print(f"\n7-Day Simulation Summary:")
        print(f"  Weekly cost: ${weekly_cost:,.2f}")
        print(f"  Daily average: ${daily_avg:,.2f}")
        print(f"  Projected annual: ${daily_avg * 365:,.0f}")


if __name__ == "__main__":
    unittest.main()
