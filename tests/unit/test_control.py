"""
Unit tests for Control and Digital Twin Integration module
"""

import unittest
import numpy as np
from firmus_ai_factory.control import DigitalTwin


class TestDigitalTwin(unittest.TestCase):
    """Test digital twin system integration"""

    def setUp(self):
        """Set up digital twin test fixtures"""
        self.config = {
            "gpu": {
                "name": "H100",
                "TDP": 700,
                "count": 8,
            },
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

    def test_initialization(self):
        """Test digital twin initialization"""
        self.assertIsNotNone(self.twin)
        self.assertIsNotNone(self.twin.config)

    def test_state_initialization(self):
        """Test initial state vector"""
        state = self.twin.get_state()

        # Should contain key state variables
        self.assertIn("gpu_power", state)
        self.assertIn("gpu_temperature", state)
        self.assertIn("cooling_power", state)
        self.assertIn("total_power", state)
        self.assertIn("grid_frequency", state)

        # Initial values should be reasonable
        self.assertGreaterEqual(state["gpu_power"], 0)
        self.assertGreater(state["gpu_temperature"], 0)

    def test_state_update(self):
        """Test state update with new inputs"""
        initial_state = self.twin.get_state()

        # Apply workload change
        new_inputs = {
            "workload_intensity": 0.9,
            "grid_frequency": 59.98,
            "ambient_temperature": 25.0,
        }

        self.twin.update_state(new_inputs, dt=1.0)
        updated_state = self.twin.get_state()

        # State should have changed
        self.assertNotEqual(
            initial_state["gpu_power"], updated_state["gpu_power"]
        )

    def test_scenario_simulation(self):
        """Test 24-hour scenario simulation"""
        results = self.twin.run_scenario(duration_hours=24, dt=300)

        # Should return time series data
        self.assertIn("time", results)
        self.assertIn("P_total", results)
        self.assertIn("T_gpu", results)
        self.assertIn("cost_rate", results)
        self.assertIn("cumulative_cost", results)

        # Time series should have correct length
        expected_steps = int(24 * 3600 / 300)
        self.assertEqual(len(results["time"]), expected_steps)

        # Power should be positive
        self.assertTrue(np.all(np.array(results["P_total"]) >= 0))

        # Temperature should be reasonable
        temps = np.array(results["T_gpu"])
        self.assertTrue(np.all(temps > 0))
        self.assertTrue(np.all(temps < 100))

        # Cumulative cost should be monotonically increasing
        costs = np.array(results["cumulative_cost"])
        self.assertTrue(np.all(np.diff(costs) >= 0))

    def test_workload_variation_response(self):
        """Test response to varying workload"""
        # Low workload
        self.twin.update_state({"workload_intensity": 0.2}, dt=60.0)
        state_low = self.twin.get_state()

        # Reset
        self.twin = DigitalTwin(self.config)

        # High workload
        self.twin.update_state({"workload_intensity": 0.95}, dt=60.0)
        state_high = self.twin.get_state()

        # Higher workload should mean higher power and temperature
        self.assertGreater(state_high["gpu_power"], state_low["gpu_power"])
        self.assertGreater(state_high["gpu_temperature"], state_low["gpu_temperature"])

    def test_grid_event_response(self):
        """Test response to grid frequency event"""
        # Normal operation
        self.twin.update_state(
            {"workload_intensity": 0.8, "grid_frequency": 60.0}, dt=60.0
        )
        state_normal = self.twin.get_state()

        # Reset
        self.twin = DigitalTwin(self.config)

        # Low frequency event
        self.twin.update_state(
            {"workload_intensity": 0.8, "grid_frequency": 59.90}, dt=60.0
        )
        state_low_freq = self.twin.get_state()

        # Should reduce power during low frequency
        self.assertLessEqual(
            state_low_freq["total_power"], state_normal["total_power"]
        )

    def test_thermal_protection(self):
        """Test thermal protection triggers"""
        T_max = 85.0

        # Run at maximum load
        for _ in range(100):
            self.twin.update_state(
                {"workload_intensity": 1.0, "ambient_temperature": 35.0}, dt=10.0
            )

        state = self.twin.get_state()

        # Temperature should be limited by thermal protection
        self.assertLessEqual(state["gpu_temperature"], T_max + 5.0)

    def test_energy_accounting(self):
        """Test energy accounting over simulation"""
        results = self.twin.run_scenario(duration_hours=1, dt=60)

        # Calculate total energy from power profile
        P_total = np.array(results["P_total"])
        dt_hours = 60.0 / 3600.0
        total_energy_kwh = np.sum(P_total) * dt_hours

        # Should be reasonable for an AI factory rack
        self.assertGreater(total_energy_kwh, 0)
        self.assertLess(total_energy_kwh, 10000)  # <10 MWh for 1 hour


class TestDigitalTwinMultiSubsystem(unittest.TestCase):
    """Test multi-subsystem coordination"""

    def setUp(self):
        """Set up multi-subsystem digital twin"""
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

    def test_subsystem_coupling(self):
        """Test that subsystems are properly coupled"""
        # Change GPU workload should affect thermal and power
        self.twin.update_state({"workload_intensity": 0.5}, dt=60.0)
        state_1 = self.twin.get_state()

        self.twin = DigitalTwin(self.config)
        self.twin.update_state({"workload_intensity": 0.9}, dt=60.0)
        state_2 = self.twin.get_state()

        # All subsystems should reflect the workload change
        self.assertGreater(state_2["gpu_power"], state_1["gpu_power"])
        self.assertGreater(state_2["gpu_temperature"], state_1["gpu_temperature"])
        self.assertGreater(state_2["total_power"], state_1["total_power"])

    def test_steady_state_convergence(self):
        """Test that system reaches steady state"""
        # Run for extended period with constant inputs
        states = []
        for _ in range(100):
            self.twin.update_state({"workload_intensity": 0.7}, dt=10.0)
            states.append(self.twin.get_state()["gpu_temperature"])

        # Temperature should converge (last 10 values should be similar)
        last_10 = states[-10:]
        temp_range = max(last_10) - min(last_10)
        self.assertLess(temp_range, 2.0)  # <2°C variation at steady state

    def test_transient_response(self):
        """Test transient response to step change"""
        # Establish steady state at low load
        for _ in range(50):
            self.twin.update_state({"workload_intensity": 0.3}, dt=10.0)

        state_before = self.twin.get_state()

        # Step change to high load
        self.twin.update_state({"workload_intensity": 0.95}, dt=10.0)
        state_after = self.twin.get_state()

        # Power should respond immediately
        self.assertGreater(state_after["gpu_power"], state_before["gpu_power"])

        # Temperature should start rising (thermal inertia)
        self.assertGreaterEqual(
            state_after["gpu_temperature"], state_before["gpu_temperature"]
        )


class TestDigitalTwinOptimization(unittest.TestCase):
    """Test digital twin optimization capabilities"""

    def setUp(self):
        """Set up digital twin for optimization tests"""
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

    def test_cost_optimization(self):
        """Test that optimization reduces cost vs baseline"""
        # Baseline: constant workload
        baseline_results = self.twin.run_scenario(
            duration_hours=24, dt=300, optimization=False
        )
        baseline_cost = baseline_results["cumulative_cost"][-1]

        # Reset
        self.twin = DigitalTwin(self.config)

        # Optimized: MPC-driven workload scheduling
        optimized_results = self.twin.run_scenario(
            duration_hours=24, dt=300, optimization=True
        )
        optimized_cost = optimized_results["cumulative_cost"][-1]

        # Optimized should be cheaper (or at least not more expensive)
        self.assertLessEqual(optimized_cost, baseline_cost * 1.01)

    def test_what_if_analysis(self):
        """Test what-if scenario analysis"""
        # Scenario 1: Normal conditions
        results_normal = self.twin.run_scenario(duration_hours=24, dt=300)

        # Reset
        self.twin = DigitalTwin(self.config)

        # Scenario 2: Higher ambient temperature
        results_hot = self.twin.run_scenario(
            duration_hours=24,
            dt=300,
            ambient_temp=40.0,  # 40°C ambient
        )

        # Higher ambient should result in higher GPU temperatures
        avg_temp_normal = np.mean(results_normal["T_gpu"])
        avg_temp_hot = np.mean(results_hot["T_gpu"])
        self.assertGreater(avg_temp_hot, avg_temp_normal)


if __name__ == "__main__":
    unittest.main()
