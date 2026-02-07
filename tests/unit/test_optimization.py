"""
Unit tests for Optimization module (MPC)
"""

import unittest
import numpy as np
from firmus_ai_factory.optimization import ModelPredictiveController


class TestModelPredictiveController(unittest.TestCase):
    """Test Model Predictive Control implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mpc = ModelPredictiveController(
            horizon=24,  # 24-hour horizon
            dt=3600,  # 1-hour time step
            weights={"cost": 1.0, "thermal": 0.5, "throughput": 0.3}
        )
    
    def test_initialization(self):
        """Test MPC initialization"""
        self.assertEqual(self.mpc.horizon, 24)
        self.assertEqual(self.mpc.dt, 3600)
        self.assertIn("cost", self.mpc.weights)
    
    def test_cost_minimization(self):
        """Test electricity cost minimization"""
        # Time-of-use pricing (high during day, low at night)
        electricity_prices = np.array([
            50, 50, 50, 50, 50, 50,  # 12am-6am: low
            100, 100, 100, 100, 100, 100,  # 6am-12pm: high
            150, 150, 150, 150, 150, 150,  # 12pm-6pm: peak
            100, 100, 100, 100, 50, 50,  # 6pm-12am: moderate to low
        ])  # $/MWh
        
        # Workload that can be deferred
        workload_energy = 1000.0  # 1 MWh total
        deadline = 24  # Complete within 24 hours
        
        schedule = self.mpc.optimize_workload_schedule(
            workload_energy, electricity_prices, deadline
        )
        
        # Should schedule more work during low-price periods
        self.assertEqual(len(schedule), 24)
        
        # Early morning hours (low price) should have higher allocation
        morning_work = np.sum(schedule[0:6])
        afternoon_work = np.sum(schedule[12:18])
        self.assertGreater(morning_work, afternoon_work)
        
        # Total work should match requirement
        self.assertAlmostEqual(np.sum(schedule), workload_energy, delta=10.0)
    
    def test_thermal_constraint(self):
        """Test thermal constraint enforcement"""
        T_max = 85.0  # Maximum GPU temperature (Celsius)
        T_ambient = 25.0
        
        # High power workload
        power_schedule = np.ones(24) * 800e3  # 800 kW constant
        
        # Check thermal feasibility
        temperatures = self.mpc.predict_temperatures(
            power_schedule, T_ambient, T_max
        )
        
        # All temperatures should be below limit
        self.assertTrue(np.all(temperatures <= T_max + 1.0))  # +1°C tolerance
    
    def test_throughput_maximization(self):
        """Test throughput maximization under constraints"""
        power_budget = 1000e3  # 1 MW power budget
        T_max = 85.0
        
        optimal_schedule = self.mpc.maximize_throughput(
            power_budget, T_max, horizon=24
        )
        
        # Should utilize available power efficiently
        avg_power = np.mean(optimal_schedule)
        self.assertGreater(avg_power, 0)
        self.assertLessEqual(avg_power, power_budget)
        
        # Should not violate thermal limits
        self.assertTrue(np.all(optimal_schedule >= 0))
    
    def test_multi_objective_optimization(self):
        """Test multi-objective optimization (cost + thermal + throughput)"""
        electricity_prices = np.random.uniform(50, 150, 24)  # $/MWh
        workload_requirement = 500.0  # MWh
        T_max = 85.0
        deadline = 24
        
        result = self.mpc.optimize_multi_objective(
            workload_requirement,
            electricity_prices,
            T_max,
            deadline
        )
        
        # Should return feasible solution
        self.assertIn("schedule", result)
        self.assertIn("cost", result)
        self.assertIn("max_temperature", result)
        
        # Constraints should be satisfied
        self.assertLessEqual(result["max_temperature"], T_max + 1.0)
        self.assertAlmostEqual(
            np.sum(result["schedule"]), 
            workload_requirement, 
            delta=workload_requirement * 0.05  # 5% tolerance
        )
    
    def test_receding_horizon(self):
        """Test receding horizon control"""
        # Initial state
        current_time = 0
        electricity_prices = np.random.uniform(50, 150, 48)  # 48-hour forecast
        workload_queue = 1000.0  # MWh remaining
        
        # First MPC solve
        schedule_1 = self.mpc.solve_receding_horizon(
            current_time, workload_queue, electricity_prices[0:24]
        )
        
        # Execute first hour
        executed_work = schedule_1[0]
        workload_queue -= executed_work
        current_time += 1
        
        # Second MPC solve (shifted horizon)
        schedule_2 = self.mpc.solve_receding_horizon(
            current_time, workload_queue, electricity_prices[1:25]
        )
        
        # Schedules should be different (adapting to new information)
        self.assertNotEqual(schedule_1[1], schedule_2[0])
        
        # Remaining workload should be accounted for
        self.assertLess(workload_queue, 1000.0)
    
    def test_demand_response_integration(self):
        """Test integration with demand response events"""
        # Normal schedule
        base_schedule = np.ones(24) * 500e3  # 500 kW baseline
        
        # DR event: reduce power by 200 kW from hour 14-16
        dr_event = {
            "start_hour": 14,
            "end_hour": 16,
            "reduction_kw": 200e3,
        }
        
        adjusted_schedule = self.mpc.apply_dr_event(base_schedule, dr_event)
        
        # Power should be reduced during DR event
        self.assertLess(
            adjusted_schedule[14], 
            base_schedule[14] - dr_event["reduction_kw"] + 10e3  # 10 kW tolerance
        )
        
        # Work should be shifted to other hours
        total_work_base = np.sum(base_schedule)
        total_work_adjusted = np.sum(adjusted_schedule)
        self.assertAlmostEqual(total_work_base, total_work_adjusted, delta=50e3)
    
    def test_solve_time(self):
        """Test MPC solve time performance"""
        import time
        
        electricity_prices = np.random.uniform(50, 150, 24)
        workload = 500.0
        T_max = 85.0
        
        start_time = time.time()
        result = self.mpc.optimize_multi_objective(
            workload, electricity_prices, T_max, deadline=24
        )
        solve_time = time.time() - start_time
        
        # Should solve quickly (<1 second for 24-hour horizon)
        self.assertLess(solve_time, 1.0)
        
        print(f"\nMPC solve time: {solve_time*1000:.1f} ms")


class TestMPCEdgeCases(unittest.TestCase):
    """Test MPC edge cases and robustness"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mpc = ModelPredictiveController(horizon=24, dt=3600)
    
    def test_infeasible_problem(self):
        """Test handling of infeasible optimization problem"""
        # Impossible constraint: complete 1000 MWh in 1 hour with 100 kW limit
        workload = 1000.0  # MWh
        power_limit = 100e3  # 100 kW
        deadline = 1  # 1 hour
        
        # Should either raise exception or return partial solution
        try:
            result = self.mpc.optimize_workload_schedule(
                workload, 
                np.ones(24) * 100.0,  # prices
                deadline,
                power_limit=power_limit
            )
            # If returns, should indicate infeasibility
            self.assertLess(np.sum(result), workload)
        except Exception as e:
            # Expected behavior: raise infeasibility exception
            self.assertIn("infeasible", str(e).lower())
    
    def test_zero_workload(self):
        """Test handling of zero workload"""
        schedule = self.mpc.optimize_workload_schedule(
            0.0, np.ones(24) * 100.0, deadline=24
        )
        
        # Should return zero schedule
        self.assertTrue(np.all(schedule == 0))
    
    def test_tight_deadline(self):
        """Test optimization with very tight deadline"""
        workload = 100.0  # MWh
        deadline = 2  # 2 hours (tight)
        prices = np.ones(24) * 100.0
        
        schedule = self.mpc.optimize_workload_schedule(workload, prices, deadline)
        
        # Work should be concentrated in first 2 hours
        early_work = np.sum(schedule[0:deadline])
        self.assertGreater(early_work, workload * 0.95)  # >95% in deadline window


if __name__ == "__main__":
    unittest.main()
