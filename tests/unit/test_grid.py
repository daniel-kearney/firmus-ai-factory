"""
Unit tests for Grid Interface module
"""

import unittest
import numpy as np
from firmus_ai_factory.grid import (
    GridInterface,
    DemandResponseManager,
    GRID_US_480V,
    GRID_EU_400V,
)


class TestGridInterface(unittest.TestCase):
    """Test grid interconnection model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.grid = GridInterface(GRID_US_480V)
    
    def test_initialization(self):
        """Test grid interface initialization"""
        self.assertEqual(self.grid.nominal_voltage, 480.0)
        self.assertEqual(self.grid.nominal_frequency, 60.0)
        self.assertGreater(self.grid.rated_power, 0)
    
    def test_frequency_response(self):
        """Test primary frequency response (droop control)"""
        f_grid = 59.95  # 50 mHz below nominal
        P_base = 1000e3  # 1 MW baseline
        
        P_response = self.grid.calculate_frequency_response(f_grid, P_base)
        
        # Should reduce power when frequency is low
        self.assertLess(P_response, P_base)
        
        # Response should be proportional to frequency deviation
        delta_f = self.grid.nominal_frequency - f_grid
        expected_response = P_base * (1 - self.grid.droop_coefficient * delta_f)
        self.assertAlmostEqual(P_response, expected_response, delta=1000.0)
    
    def test_voltage_regulation(self):
        """Test voltage regulation"""
        V_grid = 470.0  # 10V below nominal
        Q_base = 100e3  # 100 kVAR baseline
        
        Q_response = self.grid.calculate_voltage_response(V_grid, Q_base)
        
        # Should inject reactive power when voltage is low
        self.assertGreater(Q_response, Q_base)
    
    def test_power_factor_correction(self):
        """Test power factor correction"""
        P_active = 1000e3  # 1 MW
        pf_target = 0.95
        
        Q_required = self.grid.calculate_reactive_power(P_active, pf_target)
        
        # Reactive power should be positive for lagging PF
        self.assertGreater(Q_required, 0)
        
        # Check power triangle
        S_apparent = np.sqrt(P_active**2 + Q_required**2)
        pf_actual = P_active / S_apparent
        self.assertAlmostEqual(pf_actual, pf_target, delta=0.01)
    
    def test_grid_stability(self):
        """Test grid stability assessment"""
        # Test stable condition
        f_stable = 60.0
        V_stable = 480.0
        is_stable = self.grid.check_stability(f_stable, V_stable)
        self.assertTrue(is_stable)
        
        # Test unstable frequency
        f_unstable = 59.0  # 1 Hz deviation
        is_stable = self.grid.check_stability(f_unstable, V_stable)
        self.assertFalse(is_stable)
        
        # Test unstable voltage
        V_unstable = 400.0  # Large voltage drop
        is_stable = self.grid.check_stability(f_stable, V_unstable)
        self.assertFalse(is_stable)


class TestDemandResponseManager(unittest.TestCase):
    """Test demand response management"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.dr_manager = DemandResponseManager(
            baseline_power=1000e3,  # 1 MW
            max_reduction=300e3,  # 300 kW max reduction
            ramp_rate=50e3  # 50 kW/s ramp rate
        )
    
    def test_initialization(self):
        """Test DR manager initialization"""
        self.assertEqual(self.dr_manager.baseline_power, 1000e3)
        self.assertEqual(self.dr_manager.max_reduction, 300e3)
    
    def test_economic_dr_bid(self):
        """Test economic demand response bidding"""
        electricity_price = 150.0  # $/MWh
        
        bid = self.dr_manager.calculate_dr_bid(electricity_price)
        
        # Bid should be a dictionary with required fields
        self.assertIn("power_reduction", bid)
        self.assertIn("duration", bid)
        self.assertIn("price", bid)
        
        # Power reduction should be within limits
        self.assertGreaterEqual(bid["power_reduction"], 0)
        self.assertLessEqual(bid["power_reduction"], self.dr_manager.max_reduction)
        
        # Higher prices should trigger larger reductions
        high_price = 300.0
        high_bid = self.dr_manager.calculate_dr_bid(high_price)
        self.assertGreater(high_bid["power_reduction"], bid["power_reduction"])
    
    def test_emergency_dr_response(self):
        """Test emergency demand response"""
        event = {
            "type": "emergency",
            "requested_reduction": 250e3,  # 250 kW
            "duration": 3600,  # 1 hour
        }
        
        response = self.dr_manager.respond_to_dr_event(event)
        
        # Should commit to requested reduction (within capability)
        self.assertLessEqual(response["committed_reduction"], self.dr_manager.max_reduction)
        self.assertGreater(response["committed_reduction"], 0)
    
    def test_workload_deferral(self):
        """Test workload deferral strategy"""
        current_time = 14.0  # 2 PM (peak hours)
        workload_power = 200e3  # 200 kW workload
        workload_deadline = 20.0  # 8 PM deadline
        
        defer_decision = self.dr_manager.evaluate_workload_deferral(
            current_time, workload_power, workload_deadline
        )
        
        # Should defer during peak hours if deadline allows
        self.assertIn("defer", defer_decision)
        self.assertIn("optimal_start_time", defer_decision)
        
        if defer_decision["defer"]:
            # Optimal start time should be before deadline
            self.assertLess(defer_decision["optimal_start_time"], workload_deadline)
    
    def test_ramp_rate_compliance(self):
        """Test ramp rate limits"""
        P_initial = 1000e3  # 1 MW
        P_target = 700e3  # 700 kW (300 kW reduction)
        dt = 1.0  # 1 second
        
        P_next = self.dr_manager.apply_ramp_rate(P_initial, P_target, dt)
        
        # Power change should respect ramp rate
        delta_P = abs(P_next - P_initial)
        max_delta = self.dr_manager.ramp_rate * dt
        self.assertLessEqual(delta_P, max_delta + 1.0)  # +1W tolerance
    
    def test_dr_revenue_calculation(self):
        """Test demand response revenue calculation"""
        reduction_kw = 250.0  # 250 kW
        duration_hours = 2.0  # 2 hours
        dr_price = 100.0  # $/MWh
        
        revenue = self.dr_manager.calculate_dr_revenue(
            reduction_kw, duration_hours, dr_price
        )
        
        # Revenue = reduction (MW) * duration (h) * price ($/MWh)
        expected_revenue = (reduction_kw / 1000.0) * duration_hours * dr_price
        self.assertAlmostEqual(revenue, expected_revenue, delta=0.01)


class TestGridIntegration(unittest.TestCase):
    """Test integrated grid interface and DR management"""
    
    def setUp(self):
        """Set up integrated system"""
        self.grid = GridInterface(GRID_US_480V)
        self.dr_manager = DemandResponseManager(
            baseline_power=1000e3,
            max_reduction=300e3,
            ramp_rate=50e3
        )
    
    def test_coordinated_frequency_and_dr(self):
        """Test coordinated frequency response and DR"""
        # Low frequency event
        f_grid = 59.90  # 100 mHz below nominal
        P_base = 1000e3
        
        # Primary frequency response
        P_freq_response = self.grid.calculate_frequency_response(f_grid, P_base)
        
        # DR event triggered by low frequency
        dr_event = {
            "type": "frequency",
            "requested_reduction": P_base - P_freq_response,
            "duration": 600,  # 10 minutes
        }
        
        dr_response = self.dr_manager.respond_to_dr_event(dr_event)
        
        # Combined response should stabilize grid
        total_reduction = dr_response["committed_reduction"]
        self.assertGreater(total_reduction, 0)
        self.assertLessEqual(total_reduction, self.dr_manager.max_reduction)
    
    def test_grid_service_revenue(self):
        """Test revenue from grid services"""
        # Frequency regulation for 1 hour
        avg_regulation = 100e3  # 100 kW average
        duration_hours = 1.0
        regulation_price = 50.0  # $/MW-h
        
        freq_revenue = (avg_regulation / 1e6) * duration_hours * regulation_price
        
        # Demand response event
        dr_reduction = 200e3  # 200 kW
        dr_duration = 2.0  # 2 hours
        dr_price = 100.0  # $/MWh
        
        dr_revenue = self.dr_manager.calculate_dr_revenue(
            dr_reduction / 1000.0, dr_duration, dr_price
        )
        
        # Total revenue
        total_revenue = freq_revenue + dr_revenue
        self.assertGreater(total_revenue, 0)
        
        # Should be economically attractive (>$50/MW/year equivalent)
        annual_equivalent = total_revenue * (8760 / (duration_hours + dr_duration))
        self.assertGreater(annual_equivalent, 50.0)


if __name__ == "__main__":
    unittest.main()
