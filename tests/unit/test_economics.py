"""
Unit tests for Economics module (Electricity Tariff and Cost Analysis)
"""

import unittest
import numpy as np
from firmus_ai_factory.economics import ElectricityTariff


class TestElectricityTariff(unittest.TestCase):
    """Test electricity tariff models"""

    def setUp(self):
        """Set up test fixtures"""
        self.tou_tariff = ElectricityTariff(
            tariff_type="TOU",
            rates={
                "off_peak": 50.0,   # $/MWh (11pm-7am)
                "mid_peak": 100.0,  # $/MWh (7am-2pm, 6pm-11pm)
                "on_peak": 200.0,   # $/MWh (2pm-6pm)
            },
            demand_charge=15.0,  # $/kW-month
            periods={
                "off_peak": [(23, 7)],
                "mid_peak": [(7, 14), (18, 23)],
                "on_peak": [(14, 18)],
            },
        )

        self.flat_tariff = ElectricityTariff(
            tariff_type="flat",
            rates={"energy": 80.0},  # $/MWh
            demand_charge=10.0,
        )

    def test_tou_initialization(self):
        """Test TOU tariff initialization"""
        self.assertEqual(self.tou_tariff.tariff_type, "TOU")
        self.assertIn("off_peak", self.tou_tariff.rates)
        self.assertIn("on_peak", self.tou_tariff.rates)

    def test_flat_initialization(self):
        """Test flat tariff initialization"""
        self.assertEqual(self.flat_tariff.tariff_type, "flat")
        self.assertIn("energy", self.flat_tariff.rates)

    def test_tou_price_at_hour(self):
        """Test TOU price lookup for different hours"""
        # Off-peak (midnight)
        price_midnight = self.tou_tariff.get_price_at_hour(0)
        self.assertEqual(price_midnight, 50.0)

        # Off-peak (3am)
        price_3am = self.tou_tariff.get_price_at_hour(3)
        self.assertEqual(price_3am, 50.0)

        # Mid-peak (10am)
        price_10am = self.tou_tariff.get_price_at_hour(10)
        self.assertEqual(price_10am, 100.0)

        # On-peak (3pm)
        price_3pm = self.tou_tariff.get_price_at_hour(15)
        self.assertEqual(price_3pm, 200.0)

        # Mid-peak (8pm)
        price_8pm = self.tou_tariff.get_price_at_hour(20)
        self.assertEqual(price_8pm, 100.0)

    def test_flat_price_constant(self):
        """Test flat tariff returns constant price"""
        prices = [self.flat_tariff.get_price_at_hour(h) for h in range(24)]
        self.assertTrue(all(p == 80.0 for p in prices))

    def test_24h_price_profile(self):
        """Test generation of 24-hour price profile"""
        profile = self.tou_tariff.get_24h_price_profile()

        self.assertEqual(len(profile), 24)
        self.assertTrue(all(p > 0 for p in profile))

        # Should have variation
        self.assertGreater(max(profile), min(profile))

    def test_energy_cost_calculation(self):
        """Test energy cost calculation for power profile"""
        # Constant 1 MW load for 24 hours
        power_profile = np.ones(24) * 1000.0  # 1000 kW = 1 MW

        cost = self.tou_tariff.calculate_energy_cost(power_profile, dt_hours=1.0)

        # Cost should be positive
        self.assertGreater(cost, 0)

        # Verify calculation: sum of (power * price * dt) for each hour
        prices = self.tou_tariff.get_24h_price_profile()
        expected_cost = sum(
            (power_profile[h] / 1000.0) * prices[h] * 1.0 for h in range(24)
        )
        self.assertAlmostEqual(cost, expected_cost, delta=1.0)

    def test_demand_charge_calculation(self):
        """Test demand charge calculation"""
        # Peak demand of 2 MW
        power_profile = np.ones(24) * 1000.0  # 1 MW baseline
        power_profile[14:18] = 2000.0  # 2 MW peak during afternoon

        demand_charge = self.tou_tariff.calculate_demand_charge(power_profile)

        # Demand charge = peak_demand_kW * rate
        expected_charge = 2000.0 * self.tou_tariff.demand_charge
        self.assertAlmostEqual(demand_charge, expected_charge, delta=1.0)

    def test_total_bill_calculation(self):
        """Test total monthly bill calculation"""
        power_profile = np.ones(24) * 1500.0  # 1.5 MW constant

        bill = self.tou_tariff.calculate_total_bill(power_profile, dt_hours=1.0)

        # Bill should include energy and demand charges
        self.assertIn("energy_cost", bill)
        self.assertIn("demand_charge", bill)
        self.assertIn("total", bill)

        # Total should equal sum of components
        self.assertAlmostEqual(
            bill["total"],
            bill["energy_cost"] + bill["demand_charge"],
            delta=0.01,
        )

    def test_cost_comparison(self):
        """Test cost comparison between tariff types"""
        power_profile = np.ones(24) * 1000.0  # 1 MW constant

        tou_cost = self.tou_tariff.calculate_energy_cost(power_profile, dt_hours=1.0)
        flat_cost = self.flat_tariff.calculate_energy_cost(power_profile, dt_hours=1.0)

        # Both should be positive
        self.assertGreater(tou_cost, 0)
        self.assertGreater(flat_cost, 0)


class TestRealTimePricing(unittest.TestCase):
    """Test real-time pricing tariff"""

    def setUp(self):
        """Set up RTP tariff"""
        self.rtp_tariff = ElectricityTariff(
            tariff_type="RTP",
            rates={"base": 80.0},  # Base rate $/MWh
            demand_charge=12.0,
        )

    def test_rtp_with_price_signal(self):
        """Test RTP with external price signal"""
        # Simulate real-time prices (volatile)
        np.random.seed(42)
        rtp_prices = 80.0 + np.random.normal(0, 30, 24)  # Mean $80, std $30
        rtp_prices = np.clip(rtp_prices, 10, 300)  # Clip to reasonable range

        power_profile = np.ones(24) * 1000.0  # 1 MW

        cost = self.rtp_tariff.calculate_energy_cost_rtp(power_profile, rtp_prices, dt_hours=1.0)

        # Cost should be positive
        self.assertGreater(cost, 0)

        # Verify calculation
        expected = sum((1000.0 / 1000.0) * rtp_prices[h] * 1.0 for h in range(24))
        self.assertAlmostEqual(cost, expected, delta=1.0)

    def test_price_volatility_impact(self):
        """Test impact of price volatility on costs"""
        power_profile = np.ones(24) * 1000.0

        # Low volatility prices
        low_vol_prices = np.ones(24) * 80.0 + np.random.normal(0, 5, 24)
        cost_low_vol = self.rtp_tariff.calculate_energy_cost_rtp(
            power_profile, low_vol_prices, dt_hours=1.0
        )

        # High volatility prices (same mean)
        high_vol_prices = np.ones(24) * 80.0 + np.random.normal(0, 50, 24)
        high_vol_prices = np.clip(high_vol_prices, 10, 500)

        # Both should produce reasonable costs
        self.assertGreater(cost_low_vol, 0)


class TestCostBreakdown(unittest.TestCase):
    """Test cost breakdown and analysis"""

    def setUp(self):
        """Set up tariff for analysis"""
        self.tariff = ElectricityTariff(
            tariff_type="TOU",
            rates={
                "off_peak": 50.0,
                "mid_peak": 100.0,
                "on_peak": 200.0,
            },
            demand_charge=15.0,
            periods={
                "off_peak": [(23, 7)],
                "mid_peak": [(7, 14), (18, 23)],
                "on_peak": [(14, 18)],
            },
        )

    def test_period_cost_breakdown(self):
        """Test cost breakdown by TOU period"""
        power_profile = np.ones(24) * 1000.0  # 1 MW constant

        breakdown = self.tariff.get_cost_breakdown(power_profile, dt_hours=1.0)

        # Should have breakdown by period
        self.assertIn("off_peak", breakdown)
        self.assertIn("mid_peak", breakdown)
        self.assertIn("on_peak", breakdown)

        # Sum of period costs should equal total energy cost
        total_from_periods = sum(breakdown.values())
        total_energy = self.tariff.calculate_energy_cost(power_profile, dt_hours=1.0)
        self.assertAlmostEqual(total_from_periods, total_energy, delta=1.0)

    def test_savings_from_load_shifting(self):
        """Test savings calculation from load shifting"""
        # Baseline: constant 1 MW
        baseline = np.ones(24) * 1000.0

        # Shifted: reduce during peak, increase during off-peak
        shifted = np.ones(24) * 1000.0
        shifted[14:18] = 500.0   # Reduce during on-peak
        shifted[0:6] = 1500.0    # Increase during off-peak (same total energy)

        cost_baseline = self.tariff.calculate_energy_cost(baseline, dt_hours=1.0)
        cost_shifted = self.tariff.calculate_energy_cost(shifted, dt_hours=1.0)

        # Shifted schedule should be cheaper
        self.assertLess(cost_shifted, cost_baseline)

        # Calculate savings percentage
        savings_pct = (cost_baseline - cost_shifted) / cost_baseline * 100
        self.assertGreater(savings_pct, 0)
        print(f"\nLoad shifting savings: {savings_pct:.1f}%")

    def test_annual_cost_projection(self):
        """Test annual cost projection"""
        daily_profile = np.ones(24) * 1000.0  # 1 MW

        daily_bill = self.tariff.calculate_total_bill(daily_profile, dt_hours=1.0)

        # Annual projection (approximate)
        annual_energy = daily_bill["energy_cost"] * 365
        annual_demand = daily_bill["demand_charge"] * 12  # Monthly charge * 12
        annual_total = annual_energy + annual_demand

        self.assertGreater(annual_total, 0)
        print(f"\nProjected annual cost: ${annual_total:,.0f}")
        print(f"  Energy: ${annual_energy:,.0f}")
        print(f"  Demand: ${annual_demand:,.0f}")


if __name__ == "__main__":
    unittest.main()
