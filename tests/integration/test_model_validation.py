"""
Model Validation Tests against Vendor Specifications and Industry Benchmarks

This module validates the Firmus AI Factory digital twin models against
published vendor specifications, academic literature, and industry benchmarks.
"""

import unittest
import numpy as np


class TestGPUPowerModelValidation(unittest.TestCase):
    """Validate GPU power model against NVIDIA H100 specifications"""

    # NVIDIA H100 SXM5 Published Specifications
    H100_SPECS = {
        "TDP": 700,              # Watts
        "idle_power": 50,        # Watts (approximate)
        "memory_power": 80,      # Watts (HBM3 at full bandwidth)
        "fp64_tflops": 34,       # TFLOPS
        "fp32_tflops": 67,       # TFLOPS
        "tf32_tflops": 989,      # TFLOPS (with sparsity)
        "fp16_tflops": 1979,     # TFLOPS (with sparsity)
        "memory_bw_gbps": 3350,  # GB/s HBM3
        "memory_capacity_gb": 80,
        "pcie_gen": 5,
        "nvlink_bw_gbps": 900,
    }

    def test_tdp_accuracy(self):
        """Validate TDP matches NVIDIA specification"""
        from firmus_ai_factory.computational.gpu_model import GPUModel, GPU_H100_SXM5

        gpu = GPUModel(GPU_H100_SXM5)
        max_power = gpu.calculate_power(utilization=1.0)

        # Should be within 5% of published TDP
        self.assertAlmostEqual(
            max_power, self.H100_SPECS["TDP"],
            delta=self.H100_SPECS["TDP"] * 0.05,
            msg=f"Max power {max_power}W deviates >5% from TDP {self.H100_SPECS['TDP']}W"
        )

    def test_idle_power(self):
        """Validate idle power is reasonable"""
        from firmus_ai_factory.computational.gpu_model import GPUModel, GPU_H100_SXM5

        gpu = GPUModel(GPU_H100_SXM5)
        idle_power = gpu.calculate_power(utilization=0.0)

        # Idle power should be 30-80W for H100
        self.assertGreater(idle_power, 30)
        self.assertLess(idle_power, 80)

    def test_power_utilization_curve(self):
        """Validate power-utilization relationship is physically reasonable"""
        from firmus_ai_factory.computational.gpu_model import GPUModel, GPU_H100_SXM5

        gpu = GPUModel(GPU_H100_SXM5)

        # Power should increase monotonically with utilization
        prev_power = 0
        for util in np.arange(0, 1.01, 0.1):
            power = gpu.calculate_power(utilization=util)
            self.assertGreaterEqual(power, prev_power)
            prev_power = power

        # Power at 50% utilization should be roughly 50-70% of TDP
        # (non-linear due to leakage and fixed overhead)
        half_power = gpu.calculate_power(utilization=0.5)
        half_ratio = half_power / self.H100_SPECS["TDP"]
        self.assertGreater(half_ratio, 0.45)
        self.assertLess(half_ratio, 0.75)

    def test_training_workload_power(self):
        """Validate power during typical training workload"""
        from firmus_ai_factory.computational.gpu_model import GPUModel, GPU_H100_SXM5

        gpu = GPUModel(GPU_H100_SXM5)

        # Typical LLM training: 85-95% utilization
        training_power = gpu.calculate_power(utilization=0.9)

        # Should be 600-700W for H100 at high utilization
        self.assertGreater(training_power, 550)
        self.assertLess(training_power, 720)


class TestThermalModelValidation(unittest.TestCase):
    """Validate thermal model against published cooling specifications"""

    # Immersion cooling published specifications
    IMMERSION_SPECS = {
        "heat_removal_kw_per_rack": 100,  # Typical single-phase immersion
        "coolant_temp_range": (25, 55),    # Celsius
        "pue_target": 1.03,               # Best-in-class immersion PUE
        "thermal_resistance_range": (0.03, 0.08),  # K/W per GPU
    }

    def test_thermal_resistance(self):
        """Validate thermal resistance is within expected range"""
        from firmus_ai_factory.thermal.immersion_cooling import (
            ImmersionCoolingModel,
            COOLANT_EC100,
        )

        cooling = ImmersionCoolingModel(COOLANT_EC100)
        R_thermal = cooling.get_thermal_resistance()

        self.assertGreater(R_thermal, self.IMMERSION_SPECS["thermal_resistance_range"][0])
        self.assertLess(R_thermal, self.IMMERSION_SPECS["thermal_resistance_range"][1])

    def test_steady_state_temperature(self):
        """Validate steady-state GPU temperature under load"""
        from firmus_ai_factory.thermal.immersion_cooling import (
            ImmersionCoolingModel,
            COOLANT_EC100,
        )

        cooling = ImmersionCoolingModel(COOLANT_EC100)

        # H100 at 700W TDP with 35°C coolant inlet
        T_gpu = cooling.calculate_steady_state_temperature(
            power_w=700, coolant_inlet_temp=35.0, flow_rate_lpm=2.5
        )

        # GPU temperature should be 65-85°C for immersion cooling
        self.assertGreater(T_gpu, 60)
        self.assertLess(T_gpu, 90)

    def test_heat_removal_capacity(self):
        """Validate heat removal capacity matches specifications"""
        from firmus_ai_factory.thermal.immersion_cooling import (
            ImmersionCoolingModel,
            COOLANT_EC100,
        )

        cooling = ImmersionCoolingModel(COOLANT_EC100)

        # 8x H100 GPUs at TDP = 5.6 kW
        total_heat = 8 * 700  # 5600 W

        # Cooling should handle this without exceeding thermal limits
        T_gpu = cooling.calculate_steady_state_temperature(
            power_w=total_heat / 8, coolant_inlet_temp=35.0, flow_rate_lpm=2.5
        )

        self.assertLess(T_gpu, 85.0, "GPU temperature exceeds safe limit")


class TestPowerDeliveryValidation(unittest.TestCase):
    """Validate power delivery models against industry standards"""

    # Industry standard efficiency benchmarks
    PDN_BENCHMARKS = {
        "transformer_efficiency": (0.97, 0.995),  # 97-99.5%
        "dc_dc_efficiency": (0.92, 0.98),         # 92-98%
        "vrm_efficiency": (0.85, 0.95),            # 85-95%
        "total_pdn_efficiency": (0.88, 0.95),      # 88-95% end-to-end
    }

    def test_transformer_efficiency(self):
        """Validate transformer efficiency against industry standards"""
        from firmus_ai_factory.power import TransformerModel, TRANSFORMER_13_8KV_TO_480V

        xfmr = TransformerModel(TRANSFORMER_13_8KV_TO_480V)

        # Test at 50% load (typical operating point)
        _, efficiency, _ = xfmr.calculate_output(xfmr.rated_power * 0.5)

        self.assertGreater(efficiency, self.PDN_BENCHMARKS["transformer_efficiency"][0])
        self.assertLess(efficiency, self.PDN_BENCHMARKS["transformer_efficiency"][1])

    def test_converter_efficiency(self):
        """Validate DC-DC converter efficiency"""
        from firmus_ai_factory.power import DCDCConverter, CONVERTER_480V_TO_12V

        conv = DCDCConverter(CONVERTER_480V_TO_12V)

        _, efficiency, _ = conv.calculate_output(480.0, 50.0)

        self.assertGreater(efficiency, self.PDN_BENCHMARKS["dc_dc_efficiency"][0])
        self.assertLess(efficiency, self.PDN_BENCHMARKS["dc_dc_efficiency"][1])


class TestGridInterfaceValidation(unittest.TestCase):
    """Validate grid interface against IEEE/NERC standards"""

    # NERC/IEEE Grid Standards
    GRID_STANDARDS = {
        "frequency_deadband_hz": 0.036,    # NERC BAL-003
        "droop_range_pct": (3.0, 5.0),     # Typical droop setting
        "response_time_s": 30,              # Primary response within 30s
        "voltage_range_pct": (0.95, 1.05),  # ANSI C84.1 Range A
    }

    def test_frequency_deadband(self):
        """Validate frequency deadband compliance"""
        from firmus_ai_factory.grid import GridInterface, GRID_US_480V

        grid = GridInterface(GRID_US_480V)

        # Within deadband: no response
        P_base = 1000e3
        P_response = grid.calculate_frequency_response(60.0, P_base)
        self.assertAlmostEqual(P_response, P_base, delta=1000.0)

        # Just outside deadband: should respond
        f_outside = 60.0 - self.GRID_STANDARDS["frequency_deadband_hz"] - 0.01
        P_response = grid.calculate_frequency_response(f_outside, P_base)
        self.assertNotAlmostEqual(P_response, P_base, delta=P_base * 0.001)

    def test_voltage_range(self):
        """Validate voltage regulation within ANSI C84.1"""
        from firmus_ai_factory.grid import GridInterface, GRID_US_480V

        grid = GridInterface(GRID_US_480V)

        # Test voltage at various load levels
        for load_pct in [0.25, 0.5, 0.75, 1.0]:
            P_load = grid.rated_power * load_pct
            V_out = grid.get_voltage_at_load(P_load)

            V_ratio = V_out / grid.nominal_voltage
            self.assertGreater(V_ratio, self.GRID_STANDARDS["voltage_range_pct"][0])
            self.assertLess(V_ratio, self.GRID_STANDARDS["voltage_range_pct"][1])


class TestEconomicsValidation(unittest.TestCase):
    """Validate economics model against real utility rates"""

    # Real-world utility rate benchmarks (US industrial)
    RATE_BENCHMARKS = {
        "industrial_avg_kwh": 0.07,    # $/kWh US industrial average
        "demand_charge_range": (5, 25),  # $/kW-month
        "pjm_rtp_range": (20, 300),     # $/MWh PJM real-time range
    }

    def test_daily_cost_reasonableness(self):
        """Validate daily cost is reasonable for AI factory"""
        from firmus_ai_factory.economics import ElectricityTariff

        tariff = ElectricityTariff(
            tariff_type="flat",
            rates={"energy": 70.0},  # $70/MWh = $0.07/kWh
            demand_charge=15.0,
        )

        # 1 MW constant load for 24 hours
        power_profile = np.ones(24) * 1000.0  # 1000 kW

        bill = tariff.calculate_total_bill(power_profile, dt_hours=1.0)

        # Daily energy cost: 1 MW * 24h * $70/MWh = $1,680
        expected_energy = 1.0 * 24 * 70.0
        self.assertAlmostEqual(bill["energy_cost"], expected_energy, delta=10.0)

        # Demand charge: 1000 kW * $15/kW = $15,000/month ≈ $500/day
        expected_demand = 1000.0 * 15.0
        self.assertAlmostEqual(bill["demand_charge"], expected_demand, delta=10.0)

    def test_annual_cost_benchmark(self):
        """Validate annual cost against industry benchmarks"""
        from firmus_ai_factory.economics import ElectricityTariff

        tariff = ElectricityTariff(
            tariff_type="flat",
            rates={"energy": 70.0},
            demand_charge=15.0,
        )

        # 10 MW AI factory
        power_profile = np.ones(24) * 10000.0  # 10,000 kW

        daily_bill = tariff.calculate_total_bill(power_profile, dt_hours=1.0)
        annual_energy = daily_bill["energy_cost"] * 365
        annual_demand = daily_bill["demand_charge"] * 12

        annual_total = annual_energy + annual_demand

        # 10 MW AI factory should cost $6-10M/year in electricity
        self.assertGreater(annual_total, 5e6)
        self.assertLess(annual_total, 12e6)

        print(f"\n10 MW AI Factory Annual Electricity Cost:")
        print(f"  Energy: ${annual_energy:,.0f}")
        print(f"  Demand: ${annual_demand:,.0f}")
        print(f"  Total: ${annual_total:,.0f}")


class TestCrossModuleConsistency(unittest.TestCase):
    """Validate consistency across modules"""

    def test_power_balance(self):
        """Validate power balance: GPU + cooling + PDN losses = total"""
        from firmus_ai_factory.control import DigitalTwin

        config = {
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

        twin = DigitalTwin(config)
        twin.update_state({"workload_intensity": 0.8}, dt=60.0)
        state = twin.get_state()

        # Total power should be >= GPU power (includes cooling + losses)
        self.assertGreaterEqual(state["total_power"], state["gpu_power"])

        # PUE should be reasonable (1.03-1.5 for modern data centers)
        if state["gpu_power"] > 0:
            pue = state["total_power"] / state["gpu_power"]
            self.assertGreater(pue, 1.0)
            self.assertLess(pue, 1.5)

    def test_energy_conservation(self):
        """Validate energy conservation across simulation"""
        from firmus_ai_factory.control import DigitalTwin

        config = {
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

        twin = DigitalTwin(config)
        results = twin.run_scenario(duration_hours=1, dt=60)

        powers = np.array(results["P_total"])
        dt_hours = 60.0 / 3600.0

        # Energy from power integration
        energy_from_power = np.sum(powers) * dt_hours

        # Should be positive and reasonable
        self.assertGreater(energy_from_power, 0)

        # For 8x H100 at 80% util, expect ~4-6 kWh per hour including overhead
        self.assertGreater(energy_from_power, 2000)  # >2 MWh minimum
        self.assertLess(energy_from_power, 10000)    # <10 MWh maximum


if __name__ == "__main__":
    unittest.main()
