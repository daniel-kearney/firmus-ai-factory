"""
Unit tests for Power Delivery Network (PDN) module
"""

import unittest
import numpy as np
from firmus_ai_factory.power import (
    TransformerModel,
    DCDCConverter,
    VoltageRegulatorModule,
    TRANSFORMER_13_8KV_TO_480V,
    TRANSFORMER_34_5KV_TO_480V,
    CONVERTER_480V_TO_12V,
    CONVERTER_12V_TO_1V,
)


class TestTransformerModel(unittest.TestCase):
    """Test transformer models"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.transformer = TransformerModel(TRANSFORMER_13_8KV_TO_480V)
    
    def test_initialization(self):
        """Test transformer initialization"""
        self.assertEqual(self.transformer.V_primary, 13800.0)
        self.assertEqual(self.transformer.V_secondary, 480.0)
        self.assertGreater(self.transformer.rated_power, 0)
    
    def test_voltage_regulation(self):
        """Test voltage regulation under load"""
        P_load = 500e3  # 500 kW
        V_out, efficiency, losses = self.transformer.calculate_output(P_load)
        
        # Output voltage should be close to rated (within regulation)
        self.assertAlmostEqual(V_out, 480.0, delta=10.0)
        
        # Efficiency should be high (>95% for transformers)
        self.assertGreater(efficiency, 0.95)
        self.assertLess(efficiency, 1.0)
        
        # Losses should be positive and reasonable
        self.assertGreater(losses, 0)
        self.assertLess(losses, P_load * 0.05)  # <5% losses
    
    def test_overload_protection(self):
        """Test behavior under overload"""
        P_overload = self.transformer.rated_power * 1.5
        V_out, efficiency, losses = self.transformer.calculate_output(P_overload)
        
        # Should still produce output but with degraded performance
        self.assertGreater(V_out, 0)
        self.assertLess(efficiency, 0.95)
    
    def test_impedance_calculation(self):
        """Test impedance calculation"""
        Z = self.transformer.get_impedance()
        
        # Impedance should be positive and reasonable
        self.assertGreater(Z, 0)
        self.assertLess(Z, 1.0)  # Typical transformer impedance


class TestDCDCConverter(unittest.TestCase):
    """Test DC-DC converter models"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.converter = DCDCConverter(CONVERTER_480V_TO_12V)
    
    def test_initialization(self):
        """Test converter initialization"""
        self.assertEqual(self.converter.V_in_nom, 480.0)
        self.assertEqual(self.converter.V_out_nom, 12.0)
        self.assertGreater(self.converter.rated_power, 0)
    
    def test_buck_conversion(self):
        """Test buck converter operation"""
        V_in = 480.0
        I_out = 100.0  # 100 A at 12V = 1.2 kW
        
        V_out, efficiency, losses = self.converter.calculate_output(V_in, I_out)
        
        # Output voltage should be close to nominal
        self.assertAlmostEqual(V_out, 12.0, delta=0.5)
        
        # Efficiency should be good (>90% for modern converters)
        self.assertGreater(efficiency, 0.90)
        
        # Power balance check
        P_out = V_out * I_out
        P_in = P_out + losses
        self.assertAlmostEqual(P_in / P_out, 1 / efficiency, delta=0.01)
    
    def test_input_voltage_variation(self):
        """Test response to input voltage variation"""
        I_out = 50.0
        
        for V_in in [450.0, 480.0, 510.0]:  # ±30V variation
            V_out, eff, losses = self.converter.calculate_output(V_in, I_out)
            
            # Output should be regulated
            self.assertAlmostEqual(V_out, 12.0, delta=0.5)
            
            # Efficiency should remain reasonable
            self.assertGreater(eff, 0.85)
    
    def test_load_regulation(self):
        """Test output voltage regulation across load range"""
        V_in = 480.0
        
        for load_pct in [0.1, 0.5, 1.0]:  # 10%, 50%, 100% load
            I_out = (self.converter.rated_power / 12.0) * load_pct
            V_out, eff, losses = self.converter.calculate_output(V_in, I_out)
            
            # Voltage regulation should be tight
            self.assertAlmostEqual(V_out, 12.0, delta=0.3)


class TestVoltageRegulatorModule(unittest.TestCase):
    """Test voltage regulator module (VRM)"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.vrm = VoltageRegulatorModule(
            V_in=12.0,
            V_out=1.0,
            num_phases=8,
            switching_freq=1e6,  # 1 MHz
            rated_current=400.0  # 400 A
        )
    
    def test_initialization(self):
        """Test VRM initialization"""
        self.assertEqual(self.vrm.V_in, 12.0)
        self.assertEqual(self.vrm.V_out, 1.0)
        self.assertEqual(self.vrm.num_phases, 8)
    
    def test_steady_state_operation(self):
        """Test VRM steady-state performance"""
        I_load = 300.0  # 300 A
        
        V_out, efficiency, losses = self.vrm.calculate_output(I_load)
        
        # Output voltage should be tightly regulated
        self.assertAlmostEqual(V_out, 1.0, delta=0.02)  # ±20 mV
        
        # Efficiency should be reasonable for VRM
        self.assertGreater(efficiency, 0.80)
        
        # Power check
        P_out = V_out * I_load
        self.assertGreater(P_out, 0)
    
    def test_transient_response(self):
        """Test VRM transient response to load step"""
        I_initial = 100.0
        I_final = 300.0
        di_dt = 1000.0  # 1000 A/µs (typical GPU load step)
        
        V_droop, t_settle = self.vrm.calculate_transient_response(
            I_initial, I_final, di_dt
        )
        
        # Voltage droop should be limited
        self.assertLess(abs(V_droop), 0.1)  # <100 mV droop
        
        # Settling time should be reasonable
        self.assertGreater(t_settle, 0)
        self.assertLess(t_settle, 100e-6)  # <100 µs
    
    def test_impedance(self):
        """Test output impedance calculation"""
        Z_out = self.vrm.get_output_impedance()
        
        # Output impedance should be very low for VRM
        self.assertGreater(Z_out, 0)
        self.assertLess(Z_out, 0.01)  # <10 mΩ typical
    
    def test_current_sharing(self):
        """Test current sharing across phases"""
        I_total = 320.0  # 320 A total
        
        phase_currents = self.vrm.get_phase_currents(I_total)
        
        # Should have correct number of phases
        self.assertEqual(len(phase_currents), self.vrm.num_phases)
        
        # Total current should match
        self.assertAlmostEqual(sum(phase_currents), I_total, delta=1.0)
        
        # Current sharing should be balanced (within 10%)
        avg_current = I_total / self.vrm.num_phases
        for I_phase in phase_currents:
            self.assertAlmostEqual(I_phase, avg_current, delta=avg_current * 0.1)


class TestPowerDeliveryChain(unittest.TestCase):
    """Test complete power delivery chain"""
    
    def setUp(self):
        """Set up complete PDN chain"""
        self.transformer = TransformerModel(TRANSFORMER_13_8KV_TO_480V)
        self.converter_480_12 = DCDCConverter(CONVERTER_480V_TO_12V)
        self.converter_12_1 = DCDCConverter(CONVERTER_12V_TO_1V)
    
    def test_end_to_end_efficiency(self):
        """Test overall PDN efficiency"""
        # Start with 100 kW GPU load at 1V
        P_gpu = 100e3  # 100 kW
        V_gpu = 1.0
        I_gpu = P_gpu / V_gpu  # 100,000 A
        
        # Work backwards through PDN
        # Stage 3: 12V -> 1V
        V_12v, eff_12_1, loss_12_1 = self.converter_12_1.calculate_output(12.0, I_gpu)
        P_12v = P_gpu + loss_12_1
        I_12v = P_12v / 12.0
        
        # Stage 2: 480V -> 12V
        V_480v, eff_480_12, loss_480_12 = self.converter_480_12.calculate_output(480.0, I_12v)
        P_480v = P_12v + loss_480_12
        
        # Stage 1: 13.8kV -> 480V
        V_grid, eff_xfmr, loss_xfmr = self.transformer.calculate_output(P_480v)
        P_grid = P_480v + loss_xfmr
        
        # Overall efficiency
        eff_total = P_gpu / P_grid
        
        # Total efficiency should be reasonable (>85%)
        self.assertGreater(eff_total, 0.85)
        
        # Total losses
        loss_total = loss_xfmr + loss_480_12 + loss_12_1
        self.assertAlmostEqual(loss_total, P_grid - P_gpu, delta=100.0)


if __name__ == "__main__":
    unittest.main()
