"""
Unit tests for Sensor Emulator utility
"""

import unittest
import numpy as np
import pandas as pd
from firmus_ai_factory.utils.sensor_emulator import SensorEmulator, SensorReading


class TestSensorEmulator(unittest.TestCase):
    """Test sensor emulation capabilities"""

    def setUp(self):
        """Set up test fixtures"""
        self.emulator = SensorEmulator(noise_level=0.02)

    def test_initialization(self):
        """Test emulator initialization"""
        self.assertEqual(self.emulator.noise_level, 0.02)
        self.assertIsNotNone(self.emulator.sensor_specs)

    def test_synthetic_reading(self):
        """Test synthetic sensor reading generation"""
        reading = self.emulator.get_synthetic_reading(timestamp=0.0, workload_intensity=0.8)

        self.assertIsInstance(reading, SensorReading)
        self.assertEqual(reading.timestamp, 0.0)
        self.assertGreater(reading.gpu_power, 0)
        self.assertGreater(reading.gpu_utilization, 0)
        self.assertLessEqual(reading.gpu_utilization, 100.0)
        self.assertGreater(reading.gpu_temperature, 0)
        self.assertLess(reading.gpu_temperature, 100)

    def test_workload_intensity_effect(self):
        """Test that workload intensity affects sensor readings"""
        reading_low = self.emulator.get_synthetic_reading(0.0, workload_intensity=0.1)
        reading_high = self.emulator.get_synthetic_reading(0.0, workload_intensity=0.9)

        # Higher workload should produce higher power and temperature
        self.assertGreater(reading_high.gpu_power, reading_low.gpu_power)
        self.assertGreater(reading_high.gpu_temperature, reading_low.gpu_temperature)

    def test_measurement_noise(self):
        """Test measurement noise is applied"""
        np.random.seed(42)
        readings = []
        for _ in range(100):
            r = self.emulator.get_synthetic_reading(0.0, workload_intensity=0.8)
            readings.append(r.gpu_power)

        # Should have variation (noise)
        std_dev = np.std(readings)
        self.assertGreater(std_dev, 0)

        # Noise should be proportional to signal (within expected range)
        mean_power = np.mean(readings)
        relative_noise = std_dev / mean_power
        self.assertLess(relative_noise, 0.10)  # <10% relative noise

    def test_sensor_stream(self):
        """Test continuous sensor stream generation"""
        readings = self.emulator.simulate_sensor_stream(
            duration=10.0, dt=0.1, mode="synthetic"
        )

        self.assertEqual(len(readings), 100)  # 10s / 0.1s = 100

        # All readings should be valid
        for r in readings:
            self.assertIsInstance(r, SensorReading)
            self.assertGreater(r.gpu_power, 0)

    def test_readings_to_dataframe(self):
        """Test conversion of readings to DataFrame"""
        readings = self.emulator.simulate_sensor_stream(
            duration=5.0, dt=0.5, mode="synthetic"
        )

        df = self.emulator.readings_to_dataframe(readings)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 10)  # 5s / 0.5s = 10
        self.assertIn("gpu_power", df.columns)
        self.assertIn("gpu_temperature", df.columns)
        self.assertIn("gpu_utilization", df.columns)
        self.assertIn("grid_frequency", df.columns)

    def test_grid_frequency_realism(self):
        """Test grid frequency readings are realistic"""
        readings = self.emulator.simulate_sensor_stream(
            duration=60.0, dt=1.0, mode="synthetic"
        )

        frequencies = [r.grid_frequency for r in readings]

        # Mean should be close to 60 Hz
        self.assertAlmostEqual(np.mean(frequencies), 60.0, delta=0.1)

        # Standard deviation should be small (typical grid: ~10-20 mHz)
        self.assertLess(np.std(frequencies), 0.1)

    def test_dataset_mode(self):
        """Test dataset-driven sensor emulation"""
        # Create mock dataset
        mock_data = pd.DataFrame({
            "gpu_power": np.random.uniform(300, 700, 100),
            "gpu_utilization": np.random.uniform(50, 100, 100),
            "gpu_temperature": np.random.uniform(60, 85, 100),
        })

        self.emulator.load_dataset(mock_data)

        reading = self.emulator.get_reading(0.0, mode="dataset")

        self.assertIsInstance(reading, SensorReading)
        self.assertGreater(reading.gpu_power, 0)

    def test_dataset_looping(self):
        """Test that dataset loops when exhausted"""
        mock_data = pd.DataFrame({
            "gpu_power": [500.0, 600.0, 700.0],
            "gpu_utilization": [70.0, 80.0, 90.0],
            "gpu_temperature": [70.0, 75.0, 80.0],
        })

        self.emulator.load_dataset(mock_data)

        # Read more than dataset length
        readings = []
        for i in range(6):
            r = self.emulator.get_reading(float(i), mode="dataset")
            readings.append(r)

        # Should have 6 readings (looped)
        self.assertEqual(len(readings), 6)

    def test_noise_level_zero(self):
        """Test emulator with zero noise"""
        emulator = SensorEmulator(noise_level=0.0)

        # Multiple readings at same conditions should be identical
        r1 = emulator.get_synthetic_reading(0.0, workload_intensity=0.5)
        r2 = emulator.get_synthetic_reading(0.0, workload_intensity=0.5)

        # With zero noise, power should be deterministic
        # (Note: grid frequency still has random component)
        self.assertAlmostEqual(r1.gpu_power, r2.gpu_power, delta=0.01)


class TestSensorReadingDataclass(unittest.TestCase):
    """Test SensorReading dataclass"""

    def test_creation(self):
        """Test SensorReading creation"""
        reading = SensorReading(
            timestamp=1.0,
            gpu_power=500.0,
            gpu_utilization=80.0,
            gpu_temperature=75.0,
        )

        self.assertEqual(reading.timestamp, 1.0)
        self.assertEqual(reading.gpu_power, 500.0)
        self.assertEqual(reading.gpu_utilization, 80.0)
        self.assertEqual(reading.gpu_temperature, 75.0)

    def test_optional_fields(self):
        """Test optional fields have defaults"""
        reading = SensorReading(
            timestamp=0.0,
            gpu_power=500.0,
            gpu_utilization=80.0,
            gpu_temperature=75.0,
        )

        self.assertIsNone(reading.cpu_power)
        self.assertIsNone(reading.memory_usage)
        self.assertEqual(reading.grid_frequency, 60.0)
        self.assertEqual(reading.grid_voltage, 480.0)


if __name__ == "__main__":
    unittest.main()
