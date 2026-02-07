"""
Sensor Emulator for Firmus AI Factory Digital Twin

This module emulates real AI factory sensors using public datasets,
enabling closed-loop control testing without physical hardware.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SensorReading:
    """Container for sensor measurements"""
    timestamp: float
    gpu_power: float  # Watts
    gpu_utilization: float  # Percent (0-100)
    gpu_temperature: float  # Celsius
    cpu_power: Optional[float] = None  # Watts
    memory_usage: Optional[float] = None  # GB
    cooling_flow_rate: Optional[float] = None  # L/min
    coolant_temp_inlet: Optional[float] = None  # Celsius
    coolant_temp_outlet: Optional[float] = None  # Celsius
    grid_frequency: Optional[float] = 60.0  # Hz
    grid_voltage: Optional[float] = 480.0  # Volts


class SensorEmulator:
    """
    Emulate AI factory sensors using dataset-driven models
    
    This class provides realistic sensor readings based on public datasets,
    enabling closed-loop control testing without physical hardware access.
    """
    
    def __init__(self, dataset: Optional[pd.DataFrame] = None, noise_level: float = 0.02):
        """
        Initialize sensor emulator
        
        Args:
            dataset: Optional DataFrame with historical sensor data
            noise_level: Measurement noise as fraction of signal (default 2%)
        """
        self.dataset = dataset
        self.noise_level = noise_level
        self.current_index = 0
        self.time_offset = 0.0
        
        # Default sensor characteristics
        self.sensor_specs = {
            "gpu_power": {"range": (0, 1000), "resolution": 0.1, "update_rate": 10},  # 10 Hz
            "gpu_temp": {"range": (20, 95), "resolution": 0.1, "update_rate": 1},  # 1 Hz
            "gpu_util": {"range": (0, 100), "resolution": 1.0, "update_rate": 10},
            "cooling_flow": {"range": (0, 10), "resolution": 0.01, "update_rate": 1},
            "coolant_temp": {"range": (20, 60), "resolution": 0.1, "update_rate": 1},
            "grid_freq": {"range": (59.5, 60.5), "resolution": 0.001, "update_rate": 60},  # 60 Hz
        }
        
        logger.info(f"Sensor emulator initialized with noise level: {noise_level*100:.1f}%")
    
    def load_dataset(self, dataset: pd.DataFrame) -> None:
        """
        Load dataset for sensor emulation
        
        Args:
            dataset: DataFrame with columns matching sensor types
        """
        self.dataset = dataset
        self.current_index = 0
        logger.info(f"Loaded dataset with {len(dataset)} samples")
    
    def add_measurement_noise(self, value: float, sensor_type: str) -> float:
        """
        Add realistic measurement noise to sensor reading
        
        Args:
            value: True sensor value
            sensor_type: Type of sensor (for range validation)
            
        Returns:
            Noisy measurement
        """
        # Gaussian noise proportional to signal
        noise = np.random.normal(0, self.noise_level * abs(value))
        noisy_value = value + noise
        
        # Clip to sensor range
        if sensor_type in self.sensor_specs:
            min_val, max_val = self.sensor_specs[sensor_type]["range"]
            noisy_value = np.clip(noisy_value, min_val, max_val)
        
        return noisy_value
    
    def get_reading_from_dataset(self, timestamp: float) -> SensorReading:
        """
        Get sensor reading from dataset at specified timestamp
        
        Args:
            timestamp: Simulation time (seconds)
            
        Returns:
            SensorReading with emulated measurements
        """
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")
        
        # Find closest dataset entry (simple nearest-neighbor)
        if self.current_index >= len(self.dataset):
            self.current_index = 0  # Loop back
        
        row = self.dataset.iloc[self.current_index]
        self.current_index += 1
        
        # Map dataset columns to sensor readings (adjust based on actual dataset)
        reading = SensorReading(
            timestamp=timestamp,
            gpu_power=self.add_measurement_noise(
                row.get("gpu_power", row.get("power_w", 500.0)), "gpu_power"
            ),
            gpu_utilization=self.add_measurement_noise(
                row.get("gpu_utilization", row.get("utilization", 80.0)), "gpu_util"
            ),
            gpu_temperature=self.add_measurement_noise(
                row.get("gpu_temperature", row.get("temperature", 75.0)), "gpu_temp"
            ),
            cpu_power=self.add_measurement_noise(
                row.get("cpu_power", 200.0), "gpu_power"
            ) if "cpu_power" in row else None,
            memory_usage=row.get("memory_usage", None),
            cooling_flow_rate=self.add_measurement_noise(2.5, "cooling_flow"),
            coolant_temp_inlet=self.add_measurement_noise(35.0, "coolant_temp"),
            coolant_temp_outlet=self.add_measurement_noise(45.0, "coolant_temp"),
            grid_frequency=self.add_measurement_noise(60.0, "grid_freq"),
            grid_voltage=self.add_measurement_noise(480.0, "gpu_power"),
        )
        
        return reading
    
    def get_synthetic_reading(self, timestamp: float, workload_intensity: float = 0.8) -> SensorReading:
        """
        Generate synthetic sensor reading based on workload model
        
        Args:
            timestamp: Simulation time (seconds)
            workload_intensity: Workload intensity (0-1)
            
        Returns:
            SensorReading with synthetic measurements
        """
        # Base power model: P = P_idle + (P_max - P_idle) * utilization
        P_idle = 50.0  # Watts
        P_max = 700.0  # Watts (H100 TDP)
        
        # Add temporal variation (simulates batch processing)
        phase = np.sin(2 * np.pi * timestamp / 60.0)  # 60-second cycle
        utilization = workload_intensity * (0.9 + 0.1 * phase)
        utilization = np.clip(utilization, 0, 1)
        
        gpu_power = P_idle + (P_max - P_idle) * utilization
        gpu_power = self.add_measurement_noise(gpu_power, "gpu_power")
        
        # Temperature model: T = T_ambient + thermal_resistance * P
        T_ambient = 25.0  # Celsius
        R_thermal = 0.06  # K/W (effective thermal resistance)
        gpu_temp = T_ambient + R_thermal * gpu_power
        gpu_temp = self.add_measurement_noise(gpu_temp, "gpu_temp")
        
        # Cooling system response
        coolant_inlet = 35.0  # Celsius
        coolant_outlet = coolant_inlet + (gpu_power / 1000.0) * 2.0  # Simplified heat transfer
        
        # Grid parameters with small fluctuations
        grid_freq = 60.0 + np.random.normal(0, 0.01)  # ±10 mHz typical
        grid_voltage = 480.0 + np.random.normal(0, 2.0)  # ±2V typical
        
        reading = SensorReading(
            timestamp=timestamp,
            gpu_power=gpu_power,
            gpu_utilization=utilization * 100.0,
            gpu_temperature=gpu_temp,
            cpu_power=self.add_measurement_noise(200.0, "gpu_power"),
            memory_usage=32.0,  # GB
            cooling_flow_rate=self.add_measurement_noise(2.5, "cooling_flow"),
            coolant_temp_inlet=self.add_measurement_noise(coolant_inlet, "coolant_temp"),
            coolant_temp_outlet=self.add_measurement_noise(coolant_outlet, "coolant_temp"),
            grid_frequency=grid_freq,
            grid_voltage=grid_voltage,
        )
        
        return reading
    
    def get_reading(self, timestamp: float, mode: str = "synthetic", **kwargs) -> SensorReading:
        """
        Get sensor reading using specified mode
        
        Args:
            timestamp: Simulation time (seconds)
            mode: "synthetic" or "dataset"
            **kwargs: Additional arguments passed to mode-specific function
            
        Returns:
            SensorReading
        """
        if mode == "dataset":
            return self.get_reading_from_dataset(timestamp)
        elif mode == "synthetic":
            return self.get_synthetic_reading(timestamp, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def simulate_sensor_stream(
        self, 
        duration: float, 
        dt: float = 0.1, 
        mode: str = "synthetic"
    ) -> List[SensorReading]:
        """
        Simulate continuous sensor stream
        
        Args:
            duration: Simulation duration (seconds)
            dt: Time step (seconds)
            mode: "synthetic" or "dataset"
            
        Returns:
            List of SensorReading objects
        """
        timestamps = np.arange(0, duration, dt)
        readings = []
        
        logger.info(f"Simulating {len(timestamps)} sensor readings over {duration}s")
        
        for t in timestamps:
            reading = self.get_reading(t, mode=mode)
            readings.append(reading)
        
        return readings
    
    def readings_to_dataframe(self, readings: List[SensorReading]) -> pd.DataFrame:
        """
        Convert list of sensor readings to DataFrame
        
        Args:
            readings: List of SensorReading objects
            
        Returns:
            DataFrame with sensor data
        """
        data = []
        for r in readings:
            data.append({
                "timestamp": r.timestamp,
                "gpu_power": r.gpu_power,
                "gpu_utilization": r.gpu_utilization,
                "gpu_temperature": r.gpu_temperature,
                "cpu_power": r.cpu_power,
                "memory_usage": r.memory_usage,
                "cooling_flow_rate": r.cooling_flow_rate,
                "coolant_temp_inlet": r.coolant_temp_inlet,
                "coolant_temp_outlet": r.coolant_temp_outlet,
                "grid_frequency": r.grid_frequency,
                "grid_voltage": r.grid_voltage,
            })
        
        return pd.DataFrame(data)


if __name__ == "__main__":
    # Example usage
    emulator = SensorEmulator(noise_level=0.02)
    
    # Generate synthetic sensor stream
    readings = emulator.simulate_sensor_stream(duration=60.0, dt=0.1, mode="synthetic")
    
    # Convert to DataFrame
    df = emulator.readings_to_dataframe(readings)
    
    print(f"\nGenerated {len(df)} sensor readings")
    print(f"\nSummary statistics:")
    print(df.describe())
    
    print(f"\nSample readings:")
    print(df.head(10))
