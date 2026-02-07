# Firmus AI Factory — Testing Guide

## Overview

This guide covers the complete testing infrastructure for the Firmus AI Factory digital twin platform. The testing framework enables validation of all mathematical models, closed-loop control algorithms, and economic optimization strategies without requiring access to physical AI factory hardware.

## Architecture

The testing infrastructure consists of three layers:

| Layer | Purpose | Location |
|-------|---------|----------|
| **Unit Tests** | Validate individual module correctness | `tests/unit/` |
| **Integration Tests** | Validate cross-module interactions and closed-loop control | `tests/integration/` |
| **Examples** | Demonstrate end-to-end scenarios with realistic parameters | `examples/` |

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install cvxpy for optimization module
pip install cvxpy

# Optional: Install kaggle for dataset downloads
pip install kaggle
```

## Running Tests

### All Tests

```bash
# Run all unit and integration tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ -v --cov=firmus_ai_factory --cov-report=html
```

### Unit Tests Only

```bash
# All unit tests
python -m pytest tests/unit/ -v

# Individual module tests
python -m pytest tests/unit/test_power.py -v
python -m pytest tests/unit/test_grid.py -v
python -m pytest tests/unit/test_storage.py -v
python -m pytest tests/unit/test_optimization.py -v
python -m pytest tests/unit/test_economics.py -v
python -m pytest tests/unit/test_control.py -v
python -m pytest tests/unit/test_sensor_emulator.py -v
```

### Integration Tests Only

```bash
# All integration tests
python -m pytest tests/integration/ -v

# Closed-loop control tests
python -m pytest tests/integration/test_closed_loop.py -v

# Model validation tests
python -m pytest tests/integration/test_model_validation.py -v
```

### Examples

```bash
# Run closed-loop demo with emulated sensors
python examples/04_closed_loop_emulated_sensors.py

# Run complete system integration
python examples/03_complete_system_integration.py
```

## Unit Test Coverage

### Power Delivery Network (`test_power.py`)

Tests the complete power delivery chain from grid to GPU:

- **TransformerModel**: Voltage regulation, efficiency, overload protection, impedance calculation
- **DCDCConverter**: Buck conversion, input voltage variation, load regulation
- **VoltageRegulatorModule**: Steady-state operation, transient response, impedance, current sharing
- **PowerDeliveryChain**: End-to-end efficiency validation

### Grid Interface (`test_grid.py`)

Tests grid interconnection and demand response:

- **GridInterface**: Frequency response (droop control), voltage regulation, power factor correction, stability assessment
- **DemandResponseManager**: Economic DR bidding, emergency response, workload deferral, ramp rate compliance, revenue calculation
- **GridIntegration**: Coordinated frequency and DR response, grid service revenue

### Energy Storage (`test_storage.py`)

Tests battery/UPS electrochemical model:

- **LithiumIonBattery**: SOC limits, charge/discharge cycles, power limits, thermal model, degradation, OCV curve, internal resistance, energy accounting
- **BatteryConfigurations**: Tesla Megapack specs, custom configurations
- **UPSFunctionality**: Backup duration, grid failure response

### Optimization (`test_optimization.py`)

Tests Model Predictive Control:

- **ModelPredictiveController**: Cost minimization, thermal constraints, throughput maximization, multi-objective optimization, receding horizon, DR integration, solve time performance
- **MPCEdgeCases**: Infeasible problems, zero workload, tight deadlines

### Economics (`test_economics.py`)

Tests electricity tariff and cost analysis:

- **ElectricityTariff**: TOU initialization, price lookup, 24-hour profiles, energy cost calculation, demand charges, total bill
- **RealTimePricing**: RTP with price signals, volatility impact
- **CostBreakdown**: Period breakdown, load shifting savings, annual projection

### Control & Integration (`test_control.py`)

Tests digital twin system integration:

- **DigitalTwin**: Initialization, state management, scenario simulation, workload variation, grid events, thermal protection, energy accounting
- **MultiSubsystem**: Subsystem coupling, steady-state convergence, transient response
- **Optimization**: Cost optimization, what-if analysis

### Sensor Emulator (`test_sensor_emulator.py`)

Tests sensor emulation utilities:

- **SensorEmulator**: Initialization, synthetic readings, workload effects, measurement noise, sensor streams, DataFrame conversion, grid frequency realism, dataset mode, looping
- **SensorReading**: Dataclass creation, optional fields

## Integration Test Coverage

### Closed-Loop Control (`test_closed_loop.py`)

Tests complete closed-loop control with emulated sensors:

- **FrequencyResponse**: Droop response to frequency events, regulation revenue calculation
- **DemandResponse**: Economic DR event handling, ramp rate compliance
- **MPC**: Cost reduction vs baseline, thermal constraint satisfaction, workload completion
- **ThermalManagement**: Cooling system response, ambient temperature impact
- **EndToEnd**: 24-hour operation, 7-day simulation

### Model Validation (`test_model_validation.py`)

Validates models against vendor specifications and industry benchmarks:

- **GPU Power**: TDP accuracy (±5%), idle power, power-utilization curve, training workload
- **Thermal**: Thermal resistance range, steady-state temperature, heat removal capacity
- **Power Delivery**: Transformer efficiency (97-99.5%), converter efficiency (92-98%)
- **Grid Interface**: Frequency deadband (NERC BAL-003), voltage range (ANSI C84.1)
- **Economics**: Daily cost reasonableness, annual cost benchmark ($6-10M for 10 MW)
- **Cross-Module**: Power balance (PUE 1.03-1.5), energy conservation

## Sensor Emulation

The sensor emulator enables closed-loop testing without physical hardware:

### Synthetic Mode

Generates physics-based sensor readings using mathematical models:

```python
from firmus_ai_factory.utils.sensor_emulator import SensorEmulator

emulator = SensorEmulator(noise_level=0.02)
reading = emulator.get_synthetic_reading(timestamp=0.0, workload_intensity=0.8)
print(f"GPU Power: {reading.gpu_power:.1f} W")
print(f"GPU Temp: {reading.gpu_temperature:.1f} °C")
```

### Dataset Mode

Uses real-world datasets for more realistic emulation:

```python
import pandas as pd

# Load dataset (e.g., BUTTER-E)
df = pd.read_csv("data/raw/butter_e/BUTTER-E Energy.csv")

emulator = SensorEmulator(noise_level=0.02)
emulator.load_dataset(df)
reading = emulator.get_reading(0.0, mode="dataset")
```

### Available Datasets

| Dataset | Source | Description | Access |
|---------|--------|-------------|--------|
| BUTTER-E | DOE/OpenEI | 63,527 DL training energy measurements | Kaggle / OpenEI |
| IEEE Server Energy | IEEE DataPort | Real-world server telemetry | IEEE DataPort |
| MLPerf Power | MLCommons | Standardized ML benchmark power data | mlcommons.org |
| Google Cluster Traces | Google | Large-scale cluster workload traces | GitHub |
| Alibaba GPU Traces | Alibaba | Production GPU cluster traces | GitHub |

## Validation Methodology

Models are validated against three categories of benchmarks:

### Vendor Specifications

Direct comparison with published datasheets:

- NVIDIA H100 TDP: 700W (±5%)
- Immersion cooling thermal resistance: 0.03-0.08 K/W
- Transformer efficiency: 97-99.5%
- DC-DC converter efficiency: 92-98%

### Industry Standards

Compliance with regulatory and industry standards:

- NERC BAL-003: Frequency response deadband (36 mHz)
- ANSI C84.1: Voltage regulation (±5%)
- IEEE 1547: Grid interconnection requirements

### Academic Literature

Cross-validation with published research:

- NREL Chip-to-Grid data center modeling
- Google power-flexible AI infrastructure studies
- Aurora AI factory optimization results

## Troubleshooting

### Common Issues

**ImportError: No module named 'firmus_ai_factory'**

```bash
# Install package in development mode
pip install -e .
```

**cvxpy not found (optimization tests)**

```bash
pip install cvxpy
```

**Tests timing out**

Some integration tests run multi-hour simulations. Use pytest timeout:

```bash
python -m pytest tests/ -v --timeout=120
```

## Contributing

When adding new modules or features:

1. Create unit tests in `tests/unit/test_<module>.py`
2. Add integration tests in `tests/integration/` if cross-module
3. Validate against vendor specs where applicable
4. Update this guide with new test descriptions
5. Ensure all tests pass before committing
