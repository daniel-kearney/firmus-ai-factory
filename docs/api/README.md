# API Reference

Complete API documentation for the Firmus AI Factory Digital Twin framework.

## Table of Contents

1. [Package Overview](#package-overview)
2. [Computational Module](#computational-module)
3. [Thermal Module](#thermal-module)
4. [Top-Level Exports](#top-level-exports)

---

## Package Overview

```python
import firmus_ai_factory

# Version
firmus_ai_factory.__version__  # "0.1.0"
firmus_ai_factory.__author__   # "Daniel Kearney"
firmus_ai_factory.__email__    # "daniel@firmus.ai"
```

### Top-Level Exports

The following classes are available directly from the top-level package:

```python
from firmus_ai_factory import (
    AIFactorySystem,           # Core system integration
    GPUModel,                  # GPU power modeling
    ImmersionCoolingSystem,    # Thermal management
    PowerDeliveryNetwork,      # Power electronics
    MultiObjectiveOptimizer,   # Optimization
)
```

---

## Computational Module

`firmus_ai_factory.computational.gpu_model`

### Enums

#### `WorkloadType(Enum)`

Types of GPU workloads.

| Value | Description |
|-------|-------------|
| `TRAINING` | Training workload (`"training"`) |
| `INFERENCE` | Inference workload (`"inference"`) |
| `IDLE` | Idle state (`"idle"`) |

### Data Classes

#### `GPUSpecifications`

Hardware specifications for GPU accelerators.

```python
@dataclass
class GPUSpecifications:
    name: str                      # GPU model name
    tdp_watts: float               # Thermal Design Power (W)
    peak_flops_fp16: float         # Peak FP16 performance (TFLOPS)
    peak_flops_fp32: float         # Peak FP32 performance (TFLOPS)
    hbm_bandwidth_tb_s: float      # HBM bandwidth (TB/s)
    hbm_capacity_gb: float         # HBM capacity (GB)
    nvlink_bandwidth_gb_s: float   # NVLink bandwidth (GB/s)
    idle_power_fraction: float     # Fraction of TDP at idle (default: 0.15)
    base_power_fraction: float     # Base power during compute (default: 0.65)
```

**Pre-defined GPU Specifications:**

| Constant | GPU | TDP (W) | FP16 (TFLOPS) | HBM (TB/s) | HBM (GB) |
|----------|-----|---------|----------------|------------|----------|
| `H100_SXM_SPECS` | NVIDIA H100 SXM | 700 | 1979 | 3.35 | 80 |
| `H200_SPECS` | NVIDIA H200 | 700 | 1979 | 4.8 | 141 |
| `B200_SPECS` | NVIDIA B200 | 1000 | 4500 | 8.0 | 192 |

#### `PowerProfile`

Time-series power consumption profile.

```python
@dataclass
class PowerProfile:
    time: np.ndarray            # Time points in seconds
    total_power: np.ndarray     # Total power (W)
    compute_power: np.ndarray   # Compute component (W)
    memory_power: np.ndarray    # Memory component (W)
    transfer_power: np.ndarray  # Data transfer component (W)
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `mean_power` | `float` | Mean total power (W) |
| `peak_power` | `float` | Peak total power (W) |
| `energy_kwh` | `float` | Total energy consumed (kWh) |

### Classes

#### `GPUModel`

Mathematical model for GPU power and performance.

```python
gpu = GPUModel(specs: GPUSpecifications)
```

**Constructor Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `specs` | `GPUSpecifications` | Hardware specs for the target GPU |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `specs` | `GPUSpecifications` | GPU hardware specifications |
| `phase_coefficients` | `Dict[str, float]` | Training phase intensity coefficients |
| `noise_std` | `float` | Standard deviation of stochastic power variation |

**Methods:**

##### `compute_flops_per_token(num_params, sequence_length=2048) -> float`

Calculate FLOPS per token for transformer models. Uses the 6N approximation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_params` | `float` | required | Number of model parameters |
| `sequence_length` | `int` | `2048` | Sequence length (reserved) |

**Returns:** `float` - FLOPS per token

##### `compute_mfu(tokens_per_second, num_params, num_gpus=1) -> float`

Calculate Model FLOPS Utilization (MFU).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tokens_per_second` | `float` | required | Observed throughput |
| `num_params` | `float` | required | Number of model parameters |
| `num_gpus` | `int` | `1` | Number of GPUs |

**Returns:** `float` - MFU ratio (0 to 1)

##### `power_from_utilization(utilization) -> float`

Calculate instantaneous power from GPU utilization.

| Parameter | Type | Description |
|-----------|------|-------------|
| `utilization` | `float` | GPU utilization (0.0 to 1.0) |

**Returns:** `float` - Power in Watts

##### `simulate_training_workload(model_params, batch_size, duration, dt=0.001) -> PowerProfile`

Simulate a time-resolved power profile for a training workload.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_params` | `float` | required | Number of model parameters |
| `batch_size` | `int` | required | Training batch size |
| `duration` | `float` | required | Simulation duration (seconds) |
| `dt` | `float` | `0.001` | Time step (seconds) |

**Returns:** `PowerProfile` - Time-series power data

**Example:**

```python
from firmus_ai_factory.computational.gpu_model import GPUModel, H100_SXM_SPECS

gpu = GPUModel(H100_SXM_SPECS)
profile = gpu.simulate_training_workload(
    model_params=70e9,
    batch_size=32,
    duration=10.0
)
print(f"Mean power: {profile.mean_power:.1f} W")
print(f"Peak power: {profile.peak_power:.1f} W")
print(f"Energy: {profile.energy_kwh:.4f} kWh")
```

---

## Thermal Module

`firmus_ai_factory.thermal.immersion_cooling`

### Enums

#### `CoolantType(Enum)`

Dielectric coolant types.

| Value | Description |
|-------|-------------|
| `EC100` | 3M Novec/EC-100 (`"ec100"`) |
| `FC72` | 3M Fluorinert (`"fc72"`) |
| `SINGLEPHASE` | Generic single-phase (`"single_phase"`) |
| `TWOPHASE` | Generic two-phase (`"two_phase"`) |

### Data Classes

#### `CoolantProperties`

Thermophysical properties of dielectric coolants.

```python
@dataclass
class CoolantProperties:
    name: str                    # Coolant name
    density: float               # Density (kg/m^3)
    specific_heat: float         # Specific heat (J/(kg*K))
    thermal_conductivity: float  # Thermal conductivity (W/(m*K))
    viscosity: float             # Dynamic viscosity (Pa*s)
    prandtl: float               # Prandtl number
    boiling_point: float         # Boiling point (Celsius)
    latent_heat: float           # Latent heat of vaporization (J/kg)
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `thermal_diffusivity` | `float` | k / (rho * c_p) in m^2/s |

**Pre-defined Coolants:**

| Constant | Coolant | Density | c_p | k | Boiling Pt |
|----------|---------|---------|-----|---|------------|
| `EC100_PROPERTIES` | 3M EC-100 | 1510 kg/m3 | 1100 J/kgK | 0.063 W/mK | 61 C |
| `NOVEC_7100_PROPERTIES` | 3M Novec 7100 | 1510 kg/m3 | 1183 J/kgK | 0.069 W/mK | 61 C |

#### `ThermalResult`

Results from thermal analysis.

```python
@dataclass
class ThermalResult:
    T_junction_max: float       # Max junction temperature (C)
    T_junction_mean: float      # Mean junction temperature (C)
    T_coolant_out: float        # Coolant outlet temperature (C)
    P_cooling: float            # Cooling system power (W)
    thermal_resistance: float   # Total thermal resistance (K/W)
    heat_transfer_coeff: float  # Average HTC (W/m^2/K)
    pPUE: float                 # Partial Power Usage Effectiveness
```

### Classes

#### `ThermalResistanceNetwork`

Thermal resistance network model from junction to ambient.

```python
network = ThermalResistanceNetwork(R_jc=0.15, R_ch=0.08, R_ha=0.05)
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `R_jc` | `float` | `0.15` | Junction-to-case resistance (K/W) |
| `R_ch` | `float` | `0.08` | Case-to-heatsink resistance (K/W) |
| `R_ha` | `float` | `0.05` | Heatsink-to-ambient resistance (K/W) |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `total_resistance` | `float` | Sum of R_jc + R_ch + R_ha (K/W) |

**Methods:**

##### `junction_temperature(power, T_ambient) -> float`

Calculate junction temperature given power dissipation and ambient temperature.

| Parameter | Type | Description |
|-----------|------|-------------|
| `power` | `float` | GPU power dissipation (W) |
| `T_ambient` | `float` | Ambient/coolant temperature (C) |

**Returns:** `float` - Junction temperature (C)

---

#### `ImmersionCoolingSystem`

Single-phase immersion cooling system model.

```python
cooling = ImmersionCoolingSystem(
    coolant=EC100_PROPERTIES,
    flow_rate=2.5,
    inlet_temp=35.0,
    tank_volume=500.0
)
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `coolant` | `CoolantProperties` | `EC100_PROPERTIES` | Coolant thermophysical properties |
| `flow_rate` | `float` | `2.5` | Flow rate per GPU (L/min) |
| `inlet_temp` | `float` | `35.0` | Coolant inlet temperature (C) |
| `tank_volume` | `float` | `500.0` | Tank volume (Liters) |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `thermal_network` | `ThermalResistanceNetwork` | GPU thermal resistance network |
| `pump_efficiency` | `float` | Pump efficiency (default: 0.65) |
| `pressure_drop_pa` | `float` | System pressure drop (default: 15000 Pa) |

**Methods:**

##### `compute_heat_transfer_coefficient(velocity, characteristic_length=0.05) -> float`

Calculate convective HTC using Nusselt correlations.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `velocity` | `float` | required | Flow velocity (m/s) |
| `characteristic_length` | `float` | `0.05` | Characteristic length (m) |

**Returns:** `float` - Heat transfer coefficient (W/m^2/K)

##### `compute_coolant_temperature_rise(power, flow_rate) -> float`

Calculate coolant temperature rise from energy balance.

| Parameter | Type | Description |
|-----------|------|-------------|
| `power` | `float` | Total heat load (W) |
| `flow_rate` | `float` | Volumetric flow rate (m^3/s) |

**Returns:** `float` - Temperature rise (K)

##### `compute_pump_power(flow_rate) -> float`

Calculate pump electrical power requirement.

| Parameter | Type | Description |
|-----------|------|-------------|
| `flow_rate` | `float` | Volumetric flow rate (m^3/s) |

**Returns:** `float` - Pump power (W)

##### `analyze(power_profile, num_gpus=8) -> ThermalResult`

Perform full thermal analysis for given GPU power profile.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `power_profile` | `np.ndarray` or `float` | required | GPU power values (W) |
| `num_gpus` | `int` | `8` | Number of GPUs in system |

**Returns:** `ThermalResult` - Complete thermal analysis results

**Example:**

```python
from firmus_ai_factory.thermal.immersion_cooling import (
    ImmersionCoolingSystem, EC100_PROPERTIES
)
import numpy as np

cooling = ImmersionCoolingSystem(
    coolant=EC100_PROPERTIES,
    flow_rate=2.5,
    inlet_temp=35.0
)
power = np.full(1000, 650.0)  # 650W per GPU
result = cooling.analyze(power, num_gpus=8)

print(f"Max junction temp: {result.T_junction_max:.1f} C")
print(f"Coolant outlet: {result.T_coolant_out:.1f} C")
print(f"pPUE: {result.pPUE:.3f}")
```

---

#### `TwoPhaseImmersionSystem(ImmersionCoolingSystem)`

Two-phase immersion cooling with boiling heat transfer. Inherits from `ImmersionCoolingSystem`.

**Overridden Methods:**

##### `compute_heat_transfer_coefficient(heat_flux, T_surface) -> float`

Calculate boiling HTC using Rohsenow correlation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `heat_flux` | `float` | Surface heat flux (W/m^2) |
| `T_surface` | `float` | Surface temperature (C) |

**Returns:** `float` - Boiling heat transfer coefficient (W/m^2/K), capped at 15000

---

*Document Version: 1.0*
*Last Updated: February 2026*
*Author: Firmus Engineering Team*
