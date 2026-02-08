# API Reference

Complete API documentation for the Firmus AI Factory Digital Twin framework.

## Table of Contents

1. [Package Overview](#package-overview)
2. [Computational Module](#computational-module)
3. [Thermal Module](#thermal-module)
   - ImmersionCoolingSystem
   - AirCoolingSystem / NVL72PeripheralAirCooling
   - DirectToChipCooling / ColdPlateSpec
4. [Top-Level Exports](#top-level-exports)

---

## Package Overview

```python
import firmus_ai_factory

# Version
firmus_ai_factory.__version__  # "0.2.0"
firmus_ai_factory.__author__   # "Dr. Daniel Kearney"
firmus_ai_factory.__email__    # "daniel.kearney@firmus.co"
```

### Top-Level Exports

The following classes are available directly from the top-level package:

```python
from firmus_ai_factory import (
    AIFactorySystem,
    GPUModel,
    ImmersionCoolingSystem,
    AirCoolingSystem,
    NVL72PeripheralAirCooling,
    DirectToChipCooling,
    ColdPlateSpec,
    PowerDeliveryNetwork,
    MultiObjectiveOptimizer,
)
```

---

## Computational Module

### GPUModel

```python
from firmus_ai_factory.computational import GPUModel
```

Models GPU power consumption and performance characteristics.

#### Constructor

```python
GPUModel(
    name: str = "H100",
    tdp: float = 700.0,
    base_clock: float = 1830.0,
    boost_clock: float = 2520.0,
    memory_bandwidth: float = 3350.0,
    memory_capacity: float = 80.0,
    fp16_tflops: float = 1979.0,
    fp32_tflops: float = 989.0,
    fp64_tflops: float = 67.0
)
```

#### GPU Specification Presets

The computational module includes pre-defined specifications for all supported GPU architectures:

```python
from firmus_ai_factory.computational.gpu_model import (
    H100_SXM_SPECS,
    H200_SPECS,
    B200_SPECS,
    GB300_SPECS,
    VERA_RUBIN_MAX_P_SPECS,
    VERA_RUBIN_MAX_Q_SPECS,
)
```

##### Hopper Architecture (Singapore + Immersion Cooling)

**H100 SXM5:**
```python
H100_SXM_SPECS = GPUSpecifications(
    name="NVIDIA H100 SXM",
    tdp_watts=700.0,
    peak_flops_fp16=1979.0,
    peak_flops_fp32=989.5,
    hbm_bandwidth_tb_s=3.35,
    hbm_capacity_gb=80.0,
    nvlink_bandwidth_gb_s=900.0,
    max_junction_temp_c=83.0,
)
```

**H200:**
```python
H200_SPECS = GPUSpecifications(
    name="NVIDIA H200",
    tdp_watts=700.0,
    peak_flops_fp16=1979.0,
    peak_flops_fp32=989.5,
    hbm_bandwidth_tb_s=4.8,      # HBM3e upgrade
    hbm_capacity_gb=141.0,       # HBM3e upgrade
    nvlink_bandwidth_gb_s=900.0,
    max_junction_temp_c=83.0,
)
```

##### Blackwell Architecture (Australia + Benmax HCU2500)

**B200:**
```python
B200_SPECS = GPUSpecifications(
    name="NVIDIA B200",
    tdp_watts=1000.0,
    peak_flops_fp16=4500.0,
    peak_flops_fp32=2250.0,
    hbm_bandwidth_tb_s=8.0,
    hbm_capacity_gb=192.0,
    nvlink_bandwidth_gb_s=1800.0,
    max_junction_temp_c=100.0,
    energy_storage_j=65.0,       # On-board capacitance for power smoothing
)
```

**GB300 Blackwell Ultra:**
```python
GB300_SPECS = GPUSpecifications(
    name="NVIDIA GB300 Blackwell Ultra",
    tdp_watts=1400.0,
    peak_flops_fp16=5000.0,
    peak_flops_fp32=2500.0,
    hbm_bandwidth_tb_s=8.0,
    hbm_capacity_gb=288.0,
    nvlink_bandwidth_gb_s=1800.0,
    max_junction_temp_c=100.0,
    energy_storage_j=65.0,
)
```

##### Vera Rubin Architecture (Australia + Benmax HCU2500)

**Vera Rubin Max P (Maximum Performance):**
```python
VERA_RUBIN_MAX_P_SPECS = GPUSpecifications(
    name="NVIDIA Vera Rubin (Max P)",
    tdp_watts=2300.0,            # 2.3 kW TGP per GPU
    peak_flops_fp16=6500.0,
    peak_flops_fp32=3250.0,
    hbm_bandwidth_tb_s=12.0,     # HBM4
    hbm_capacity_gb=384.0,
    nvlink_bandwidth_gb_s=3600.0,  # NVLink6
    max_junction_temp_c=100.0,
    energy_storage_j=400.0,      # Enhanced power smoothing
)
```

**Vera Rubin Max Q (Optimized Efficiency):**
```python
VERA_RUBIN_MAX_Q_SPECS = GPUSpecifications(
    name="NVIDIA Vera Rubin (Max Q)",
    tdp_watts=1800.0,            # 1.8 kW TGP per GPU
    peak_flops_fp16=5500.0,
    peak_flops_fp32=2750.0,
    hbm_bandwidth_tb_s=12.0,
    hbm_capacity_gb=384.0,
    nvlink_bandwidth_gb_s=3600.0,
    max_junction_temp_c=100.0,
    energy_storage_j=400.0,
)
```

#### Platform-Region-Cooling Mapping

| GPU Architecture | Deployment Region | Cooling Technology |
|---|---|---|
| H100 SXM / H200 | Singapore | Single-phase immersion cooling |
| B200 / GB300 | Australia (Batam BT1-2) | Benmax HCU2500 Hypercube (liquid-to-liquid CDU) |
| Vera Rubin | Australia (Batam BT1-2) | Benmax HCU2500 Hypercube (liquid-to-liquid CDU) |

---

## Thermal Module

### ImmersionCoolingSystem

```python
from firmus_ai_factory.thermal import ImmersionCoolingSystem
```

Models single-phase and two-phase immersion cooling systems.

#### Constructor

```python
ImmersionCoolingSystem(
    coolant_type: str = "EC-100",
    tank_volume: float = 1000.0,
    flow_rate: float = 50.0,
    inlet_temp: float = 35.0
)
```

#### Methods

- `calculate_htc(re: float) -> float`: Calculate heat transfer coefficient
- `calculate_junction_temp(power: float, ambient: float) -> float`: Calculate GPU junction temperature
- `calculate_ppue(it_power: float, cooling_power: float) -> float`: Calculate partial PUE

---

### AirCoolingSystem

```python
from firmus_ai_factory.thermal import AirCoolingSystem
```

Models forced-air cooling for data center components.

#### Constructor

```python
AirCoolingSystem(
    flow_rate_cfm: float = 500.0,
    inlet_temp: float = 25.0,
    ambient_pressure: float = 101325.0
)
```

#### Methods

- `calculate_htc(velocity: float, char_length: float) -> float`: Calculate convective HTC
- `calculate_outlet_temp(heat_load: float) -> float`: Calculate air outlet temperature
- `calculate_fan_power(pressure_drop: float) -> float`: Calculate fan power consumption
- `calculate_required_flow(heat_load: float, delta_t: float) -> float`: Calculate required CFM

---

### NVL72PeripheralAirCooling

```python
from firmus_ai_factory.thermal import NVL72PeripheralAirCooling
```

Specialized air cooling model for GB300 NVL72 rack peripherals.

#### Constructor

```python
NVL72PeripheralAirCooling(
    num_switches: int = 18,
    switch_power: float = 3200.0,
    num_nics: int = 36,
    nic_power: float = 25.0,
    ambient_temp: float = 25.0
)
```

#### Methods

- `total_heat_load() -> float`: Calculate total peripheral heat load (W)
- `required_airflow() -> float`: Calculate required airflow (CFM)
- `calculate_fan_power() -> float`: Calculate total fan power (W)
- `analyze() -> dict`: Return comprehensive cooling analysis

#### Example

```python
from firmus_ai_factory.thermal import NVL72PeripheralAirCooling

cooling = NVL72PeripheralAirCooling(
    num_switches=18,
    switch_power=3200.0,  # W per NVSwitch
    ambient_temp=25.0
)

result = cooling.analyze()
print(f"Total heat load: {result['total_heat_load']/1000:.1f} kW")
print(f"Required airflow: {result['airflow_cfm']:.0f} CFM")
print(f"Fan power: {result['fan_power']:.0f} W")
```

---

### DirectToChipCooling

```python
from firmus_ai_factory.thermal import DirectToChipCooling
```

Models direct-to-chip liquid cooling (DLC) systems with cold plates.

#### Constructor

```python
DirectToChipCooling(
    supply_temp: float = 25.0,
    max_return_temp: float = 45.0,
    cdu_capacity_kw: float = 250.0,
    coolant_type: str = "water-glycol"
)
```

#### Methods

- `calculate_junction_temp(power: float, cold_plate_htc: float) -> float`: Calculate T_junction
- `calculate_flow_rate(heat_load: float) -> float`: Calculate required flow rate (L/min)
- `calculate_cdu_power(heat_load: float) -> float`: Calculate CDU power consumption
- `calculate_ppue(it_power: float) -> float`: Calculate partial PUE
- `analyze_nvl72_rack(...) -> NVL72ThermalResult`: Full NVL72 rack analysis

#### analyze_nvl72_rack Method

```python
result = dlc.analyze_nvl72_rack(
    gpu_power: float = 1400.0,      # W per GPU
    num_gpus: int = 72,
    num_cpus: int = 36,
    cpu_power: float = 250.0,       # W per CPU
    switch_power: float = 3200.0    # W total for switches
)
```

Returns `NVL72ThermalResult` dataclass with:
- `T_junction_max`: Maximum GPU junction temperature (°C)
- `T_junction_mean`: Mean GPU junction temperature (°C)
- `T_coolant_supply`: Coolant supply temperature (°C)
- `T_coolant_return`: Coolant return temperature (°C)
- `flow_rate_lpm`: Required flow rate (L/min)
- `P_pump`: Pump power (W)
- `P_cdu`: CDU total power (W)
- `P_cooling_total`: Total cooling power (W)
- `pPUE`: Partial power usage effectiveness

#### Example

```python
from firmus_ai_factory.thermal import DirectToChipCooling

dlc = DirectToChipCooling(
    supply_temp=25.0,
    max_return_temp=45.0,
    cdu_capacity_kw=250.0
)

result = dlc.analyze_nvl72_rack(
    gpu_power=1400.0,   # GB300 at TDP
    num_gpus=72,
    num_cpus=36,
    cpu_power=250.0,
    switch_power=3200.0
)

print(f"GB300 NVL72 DLC Analysis:")
print(f"  T_junction max: {result.T_junction_max:.1f} C")
print(f"  Coolant return: {result.T_coolant_return:.1f} C")
print(f"  Flow rate: {result.flow_rate_lpm:.1f} L/min")
print(f"  pPUE: {result.pPUE:.3f}")
```

---

### ColdPlateSpec

```python
from firmus_ai_factory.thermal import ColdPlateSpec
```

Dataclass defining cold plate thermal specifications.

#### Attributes

```python
@dataclass
class ColdPlateSpec:
    htc: float = 10000.0           # Heat transfer coefficient (W/m²·K)
    contact_area: float = 0.004    # Contact area (m²)
    r_jc: float = 0.10             # Junction-to-case resistance (K/W)
    r_tim: float = 0.02            # TIM thermal resistance (K/W)
    flow_rate_lpm: float = 3.0     # Design flow rate (L/min)
    pressure_drop_kpa: float = 40.0  # Pressure drop (kPa)
```

---

## Module Locations

| Class | Import Path |
|-------|-------------|
| GPUModel | `firmus_ai_factory.computational` |
| ImmersionCoolingSystem | `firmus_ai_factory.thermal` |
| AirCoolingSystem | `firmus_ai_factory.thermal` |
| NVL72PeripheralAirCooling | `firmus_ai_factory.thermal` |
| DirectToChipCooling | `firmus_ai_factory.thermal` |
| ColdPlateSpec | `firmus_ai_factory.thermal` |
| PowerDeliveryNetwork | `firmus_ai_factory.power` |
| MultiObjectiveOptimizer | `firmus_ai_factory.optimization` |

---

*Document Version: 2.0*
*Last Updated: February 2026*
