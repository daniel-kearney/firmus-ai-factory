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
firmus_ai_factory.__author__   # "Daniel Kearney"
firmus_ai_factory.__email__    # "daniel@firmus.ai"
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

#### GB300 Preset

```python
from firmus_ai_factory.computational.gpu_model import GB300_SPECS

# GB300 specifications
GB300_SPECS = {
    "name": "GB300",
    "tdp": 1400.0,
    "fp8_tflops": 9000.0,
    "fp16_tflops": 4500.0,
    "hbm_bandwidth": 8000.0,
    "hbm_capacity": 192.0,
}
```

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
