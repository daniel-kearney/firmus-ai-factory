# Mathematical Theory

This document provides the comprehensive mathematical foundations underlying the Firmus AI Factory Digital Twin framework. The models enable accurate simulation of multi-physics phenomena from GPU-level power consumption to facility-wide thermal management.

The framework is designed to support arbitrary NVL72-class rack configurations with different liquid/air cooling partitions, including GB300 NVL72 and Vera Rubin NVL72 racks.

## Table of Contents

1. [GPU Power Modeling](#1-gpu-power-modeling)
2. [Thermal Modeling](#2-thermal-modeling)
   - 2.1-2.6: Immersion Cooling (single/two-phase)
   - 2.7: Air Cooling for Peripheral Components  
   - 2.8: Direct-to-Chip Liquid Cooling
   - 2.9: NVL72-Class Rack-Scale Thermal Analysis
   - 2.10: Vendor Rack-Level P-Q Curves
3. [Power Delivery Network](#3-power-delivery-network)
4. [Numerical Methods](#4-numerical-methods)
5. [Model Validation](#5-model-validation)

---

## 1. GPU Power Modeling

### 1.1 Power Decomposition Model

Instantaneous GPU power consumption is decomposed into three principal components:

$$P_{GPU}(t) = P_{compute}(t) + P_{memory}(t) + P_{transfer}(t)$$

Where:
- **P_compute(t)**: Power consumed by compute units (SM/CU)
- **P_memory(t)**: Power consumed by HBM access operations
- **P_transfer(t)**: Power consumed by data transfer (NVLink, PCIe)

### 1.2 Training Workload Power Model

For deep learning training workloads, we employ a phase-aware power model:

$$P_{train}(t) = P_{base} + \sum_i \alpha_i \cdot f_i(t) + \epsilon(t)$$

Where:
- **P_base**: Sustained baseline power (~65-70% of TDP)
- **α_i**: Phase intensity coefficients
- **f_i(t)**: Phase indicator functions (forward, backward, gradient sync, optimizer)
- **ε(t)**: Stochastic variation term ~ N(0, σ²)

#### Phase Coefficients

| Phase | Coefficient (α) | Duration Fraction |
|-------|-----------------|-------------------|
| Forward Pass | 0.85 | 30% |
| Backward Pass | 0.95 | 45% |
| Gradient Sync | 0.60 | 15% |
| Optimizer Step | 0.75 | 10% |

### 1.3 NVL72-Class Rack Configurations

The framework supports parameterized NVL72-class rack configurations. Two reference implementations:

#### GB300 NVL72 Configuration

| Parameter | Value | Units |
|-----------|-------|-------|
| GPU TDP | 1400 | W |
| GPUs per rack | 72 | - |
| CPUs per rack | 36 | - |
| CPU TDP | 250 | W |
| NVLink Switches | 9 trays | - |
| **Liquid IT Power** | **118** | **kW** |
| **Air IT Power** | **18** | **kW** |
| **Total Rack IT Power** | **136** | **kW** |
| Cooling Type | Hybrid DLC + air | - |

Liquid cooling breakdown:
- GPUs: 72 × 1400 W = 100.8 kW (DLC)
- CPUs: 36 × 250 W = 9.0 kW (DLC)
- Other liquid loads: ~8.2 kW (board-level DLC)
- **Total liquid: 118 kW**

Air cooling breakdown:
- NVSwitch/NIC/peripherals: 18 kW

#### Vera Rubin NVL72 Configuration

| Parameter | Value | Units |
|-----------|-------|-------|
| GPU TDP | (per device spec) | W |
| GPUs per rack | 72 | - |
| CPUs per rack | 36 | - |
| CPU TDP | 250 | W |
| **Liquid IT Power** | **220** | **kW** |
| **Air IT Power** | **~0** | **kW** |
| **Total Rack IT Power** | **220** | **kW** |
| Cooling Type | Full DLC (100% liquid) | - |

Vera Rubin uses 100% direct liquid cooling for all compute/switch components, with minimal air cooling only for PSUs and management switches.

### 1.4 Generic Rack Model

For arbitrary NVL72-class racks, define:

$$Q_{rack} = N_{GPU} P_{GPU,TDP} + N_{CPU} P_{CPU,TDP} + N_{SW} P_{SW,TDP} + Q_{other}$$

Split into liquid vs air:

$$Q_{liq} = N_{GPU} P_{GPU,TDP} f_{GPU,liq} + N_{CPU} P_{CPU,TDP} f_{CPU,liq} + N_{SW} P_{SW,TDP} f_{SW,liq} + Q_{other,liq}$$

$$Q_{air} = Q_{rack} - Q_{liq}$$

Where $f_{\cdot,liq}$ are liquid cooling fractions (0 to 1) for each device type.

### 1.5 Utilization-Based Power Model

For general workloads, power scales with GPU utilization:

$$P(u) = P_{idle} + (P_{TDP} - P_{idle}) \cdot u^\alpha$$

Where:
- **u**: GPU utilization (0 to 1)
- **α**: Non-linearity exponent (typically 1.2)
- **P_idle**: Idle power (~8-15% of TDP)

---
## 2. Thermal Modeling

### 2.1 Conjugate Heat Transfer

The thermal model couples fluid and solid domains through the Navier-Stokes and heat equations.

#### Fluid Domain (Coolant)

$$\rho c_p \left(\frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T\right) = k_f \nabla^2 T$$

#### Solid Domain (GPU/Heatsink)

$$\rho_s c_{ps} \frac{\partial T}{\partial t} = \nabla \cdot (k_s \nabla T) + q_{gen}$$

### 2.2 Thermal Resistance Network

For reduced-order modeling:

$$T_j = T_{ambient} + P_{GPU} \cdot (R_{jc} + R_{ch} + R_{ha})$$

| Resistance | Description | Air Cooling | Liquid Cooling |
|------------|-------------|-------------|----------------|
| R_jc | Junction-to-case | 0.10-0.15 K/W | 0.10-0.15 K/W |
| R_ch | Case-to-heatsink | 0.08-0.12 K/W | 0.02-0.04 K/W |
| R_ha | Heatsink-to-ambient | 0.05-0.10 K/W | 0.01-0.03 K/W |

### 2.3 Convective Heat Transfer

#### Nusselt Correlation

$$Nu = 0.45 \cdot Re^{0.43}$$

$$h = \frac{Nu \cdot k_f}{L_c}$$

### 2.4 Coolant Energy Balance

$$\Delta T_{coolant} = \frac{P_{total}}{\dot{m} \cdot c_p}$$

### 2.5 Two-Phase Boiling Heat Transfer

For nucleate boiling (immersion cooling):

$$h_{boiling} \approx 5000 \cdot \left(\frac{q''}{10000}\right)^{0.7}$$

### 2.6 Dielectric Coolant Properties

| Property | 3M EC-100 | Novec 7100 | Units |
|----------|-----------|------------|-------|
| Density | 1510 | 1510 | kg/m³ |
| Specific Heat | 1100 | 1183 | J/kg·K |
| Thermal Conductivity | 0.063 | 0.069 | W/m·K |
| Boiling Point | 61 | 61 | °C |

### 2.7 Air Cooling for Peripheral Components

For NVL72 racks, peripheral components may be air-cooled (GB300: 18 kW air) or liquid-cooled (Vera Rubin: ~0 kW air).

#### Forced Air Convection

Heat transfer coefficient for forced air over heatsinks:

$$h_{air} = \frac{Nu \cdot k_{air}}{L_c}$$

Where the Nusselt number for turbulent flow:

$$Nu = 0.023 \cdot Re^{0.8} \cdot Pr^{0.4}$$

#### Air Flow Requirements

Volumetric flow rate for target temperature rise:

$$\dot{V} = \frac{Q}{\rho_{air} \cdot c_{p,air} \cdot \Delta T}$$

#### Fan Power Model

Fan power scales with flow rate cubed:

$$P_{fan} = k_{fan} \cdot \dot{V}^3$$

Typically k_fan = 0.5-1.5 for data center fans.

#### NVL72 Air Cooling Specifications (GB300 Example)

| Component | Power (W) | Quantity | Total (kW) |
|-----------|-----------|----------|------------|
| NVSwitch/NIC/peripheral | Various | - | 18 |

For Vera Rubin, air cooling is minimal (~0 kW for compute, small load for PSU/MGMT).

### 2.8 Direct-to-Chip Liquid Cooling (DLC)

Direct-to-chip cooling uses cold plates attached directly to GPU/CPU packages with liquid coolant flowing through microchannels.

#### Cold Plate Thermal Resistance

$$R_{cp} = R_{base} + R_{channel} + R_{convective}$$

Where:
- **R_base**: Conduction through cold plate base (~0.005 K/W)
- **R_channel**: Conduction to channel walls (~0.003 K/W)
- **R_convective**: Convection to coolant (~0.010-0.015 K/W)

#### Microchannel Heat Transfer

For laminar flow in microchannels:

$$Nu = 4.36$$ (constant heat flux)

For turbulent flow (Re > 2300):

$$Nu = 0.023 \cdot Re^{0.8} \cdot Pr^{0.4}$$

#### Junction Temperature Calculation

$$T_j = T_{supply} + Q \cdot (R_{jc} + R_{TIM} + R_{cp}) + \frac{Q}{\dot{m} \cdot c_p}$$

#### Flow Rate Requirements

Minimum flow rate for target return temperature:

$$\dot{m} = \frac{Q}{c_p \cdot (T_{return,max} - T_{supply})}$$

#### CDU (Coolant Distribution Unit) Power

$$P_{CDU} = P_{pump} + P_{heat\_exchanger}$$

$$P_{pump} = \frac{\dot{V} \cdot \Delta P}{\eta_{pump}}$$

Typical CDU efficiency: 95-98%

#### Cold Plate Specifications (Generic)

| Parameter | Value | Units |
|-----------|-------|-------|
| Cold Plate HTC | 8000-12000 | W/m²·K |
| Flow Rate (per GPU) | 2-4 | L/min |
| Pressure Drop | 30-50 | kPa |
| Supply Temperature | 17-45 | °C |
| Max Return Temperature | 45-65 | °C |

### 2.9 NVL72-Class Rack-Scale Thermal Analysis

The NVL72-class rack model supports generic liquid/air hybrid or pure-liquid configurations.

#### Generic Total Cooling Load

$$Q_{total} = Q_{liq} + Q_{air}$$

Where:
- $Q_{liq}$ = total liquid-cooled IT load (GPUs + CPUs + switches + other)
- $Q_{air}$ = total air-cooled IT load (switches/NICs/peripherals)

#### GB300 NVL72 Rack (Hybrid Cooling)

- Liquid: 118 kW (GPUs 100.8 kW + CPUs 9 kW + other 8.2 kW)
- Air: 18 kW (NVSwitch/NIC/peripherals)
- **Total: 136 kW**

#### Vera Rubin NVL72 Rack (Full Liquid Cooling)

- Liquid: 220 kW (GPUs + CPUs + switches + other, 100% DLC)
- Air: ~0 kW (minimal PSU/MGMT only)
- **Total: ~220 kW**

#### Liquid Cooling Loop Sizing

For DLC components:

$$\dot{m}_{DLC} = \frac{Q_{liq}}{c_p \cdot \Delta T}$$

For GB300 with $Q_{liq} = 118\ \text{kW}$ and $\Delta T = 20\ \text{K}$:

$$\dot{m}_{DLC} = \frac{118{,}000}{4180 \cdot 20} \approx 1.41\ \text{kg/s} \approx 84.7\ \text{lpm}$$

For Vera Rubin with $Q_{liq} = 220\ \text{kW}$ and $\Delta T = 20\ \text{K}$:

$$\dot{m}_{DLC} = \frac{220{,}000}{4180 \cdot 20} \approx 2.63\ \text{kg/s} \approx 157.7\ \text{lpm}$$

#### pPUE Calculation

$$pPUE = \frac{P_{IT} + P_{cooling}}{P_{IT}}$$

Where:

$$P_{cooling} = P_{CDU} + P_{fans} + P_{chillers}$$

Target pPUE for DLC: 1.05-1.10

### 2.10 Vendor Rack-Level P-Q Curves

NVIDIA provides rack-level flow vs inlet temperature curves that must be satisfied.

#### GB300 NVL72 Rack-Level P-Q Curve

For a 72-GPU single rack with 2 × 1.5" supply hoses:

$$\dot{V}_{GB300}(T_{in}) = \frac{2101.55}{58.97 - T_{in}^{1.0124}} \quad [\text{lpm}]$$

Valid for $T_{in} \in [25, 45]$ °C.

**Table: GB300 NVL72 Rack Flow/Pressure Requirements**

| Inlet Temp (°C) | Flow (lpm) | Pressure Drop (psi) |
|----------------|------------|---------------------|
| 25 | 65 | 3.6 |
| 30 | 80 | 6.2 |
| 35 | 100 | 10.0 |
| 40 | 130 | 14.8 |
| 45 | 160 | 19.4 |

#### Vera Rubin NVL72 Rack-Level P-Q Curves

For a 72-GPU full rack with 2" supply hoses, Vera Rubin provides two operating modes:

**MaxQ Mode (Low Power)**

For full rack (GPUs + CPUs + NVSwitch):

$$\dot{V}_{VR,MaxQ,full}(T_{in}) = \frac{2570.2}{61.34 - T_{in}^{1.0124}} \quad [\text{lpm}]$$

For GPUs only:

$$\dot{V}_{VR,MaxQ,GPU}(T_{in}) = \frac{2050.5}{61.62 - T_{in}^{1.0124}} \quad [\text{lpm}]$$

**MaxP Mode (High Power)**

For full rack:

$$\dot{V}_{VR,MaxP,full}(T_{in}) = \frac{5461.5}{61.95 - T_{in}^{1.0124}} \quad [\text{lpm}]$$

For GPUs only:

$$\dot{V}_{VR,MaxP,GPU}(T_{in}) = \frac{4355.8}{62.22 - T_{in}^{1.0124}} \quad [\text{lpm}]$$

**Table: Vera Rubin NVL72 Full Rack Flow/Pressure (MaxP Mode)**

| Inlet Temp (°C) | Flow (lpm) | Pressure Drop (psi) |
|----------------|------------|---------------------|
| 25 | 150 | 5.0 |
| 30 | 175 | 7.5 |
| 35 | 210 | 11.0 |
| 40 | 260 | 15.5 |
| 45 | 320 | 20.5 |

**Table: Vera Rubin NVL72 GPU-Only Flow/Pressure (MaxP Mode)**

| Inlet Temp (°C) | Flow (lpm) | Pressure Drop (psi) |
|----------------|------------|---------------------|
| 25 | 120 | 4.0 |
| 30 | 140 | 6.0 |
| 35 | 170 | 9.0 |
| 40 | 210 | 12.5 |
| 45 | 260 | 16.5 |

#### GB300 NVL72 Rack Airflow Requirements

For the 18 kW air-cooled peripherals:

**Table: GB300 Rack Airflow vs Inlet Temperature**

| Inlet Temp (°C) | Required CFM | Max Exhaust (°C) |
|----------------|--------------|------------------|
| 20 | 1400 | 60 |
| 25 | 1600 | 60 |
| 30 | 1900 | 60 |
| 35 | 2200 | 60 |
| 40 | 2560 | 60 |

#### Vera Rubin Airflow Requirements (PSU/MGMT Only)

For the minimal air-cooled power shelf and management switches:

**Table: Vera Rubin Power Shelf + MGMT Switch Airflow (MaxP Mode)**

| Inlet Temp (°C) | CFM (MaxP) | CFM (MaxQ) | CFM/kW |
|----------------|------------|------------|--------|
| 20 | 330 | 260 | 48 |
| 25 | 390 | 310 | 56 |
| 30 | 490 | 390 | 70 |
| 35 | 640 | 510 | 91 |
| 40 | 900 | 700 | 128 |

---
## 3. Power Delivery Network

### 3.1 Multi-Stage Converter Cascade

$$\eta_{PDN} = \prod_{i=1}^{N} \eta_i$$

| Stage | Efficiency |
|-------|------------|
| Transformer | 98-99% |
| UPS | 94-97% |
| PDU | 98-99% |
| PSU | 94-96% |
| VRM | 90-95% |

### 3.2 Target Impedance Design

$$Z_{target} = \frac{\Delta V_{allowed}}{I_{transient}}$$

---

## 4. Numerical Methods

### 4.1 Time Integration

Explicit Euler:

$$T^{n+1} = T^n + \Delta t \cdot f(T^n, t^n)$$

Stability: $\Delta t \leq \frac{(\Delta x)^2}{2\alpha}$

### 4.2 Spatial Discretization

$$\nabla^2 T \approx \frac{T_{i+1} - 2T_i + T_{i-1}}{(\Delta x)^2}$$

---

## 5. Model Validation

### 5.1 GPU Power Model Validation

Validated against NVIDIA SMI, RAPL counters, and external power meters. Typical accuracy: ±5% for mean power.

### 5.2 Thermal Model Validation

Validated against GPU junction sensors and CFD simulations. Typical accuracy: ±2°C for junction temperature.

### 5.3 NVL72-Class Rack Validation Targets

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| Junction Temp | ±3°C | Sensor data |
| Coolant ΔT | ±1°C | Flow/temp sensors |
| pPUE | ±2% | Facility metering |
| Flow Rate | ±5% | Flow meters |
| Rack P-Q Curve | ±5% | Vendor spec compliance |

---

## References

1. NVIDIA GB300/GB200/H200/B200 GPU Architecture Whitepapers
2. NVIDIA Vera Rubin NVL72 Thermal and Mechanical Specification
3. 3M Novec Engineered Fluids Thermal Properties Data
4. Incropera, F.P. et al., "Fundamentals of Heat and Mass Transfer"
5. ASHRAE TC 9.9 Data Center Thermal Guidelines
6. NVIDIA DGX SuperPOD Reference Architecture

---

*Document Version: 3.0*  
*Last Updated: February 2026*  
*Author: Firmus Engineering Team*
