# Mathematical Theory

This document provides the comprehensive mathematical foundations underlying the Firmus AI Factory Digital Twin framework. The models enable accurate simulation of multi-physics phenomena from GPU-level power consumption to facility-wide thermal management.

## Table of Contents

1. [GPU Power Modeling](#1-gpu-power-modeling)
2. [Thermal Modeling](#2-thermal-modeling)
   - 2.1-2.6: Immersion Cooling (single/two-phase)
   - 2.7: Air Cooling for Peripheral Components
   - 2.8: Direct-to-Chip Liquid Cooling
   - 2.9: GB300 NVL72 Rack-Scale Thermal Analysis
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

### 1.3 GB300 Blackwell Ultra Specifications

The GB300 NVL72 rack configuration specifications:

| Parameter | Value | Units |
|-----------|-------|-------|
| GPU TDP | 1400 | W |
| GPUs per rack | 72 | - |
| CPUs per rack | 36 | - |
| CPU TDP | 250 | W |
| NVLink Switches | 18 | - |
| Rack IT Power | ~150 | kW |
| Cooling Type | Direct-to-chip liquid + air hybrid | - |

### 1.4 Utilization-Based Power Model

For general workloads, power scales with GPU utilization:

$$P(u) = P_{idle} + (P_{TDP} - P_{idle}) \cdot u^\alpha$$

Where:
- **u**: GPU utilization (0 to 1)
- **α**: Non-linearity exponent (typically 1.2)
- **P_idle**: Idle power (~10-15% of TDP)

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

For GB300 NVL72 racks, peripheral components (NVLink switches, CPUs, NICs) are air-cooled while GPUs use direct-to-chip liquid cooling.

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

#### NVL72 Air Cooling Specifications

| Component | Power (W) | Quantity | Total (kW) |
|-----------|-----------|----------|------------|
| NVLink Switches | 2000-3200 | 18 | 36-58 |
| CPUs | 250 | 36 | 9 |
| NICs | 25 | 36 | 0.9 |
| Memory/Other | - | - | ~5 |

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

#### Cold Plate Specifications for GB300

| Parameter | Value | Units |
|-----------|-------|-------|
| Cold Plate HTC | 8000-12000 | W/m²·K |
| Flow Rate (per GPU) | 2-4 | L/min |
| Pressure Drop | 30-50 | kPa |
| Supply Temperature | 25-35 | °C |
| Max Return Temperature | 45-50 | °C |

### 2.9 GB300 NVL72 Rack-Scale Thermal Analysis

The NVL72 rack employs a hybrid cooling architecture:
- **GPUs**: Direct-to-chip liquid cooling (1400W × 72 = 100.8 kW)
- **CPUs**: Direct-to-chip liquid cooling (250W × 36 = 9 kW)
- **Switches/NICs**: Forced air cooling (~40 kW)

#### Total Cooling Load

$$Q_{total} = Q_{GPU} + Q_{CPU} + Q_{switch} + Q_{peripheral}$$

$$Q_{total} \approx 100.8 + 9 + 40 + 5 \approx 155 \text{ kW}$$

#### Liquid Cooling Loop Sizing

For DLC components (GPUs + CPUs):

$$\dot{m}_{DLC} = \frac{Q_{DLC}}{c_p \cdot \Delta T} = \frac{110,000}{4180 \cdot 20} \approx 1.3 \text{ kg/s}$$

#### pPUE Calculation

$$pPUE = \frac{P_{IT} + P_{cooling}}{P_{IT}}$$

Where:
$$P_{cooling} = P_{CDU} + P_{fans} + P_{chillers}$$

Target pPUE for DLC: 1.05-1.10

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

Validated against NVIDIA SMI, RAPL counters, and external power meters.
Typical accuracy: ±5% for mean power.

### 5.2 Thermal Model Validation

Validated against GPU junction sensors and CFD simulations.
Typical accuracy: ±2°C for junction temperature.

### 5.3 GB300 NVL72 Validation Targets

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| Junction Temp | ±3°C | Sensor data |
| Coolant ΔT | ±1°C | Flow/temp sensors |
| pPUE | ±2% | Facility metering |
| Flow Rate | ±5% | Flow meters |

---

## References

1. NVIDIA GB300/H200/B200 GPU Architecture Whitepapers
2. 3M Novec Engineered Fluids Thermal Properties Data
3. Incropera, F.P. et al., "Fundamentals of Heat and Mass Transfer"
4. ASHRAE TC 9.9 Data Center Thermal Guidelines
5. NVIDIA DGX SuperPOD Reference Architecture

---

*Document Version: 2.0*
*Last Updated: February 2026*
*Author: Firmus Engineering Team*
