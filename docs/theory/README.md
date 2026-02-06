# Mathematical Theory

This document provides the comprehensive mathematical foundations underlying the Firmus AI Factory Digital Twin framework. The models enable accurate simulation of multi-physics phenomena from GPU-level power consumption to facility-wide thermal management.

## Table of Contents

1. [GPU Power Modeling](#1-gpu-power-modeling)
2. [Thermal Modeling](#2-thermal-modeling)
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

$$P_{train}(t) = P_{base} + \sum_{i} \alpha_i \cdot f_i(t) + \epsilon(t)$$

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

### 1.3 Utilization-Based Power Model

For general workloads, power scales with GPU utilization:

$$P(u) = P_{idle} + (P_{TDP} - P_{idle}) \cdot u^\alpha$$

Where:
- **u**: GPU utilization (0 to 1)
- **α**: Non-linearity exponent (typically 1.2 for slight super-linear scaling)
- **P_idle**: Idle power (~10-15% of TDP)

### 1.4 FLOPS and Model Efficiency

#### FLOPS per Token (Transformer Models)

For transformer architectures, the computational cost per token is:

$$FLOP_{token} \approx 6 \cdot N_{params}$$

This accounts for:
- 2N for forward pass matrix multiplications
- 4N for backward pass (gradients w.r.t. inputs and weights)

#### Model FLOPS Utilization (MFU)

$$MFU = \frac{\text{Observed Throughput} \times FLOP_{token}}{\text{Peak FLOPS}}$$

Typical MFU values:
- Optimal: 50-60%
- Good: 40-50%
- Suboptimal: <40%

### 1.5 Roofline Model Analysis

Iteration time is bounded by either compute or memory:

$$T_{iter} = \max(T_{compute}, T_{memory}) \cdot (1 + \eta_{overhead})$$

Where:

$$T_{compute} = \frac{6 \cdot N_{params} \cdot B}{\text{Peak FLOPS} \cdot MFU}$$

$$T_{memory} = \frac{2 \cdot N_{params} \cdot 4}{\text{HBM Bandwidth}}$$

---

## 2. Thermal Modeling

### 2.1 Conjugate Heat Transfer

The thermal model couples fluid and solid domains through the Navier-Stokes and heat equations.

#### Fluid Domain (Coolant)

$$\rho c_p \left(\frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T\right) = k_f \nabla^2 T$$

Where:
- **ρ**: Fluid density (kg/m³)
- **c_p**: Specific heat capacity (J/kg·K)
- **u**: Velocity field (m/s)
- **k_f**: Thermal conductivity (W/m·K)

#### Solid Domain (GPU/Heatsink)

$$\rho_s c_{ps} \frac{\partial T}{\partial t} = \nabla \cdot (k_s \nabla T) + q_{gen}$$

Where:
- **q_gen**: Volumetric heat generation (W/m³)
- **k_s**: Solid thermal conductivity (W/m·K)

#### Interface Conditions

At the fluid-solid interface Γ:

$$T_f|_\Gamma = T_s|_\Gamma \quad \text{(temperature continuity)}$$

$$-k_f \nabla T_f \cdot \mathbf{n}|_\Gamma = -k_s \nabla T_s \cdot \mathbf{n}|_\Gamma \quad \text{(heat flux continuity)}$$

### 2.2 Thermal Resistance Network

For reduced-order modeling, we employ a thermal resistance network:

$$T_j = T_{ambient} + P_{GPU} \cdot (R_{jc} + R_{ch} + R_{ha})$$

| Resistance | Description | Typical Value |
|------------|-------------|---------------|
| R_jc | Junction-to-case | 0.10-0.15 K/W |
| R_ch | Case-to-heatsink (TIM) | 0.05-0.10 K/W |
| R_ha | Heatsink-to-ambient | 0.03-0.08 K/W |

**Note**: For immersion cooling, R_ch is significantly reduced (0.03-0.05 K/W) due to direct liquid contact eliminating the TIM interface.

### 2.3 Convective Heat Transfer

#### Forced Convection (Single-Phase Immersion)

Nusselt number correlation for flow over GPU heatsinks:

$$Nu = 0.45 \cdot Re^{0.43}$$

Heat transfer coefficient:

$$h = \frac{Nu \cdot k_f}{L_c}$$

Where:
- **L_c**: Characteristic length (m)
- **Re**: Reynolds number = ρuL/μ

#### Reynolds Number

$$Re = \frac{\rho u L_c}{\mu}$$

### 2.4 Coolant Energy Balance

For the coolant temperature rise through the system:

$$Q = \dot{m} \cdot c_p \cdot \Delta T_{coolant}$$

Solving for temperature rise:

$$\Delta T_{coolant} = \frac{P_{total}}{\dot{m} \cdot c_p}$$

Where:
- **ṁ**: Mass flow rate (kg/s)
- **P_total**: Total heat dissipation (W)

### 2.5 Two-Phase Boiling Heat Transfer

For two-phase immersion cooling with nucleate boiling, we use the Rohsenow correlation:

$$h_{boiling} = C_{sf} \left(\frac{c_p \Delta T_{sat}}{h_{fg} \cdot Pr^n}\right)^3$$

Simplified correlation for dielectric fluids:

$$h_{boiling} \approx 5000 \cdot \left(\frac{q''}{10000}\right)^{0.7}$$

Where:
- **q''**: Heat flux (W/m²)
- **C_sf**: Surface-fluid interaction coefficient (~0.013)
- **h_fg**: Latent heat of vaporization (J/kg)

### 2.6 Dielectric Coolant Properties

| Property | 3M EC-100 | Novec 7100 | Units |
|----------|-----------|------------|-------|
| Density (ρ) | 1510 | 1510 | kg/m³ |
| Specific Heat (c_p) | 1100 | 1183 | J/kg·K |
| Thermal Conductivity (k) | 0.063 | 0.069 | W/m·K |
| Viscosity (μ) | 0.00077 | 0.00058 | Pa·s |
| Prandtl Number | 13.4 | 9.9 | - |
| Boiling Point | 61 | 61 | °C |
| Latent Heat | 88,000 | 112,000 | J/kg |

#### Thermal Diffusivity

$$\alpha_{thermal} = \frac{k}{\rho c_p}$$

---

## 3. Power Delivery Network

### 3.1 Multi-Stage Converter Cascade

The power delivery network consists of cascaded converters from grid to GPU:

$$G_{system}(s) = \prod_{i=1}^{N} G_i(s)$$

Typical stages:
1. **Medium Voltage Transformer**: 13.8kV → 480V
2. **UPS System**: 480V AC → 480V AC (conditioned)
3. **PDU**: 480V → 208V
4. **PSU**: 208V AC → 12V DC
5. **VRM**: 12V → GPU core voltages

### 3.2 Target Impedance Design

For voltage regulation under transient loads:

$$Z_{target} = \frac{\Delta V_{allowed}}{I_{transient}}$$

Where:
- **ΔV_allowed**: Maximum voltage droop (typically 3-5% of nominal)
- **I_transient**: Maximum current step

### 3.3 Efficiency Modeling

Overall PDN efficiency:

$$\eta_{PDN} = \prod_{i=1}^{N} \eta_i$$

Typical stage efficiencies:
| Stage | Efficiency |
|-------|------------|
| Transformer | 98-99% |
| UPS | 94-97% |
| PDU | 98-99% |
| PSU | 94-96% |
| VRM | 90-95% |

---

## 4. Numerical Methods

### 4.1 Time Integration

For transient simulations, we employ explicit Euler integration:

$$T^{n+1} = T^n + \Delta t \cdot f(T^n, t^n)$$

Stability criterion (CFL condition):

$$\Delta t \leq \frac{(\Delta x)^2}{2\alpha}$$

### 4.2 Spatial Discretization

Finite difference approximation for the Laplacian:

$$\nabla^2 T \approx \frac{T_{i+1} - 2T_i + T_{i-1}}{(\Delta x)^2}$$

### 4.3 Stochastic Modeling

Power variations are modeled with Gaussian noise:

$$\epsilon(t) \sim \mathcal{N}(0, \sigma^2)$$

Where σ ≈ 0.02 × TDP for typical GPU workloads.

---

## 5. Model Validation

### 5.1 GPU Power Model Validation

The power model has been validated against:
- NVIDIA SMI power readings
- RAPL energy counters
- External power meter measurements

Typical accuracy: ±5% for mean power, ±10% for transient peaks.

### 5.2 Thermal Model Validation

Thermal predictions validated against:
- GPU junction temperature sensors
- CFD simulations (ANSYS Fluent, OpenFOAM)
- Experimental immersion cooling data

Typical accuracy: ±2°C for junction temperature.

### 5.3 System-Level Validation

Partial PUE (pPUE) predictions validated against:
- Facility metering data
- CDU power measurements
- Pump flow rate verification

---

## References

1. NVIDIA H100/H200/B200 GPU Architecture Whitepapers
2. 3M Novec Engineered Fluids Thermal Properties Data
3. Incropera, F.P. et al., "Fundamentals of Heat and Mass Transfer"
4. Kaplan, J.M. et al., "Scaling Laws for Neural Language Models"
5. ASHRAE TC 9.9 Data Center Thermal Guidelines

---

*Document Version: 1.0*  
*Last Updated: February 2026*  
*Author: Firmus Engineering Team*
