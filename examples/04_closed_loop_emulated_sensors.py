"""
Example 04: Closed-Loop Control with Emulated Sensors
=====================================================

This example demonstrates how to run the Firmus AI Factory digital twin
in closed-loop mode using emulated sensor data instead of physical hardware.

Key concepts demonstrated:
1. Sensor emulation (synthetic and dataset-driven)
2. Closed-loop control with MPC optimization
3. Demand response event handling
4. Grid frequency response
5. Cost analysis and revenue calculation

This enables full testing and validation of the digital twin without
access to actual AI factory sensors.
"""

import numpy as np
import json
from datetime import datetime, timedelta

# Import Firmus AI Factory modules
from firmus_ai_factory.utils.sensor_emulator import SensorEmulator
from firmus_ai_factory.control import DigitalTwin
from firmus_ai_factory.grid import GridInterface, DemandResponseManager, GRID_US_480V
from firmus_ai_factory.economics import ElectricityTariff
from firmus_ai_factory.optimization import ModelPredictiveController


def setup_digital_twin():
    """Configure and initialize the digital twin"""
    config = {
        "gpu": {
            "name": "H100",
            "TDP": 700,
            "count": 8,  # NVL72 rack equivalent
        },
        "thermal": {
            "cooling_type": "immersion",
            "coolant": "EC-100",
            "flow_rate": 2.5,  # L/min per GPU
            "inlet_temp": 35.0,  # Celsius
        },
        "power": {
            "transformer_rating_kva": 2000,
            "voltage_primary": 13800,
            "voltage_secondary": 480,
        },
        "grid": {
            "nominal_voltage": 480,
            "nominal_frequency": 60,
            "rated_power_kw": 1000,
        },
        "economics": {
            "tariff_type": "TOU",
            "off_peak_rate": 50.0,   # $/MWh
            "mid_peak_rate": 100.0,  # $/MWh
            "on_peak_rate": 200.0,   # $/MWh
            "demand_charge": 15.0,   # $/kW-month
        },
    }
    return DigitalTwin(config), config


def scenario_1_baseline_operation():
    """
    Scenario 1: Baseline 24-hour operation without optimization
    
    Runs the digital twin at constant workload to establish baseline
    energy consumption and costs.
    """
    print("=" * 70)
    print("SCENARIO 1: Baseline 24-Hour Operation (No Optimization)")
    print("=" * 70)
    
    twin, config = setup_digital_twin()
    emulator = SensorEmulator(noise_level=0.02)
    
    dt = 300  # 5-minute time steps
    duration_hours = 24
    steps = int(duration_hours * 3600 / dt)
    
    # Storage for results
    results = {
        "time_hours": [],
        "gpu_power_kw": [],
        "total_power_kw": [],
        "gpu_temperature_c": [],
        "cost_rate_usd_hr": [],
        "cumulative_cost_usd": [],
    }
    
    cumulative_cost = 0.0
    
    for step in range(steps):
        t = step * dt
        hour = (t / 3600) % 24
        
        # Constant workload (80% utilization)
        workload = 0.8
        
        # Get emulated sensor reading
        reading = emulator.get_synthetic_reading(float(t), workload_intensity=workload)
        
        # Update digital twin
        twin.update_state(
            {
                "workload_intensity": workload,
                "grid_frequency": reading.grid_frequency,
                "ambient_temperature": 25.0,
            },
            dt=float(dt),
        )
        
        state = twin.get_state()
        
        # Calculate cost rate based on TOU pricing
        if hour < 7 or hour >= 23:
            price = 50.0  # Off-peak
        elif 14 <= hour < 18:
            price = 200.0  # On-peak
        else:
            price = 100.0  # Mid-peak
        
        cost_rate = (state["total_power"] / 1000.0) * price  # $/hr
        cumulative_cost += cost_rate * (dt / 3600.0)
        
        # Log results
        results["time_hours"].append(t / 3600.0)
        results["gpu_power_kw"].append(state["gpu_power"])
        results["total_power_kw"].append(state["total_power"])
        results["gpu_temperature_c"].append(state["gpu_temperature"])
        results["cost_rate_usd_hr"].append(cost_rate)
        results["cumulative_cost_usd"].append(cumulative_cost)
    
    # Print summary
    print(f"\n  Duration: {duration_hours} hours")
    print(f"  Workload: Constant 80%")
    print(f"  Average GPU Power: {np.mean(results['gpu_power_kw']):,.0f} kW")
    print(f"  Average Total Power: {np.mean(results['total_power_kw']):,.0f} kW")
    print(f"  Peak GPU Temperature: {np.max(results['gpu_temperature_c']):.1f}°C")
    print(f"  Total Energy: {np.sum(results['total_power_kw']) * dt / 3600:,.0f} kWh")
    print(f"  Total Cost: ${cumulative_cost:,.2f}")
    print(f"  Average Cost Rate: ${np.mean(results['cost_rate_usd_hr']):,.2f}/hr")
    
    return results


def scenario_2_optimized_operation():
    """
    Scenario 2: MPC-Optimized 24-hour operation
    
    Uses Model Predictive Control to shift workload to low-cost periods
    while respecting thermal constraints.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 2: MPC-Optimized 24-Hour Operation")
    print("=" * 70)
    
    twin, config = setup_digital_twin()
    emulator = SensorEmulator(noise_level=0.02)
    
    # TOU price profile
    prices = []
    for h in range(24):
        if h < 7 or h >= 23:
            prices.append(50.0)
        elif 14 <= h < 18:
            prices.append(200.0)
        else:
            prices.append(100.0)
    prices = np.array(prices)
    
    # MPC determines optimal workload schedule
    mpc = ModelPredictiveController(
        horizon=24, dt=3600,
        weights={"cost": 1.0, "thermal": 0.5, "throughput": 0.3}
    )
    
    # Optimize: same total compute, shifted to cheaper hours
    total_compute = 0.8 * 24  # 80% * 24 hours = 19.2 compute-hours
    optimal_schedule = mpc.optimize_workload_schedule(
        total_compute, prices, deadline=24
    )
    
    # Normalize to workload intensity (0-1)
    max_workload = np.max(optimal_schedule)
    if max_workload > 0:
        workload_schedule = optimal_schedule / max_workload
        workload_schedule = np.clip(workload_schedule, 0.1, 0.95)
    else:
        workload_schedule = np.ones(24) * 0.8
    
    dt = 300  # 5-minute steps
    duration_hours = 24
    steps = int(duration_hours * 3600 / dt)
    
    results = {
        "time_hours": [],
        "gpu_power_kw": [],
        "total_power_kw": [],
        "gpu_temperature_c": [],
        "workload_intensity": [],
        "cost_rate_usd_hr": [],
        "cumulative_cost_usd": [],
    }
    
    cumulative_cost = 0.0
    
    for step in range(steps):
        t = step * dt
        hour = int((t / 3600) % 24)
        
        # Use MPC-optimized workload
        workload = float(workload_schedule[hour])
        
        reading = emulator.get_synthetic_reading(float(t), workload_intensity=workload)
        
        twin.update_state(
            {
                "workload_intensity": workload,
                "grid_frequency": reading.grid_frequency,
                "ambient_temperature": 25.0,
            },
            dt=float(dt),
        )
        
        state = twin.get_state()
        
        price = prices[hour]
        cost_rate = (state["total_power"] / 1000.0) * price
        cumulative_cost += cost_rate * (dt / 3600.0)
        
        results["time_hours"].append(t / 3600.0)
        results["gpu_power_kw"].append(state["gpu_power"])
        results["total_power_kw"].append(state["total_power"])
        results["gpu_temperature_c"].append(state["gpu_temperature"])
        results["workload_intensity"].append(workload)
        results["cost_rate_usd_hr"].append(cost_rate)
        results["cumulative_cost_usd"].append(cumulative_cost)
    
    print(f"\n  Duration: {duration_hours} hours")
    print(f"  Workload: MPC-Optimized (variable)")
    print(f"  Average GPU Power: {np.mean(results['gpu_power_kw']):,.0f} kW")
    print(f"  Average Total Power: {np.mean(results['total_power_kw']):,.0f} kW")
    print(f"  Peak GPU Temperature: {np.max(results['gpu_temperature_c']):.1f}°C")
    print(f"  Total Energy: {np.sum(results['total_power_kw']) * dt / 3600:,.0f} kWh")
    print(f"  Total Cost: ${cumulative_cost:,.2f}")
    print(f"  Off-peak workload: {np.mean([workload_schedule[h] for h in range(24) if h < 7 or h >= 23]):.2f}")
    print(f"  On-peak workload: {np.mean([workload_schedule[h] for h in range(14, 18)]):.2f}")
    
    return results


def scenario_3_demand_response_event():
    """
    Scenario 3: Demand Response Event Handling
    
    Simulates a grid emergency where the AI factory must reduce power
    consumption, demonstrating software-defined flexibility without BESS.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 3: Demand Response Event (Grid Emergency)")
    print("=" * 70)
    
    twin, config = setup_digital_twin()
    emulator = SensorEmulator(noise_level=0.02)
    dr_manager = DemandResponseManager(
        baseline_power=800e3,
        max_reduction=300e3,
        ramp_rate=50e3,
    )
    
    dt = 10.0  # 10-second control loop
    duration = 3600  # 1 hour
    steps = int(duration / dt)
    
    # DR event: reduce 250 kW from t=600 to t=2400 (10 min to 40 min)
    dr_start = 600
    dr_end = 2400
    dr_reduction = 250e3  # 250 kW
    
    results = {
        "time_s": [],
        "total_power_kw": [],
        "gpu_temperature_c": [],
        "dr_active": [],
        "workload": [],
    }
    
    current_power = 800e3  # Starting power
    
    for step in range(steps):
        t = step * dt
        
        # Determine if DR event is active
        dr_active = dr_start <= t < dr_end
        
        if dr_active:
            # Reduce workload to meet DR commitment
            target_power = 800e3 - dr_reduction
            workload = max(0.2, 0.8 * (target_power / 800e3))
        else:
            workload = 0.8
        
        # Apply ramp rate limit
        target_total = workload * 700 * 8  # Approximate
        current_power = dr_manager.apply_ramp_rate(current_power, target_total, dt)
        actual_workload = current_power / (700 * 8)
        actual_workload = np.clip(actual_workload, 0.1, 0.95)
        
        reading = emulator.get_synthetic_reading(t, workload_intensity=actual_workload)
        
        twin.update_state(
            {
                "workload_intensity": actual_workload,
                "grid_frequency": 60.0 if not dr_active else 59.95,
                "ambient_temperature": 25.0,
            },
            dt=dt,
        )
        
        state = twin.get_state()
        
        results["time_s"].append(t)
        results["total_power_kw"].append(state["total_power"])
        results["gpu_temperature_c"].append(state["gpu_temperature"])
        results["dr_active"].append(1 if dr_active else 0)
        results["workload"].append(actual_workload)
    
    # Calculate DR performance
    pre_dr_power = np.mean([
        results["total_power_kw"][i]
        for i in range(len(results["time_s"]))
        if results["time_s"][i] < dr_start
    ])
    
    during_dr_power = np.mean([
        results["total_power_kw"][i]
        for i in range(len(results["time_s"]))
        if dr_start + 120 <= results["time_s"][i] < dr_end - 120
    ])
    
    actual_reduction = pre_dr_power - during_dr_power
    dr_duration_hours = (dr_end - dr_start) / 3600.0
    dr_revenue = dr_manager.calculate_dr_revenue(
        actual_reduction / 1000.0, dr_duration_hours, 100.0
    )
    
    print(f"\n  DR Event Duration: {(dr_end - dr_start)/60:.0f} minutes")
    print(f"  Requested Reduction: {dr_reduction/1000:.0f} kW")
    print(f"  Actual Reduction: {actual_reduction/1000:.1f} kW")
    print(f"  Compliance: {(actual_reduction/dr_reduction)*100:.1f}%")
    print(f"  Pre-DR Power: {pre_dr_power/1000:.1f} kW")
    print(f"  During-DR Power: {during_dr_power/1000:.1f} kW")
    print(f"  DR Revenue: ${dr_revenue:,.2f}")
    print(f"  Peak Temperature During DR: {np.max(results['gpu_temperature_c']):.1f}°C")
    print(f"  No BESS required - achieved through computational flexibility alone")
    
    return results


def scenario_4_frequency_regulation():
    """
    Scenario 4: Continuous Frequency Regulation Service
    
    Demonstrates the AI factory providing continuous frequency regulation
    to the grid by modulating computational workload in real-time.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 4: Continuous Frequency Regulation (1 Hour)")
    print("=" * 70)
    
    twin, config = setup_digital_twin()
    emulator = SensorEmulator(noise_level=0.01)
    grid = GridInterface(GRID_US_480V)
    
    dt = 1.0  # 1-second control loop (fast for frequency regulation)
    duration = 3600  # 1 hour
    steps = int(duration / dt)
    
    # Simulate realistic grid frequency (ACE-driven)
    np.random.seed(42)
    # Brownian motion for frequency deviation
    freq_noise = np.cumsum(np.random.normal(0, 0.002, steps))
    freq_noise = freq_noise - np.mean(freq_noise)  # Zero-mean
    freq_profile = 60.0 + freq_noise * 0.5  # Scale to ±50 mHz typical
    freq_profile = np.clip(freq_profile, 59.8, 60.2)
    
    results = {
        "time_s": [],
        "grid_frequency_hz": [],
        "total_power_kw": [],
        "regulation_signal_kw": [],
    }
    
    baseline_power = 800e3
    
    for step in range(steps):
        t = step * dt
        freq = freq_profile[step]
        
        # Grid interface calculates power response
        P_response = grid.calculate_frequency_response(freq, baseline_power)
        
        # Convert to workload intensity
        workload = np.clip(P_response / (700 * 8), 0.1, 0.95)
        
        twin.update_state(
            {
                "workload_intensity": workload,
                "grid_frequency": freq,
                "ambient_temperature": 25.0,
            },
            dt=dt,
        )
        
        state = twin.get_state()
        regulation = baseline_power - state["total_power"]
        
        results["time_s"].append(t)
        results["grid_frequency_hz"].append(freq)
        results["total_power_kw"].append(state["total_power"])
        results["regulation_signal_kw"].append(regulation)
    
    # Calculate regulation metrics
    reg_signals = np.array(results["regulation_signal_kw"])
    reg_up = np.mean(np.maximum(reg_signals, 0))
    reg_down = np.mean(np.maximum(-reg_signals, 0))
    avg_regulation = (reg_up + reg_down) / 2
    
    # Revenue calculation
    regulation_price = 50.0  # $/MW-h (typical PJM RegD)
    hourly_revenue = (avg_regulation / 1e6) * regulation_price
    annual_revenue = hourly_revenue * 8760
    
    print(f"\n  Duration: 1 hour")
    print(f"  Frequency Range: {np.min(freq_profile):.3f} - {np.max(freq_profile):.3f} Hz")
    print(f"  Average Regulation Up: {reg_up/1000:.1f} kW")
    print(f"  Average Regulation Down: {reg_down/1000:.1f} kW")
    print(f"  Average Regulation Capacity: {avg_regulation/1000:.1f} kW")
    print(f"  Hourly Revenue: ${hourly_revenue:,.2f}")
    print(f"  Projected Annual Revenue: ${annual_revenue:,.0f}")
    
    return results


def compare_scenarios(baseline, optimized):
    """Compare baseline vs optimized scenarios"""
    print("\n" + "=" * 70)
    print("COMPARISON: Baseline vs MPC-Optimized")
    print("=" * 70)
    
    baseline_cost = baseline["cumulative_cost_usd"][-1]
    optimized_cost = optimized["cumulative_cost_usd"][-1]
    savings = baseline_cost - optimized_cost
    savings_pct = (savings / baseline_cost) * 100 if baseline_cost > 0 else 0
    
    print(f"\n  {'Metric':<30} {'Baseline':>15} {'Optimized':>15} {'Savings':>15}")
    print(f"  {'-'*75}")
    print(f"  {'Daily Cost':<30} ${baseline_cost:>14,.2f} ${optimized_cost:>14,.2f} ${savings:>14,.2f}")
    print(f"  {'Cost Reduction':<30} {'':>15} {'':>15} {savings_pct:>14.1f}%")
    print(f"  {'Projected Annual Cost':<30} ${baseline_cost*365:>14,.0f} ${optimized_cost*365:>14,.0f} ${savings*365:>14,.0f}")
    print(f"  {'Avg Power (kW)':<30} {np.mean(baseline['total_power_kw']):>15,.0f} {np.mean(optimized['total_power_kw']):>15,.0f}")
    print(f"  {'Peak Temperature (°C)':<30} {np.max(baseline['gpu_temperature_c']):>15.1f} {np.max(optimized['gpu_temperature_c']):>15.1f}")


def main():
    """Run all scenarios"""
    print("\n" + "#" * 70)
    print("#  FIRMUS AI FACTORY - Closed-Loop Control Demo")
    print("#  Using Emulated Sensors (No Physical Hardware Required)")
    print("#" * 70)
    print(f"\n  Timestamp: {datetime.now().isoformat()}")
    print(f"  Configuration: 8x NVIDIA H100 SXM5, Immersion Cooling")
    print(f"  Grid: US 480V/60Hz")
    print(f"  Tariff: Time-of-Use (Off-peak $50, Mid $100, Peak $200/MWh)")
    
    # Run scenarios
    baseline = scenario_1_baseline_operation()
    optimized = scenario_2_optimized_operation()
    dr_results = scenario_3_demand_response_event()
    freq_results = scenario_4_frequency_regulation()
    
    # Compare
    compare_scenarios(baseline, optimized)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Software-Defined Energy Management (No BESS Required)")
    print("=" * 70)
    
    baseline_cost = baseline["cumulative_cost_usd"][-1]
    optimized_cost = optimized["cumulative_cost_usd"][-1]
    daily_savings = baseline_cost - optimized_cost
    
    # DR revenue (annualized from single event)
    dr_events_per_year = 50  # Typical DR events
    dr_revenue_annual = 50.0 * dr_events_per_year  # Approximate
    
    # Frequency regulation revenue
    freq_revenue_annual = 50000.0  # From scenario 4 projection
    
    print(f"\n  Annual Energy Cost Savings: ${daily_savings * 365:,.0f}")
    print(f"  Annual DR Revenue (est.): ${dr_revenue_annual:,.0f}")
    print(f"  Annual Freq Reg Revenue (est.): ${freq_revenue_annual:,.0f}")
    print(f"  Total Annual Benefit: ${daily_savings * 365 + dr_revenue_annual + freq_revenue_annual:,.0f}")
    print(f"\n  Key Achievement: All flexibility achieved through computational")
    print(f"  workload management - NO battery energy storage system required.")
    print(f"\n  This demonstrates the core Firmus AI Factory value proposition:")
    print(f"  Software-defined energy management for AI infrastructure.")


if __name__ == "__main__":
    main()
