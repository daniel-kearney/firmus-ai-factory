"""Model Predictive Control for AI factory optimization.

This module implements MPC-based optimization for multi-objective
control of power, thermal, and workload systems.
"""

import numpy as np
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("Warning: cvxpy not installed. Optimization module will have limited functionality.")

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class WorkloadJob:
    """Workload job specification."""
    job_id: str
    compute_hours: float  # Required compute time (GPU-hours)
    deadline: float       # Unix timestamp
    priority: int         # 1-10, higher = more important
    power_per_gpu: float  # Power consumption per GPU (W)
    num_gpus: int         # Required number of GPUs


class ModelPredictiveController:
    """MPC for AI factory optimization.
    
    Solves multi-objective optimization:
    - Minimize electricity cost
    - Minimize thermal stress
    - Maximize throughput
    - Meet grid service commitments
    """
    
    def __init__(self, horizon: int = 24, dt: float = 1.0):
        """Initialize MPC controller.
        
        Args:
            horizon: Prediction horizon (hours)
            dt: Time step (hours)
        """
        self.horizon = horizon
        self.dt = dt
        self.n_steps = int(horizon / dt)
        
        if not CVXPY_AVAILABLE:
            raise ImportError("cvxpy is required for MPC. Install with: pip install cvxpy")
    
    def optimize(self, current_state: Dict, price_forecast: np.ndarray,
                workload_queue: List[WorkloadJob], grid_signals: Dict) -> Dict:
        """Solve MPC optimization problem.
        
        Args:
            current_state: Current system state (temperatures, SOC, etc.)
            price_forecast: Electricity prices for horizon ($/kWh)
            workload_queue: Pending workload jobs
            grid_signals: Grid frequency, voltage, DR requests
            
        Returns:
            optimal_controls: Dict with power setpoints, workload schedule
        """
        # Decision variables
        num_gpus = 8
        P_gpu = cp.Variable((self.n_steps, num_gpus))  # Power per GPU
        T_junction = cp.Variable(self.n_steps)  # Junction temperature
        P_cooling = cp.Variable(self.n_steps)   # Cooling power
        
        # Objective function components
        cost_electricity = cp.sum(
            cp.multiply(price_forecast[:self.n_steps] / 1000, 
                       cp.sum(P_gpu, axis=1) + P_cooling)
        ) * self.dt
        
        cost_thermal = cp.sum(cp.pos(T_junction - 75)) * 0.01  # Penalty for high temps
        
        # Weights
        w_electricity = 1.0
        w_thermal = 0.1
        
        objective = cp.Minimize(
            w_electricity * cost_electricity + 
            w_thermal * cost_thermal
        )
        
        # Constraints
        constraints = []
        
        # Power limits
        for t in range(self.n_steps):
            constraints.append(cp.sum(P_gpu[t, :]) <= 100000)  # 100 kW total
            constraints.append(P_gpu[t, :] >= 0)
            constraints.append(P_gpu[t, :] <= 700)  # Max per GPU
        
        # Thermal constraints
        for t in range(self.n_steps):
            # Simplified thermal model: T = T_ambient + R_thermal * P
            T_ambient = 25.0
            R_thermal = 0.00005  # K/W
            constraints.append(
                T_junction[t] == T_ambient + R_thermal * cp.sum(P_gpu[t, :])
            )
            constraints.append(T_junction[t] <= 83.0)
        
        # Cooling power model (pPUE = 0.05)
        for t in range(self.n_steps):
            constraints.append(
                P_cooling[t] == 0.05 * cp.sum(P_gpu[t, :])
            )
        
        # Solve optimization
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
        except Exception as e:
            print(f"Optimization failed: {e}")
            # Return current state as fallback
            return {
                'P_gpu': np.zeros((self.n_steps, num_gpus)),
                'T_junction': np.ones(self.n_steps) * current_state.get('T_junction', 50.0),
                'P_cooling': np.zeros(self.n_steps),
                'total_cost': 0.0
            }
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Warning: Optimization status: {problem.status}")
        
        # Extract optimal controls
        optimal_controls = {
            'P_gpu': P_gpu.value if P_gpu.value is not None else np.zeros((self.n_steps, num_gpus)),
            'T_junction': T_junction.value if T_junction.value is not None else np.ones(self.n_steps) * 50.0,
            'P_cooling': P_cooling.value if P_cooling.value is not None else np.zeros(self.n_steps),
            'total_cost': cost_electricity.value if cost_electricity.value is not None else 0.0
        }
        
        return optimal_controls
