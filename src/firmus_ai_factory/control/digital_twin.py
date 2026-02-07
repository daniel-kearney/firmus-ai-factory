"""Digital twin integration for complete AI factory system.

This module integrates all subsystems into a unified digital twin
with real-time optimization and grid communication.
"""

import numpy as np
from typing import Dict, Optional
import time


class DigitalTwin:
    """Complete AI factory digital twin.
    
    Integrates:
    - GPU computational models
    - Thermal management systems
    - Power delivery networks
    - Grid interface
    - Energy storage
    - Economic optimization
    """
    
    def __init__(self, config: Dict):
        """Initialize digital twin.
        
        Args:
            config: Configuration dict with all subsystem parameters
        """
        self.config = config
        
        # System state
        self.state = {
            'T_junction': 50.0,
            'T_coolant': 35.0,
            'SOC': 0.5,
            'P_gpu': np.zeros(8),
            'workload_queue': []
        }
        
        # Control outputs
        self.controls = {
            'P_gpu_setpoint': np.zeros(8),
            'cooling_setpoint': 35.0,
        }
        
        # Time tracking
        self.current_time = time.time()
    
    def update(self, dt: float = 1.0, external_inputs: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Update digital twin for one time step.
        
        Args:
            dt: Time step (seconds)
            external_inputs: Dict with external signals (prices, weather, etc.)
            
        Returns:
            state: Updated system state
            outputs: System outputs (power, cost, etc.)
        """
        if external_inputs is None:
            external_inputs = {}
        
        # Extract external inputs
        T_ambient = external_inputs.get('T_ambient', 25.0)
        electricity_price = external_inputs.get('electricity_price', 0.12)
        
        # Update GPU power based on workload
        total_gpu_power = np.sum(self.state['P_gpu'])
        
        # Simple thermal model
        R_thermal = 0.00005  # K/W
        self.state['T_junction'] = T_ambient + R_thermal * total_gpu_power
        
        # Cooling power (pPUE = 0.05)
        P_cooling = 0.05 * total_gpu_power
        
        # Total power
        P_total = total_gpu_power + P_cooling
        
        # Calculate outputs
        outputs = {
            'P_total': P_total,
            'T_junction_max': self.state['T_junction'],
            'pPUE': 1.05,
            'electricity_cost': P_total * electricity_price * dt / (3600 * 1000),
            'SOC': self.state['SOC']
        }
        
        self.current_time += dt
        
        return self.state, outputs
    
    def run_scenario(self, duration_hours: float = 24, dt: float = 60) -> Dict:
        """Run scenario simulation.
        
        Args:
            duration_hours: Simulation duration (hours)
            dt: Time step (seconds)
            
        Returns:
            results: Dict with time-series results
        """
        n_steps = int(duration_hours * 3600 / dt)
        
        # Initialize result arrays
        results = {
            'time': np.zeros(n_steps),
            'P_total': np.zeros(n_steps),
            'T_junction': np.zeros(n_steps),
            'cost': np.zeros(n_steps)
        }
        
        # Simulation loop
        for k in range(n_steps):
            # External inputs
            external_inputs = {
                'T_ambient': 25.0 + 5 * np.sin(2 * np.pi * k / (24 * 3600 / dt)),
                'electricity_price': 0.10 + 0.05 * np.sin(2 * np.pi * k / (24 * 3600 / dt))
            }
            
            # Update digital twin
            state, outputs = self.update(dt, external_inputs)
            
            # Store results
            results['time'][k] = k * dt / 3600
            results['P_total'][k] = outputs['P_total']
            results['T_junction'][k] = outputs['T_junction_max']
            results['cost'][k] = outputs['electricity_cost']
        
        # Cumulative cost
        results['cumulative_cost'] = np.cumsum(results['cost'])
        
        return results
