"""Demand response algorithms and bidding strategies.

This module implements demand response optimization, bid calculation,
and workload deferral strategies for grid service participation.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
import time


@dataclass
class DREvent:
    """Demand response event specification."""
    event_id: str
    start_time: float       # Unix timestamp
    end_time: float         # Unix timestamp
    target_reduction: float # Power reduction target (W)
    price: float            # Compensation ($/kWh curtailed)
    penalty: float          # Penalty for non-performance ($/kWh)


class DemandResponseManager:
    """Demand response bid optimization and execution.
    
    Manages participation in utility demand response programs:
    - Economic DR (price-responsive load reduction)
    - Emergency DR (grid reliability events)
    - Ancillary services (regulation, reserves)
    """
    
    def __init__(self, P_max: float, P_min: float = 0.0):
        """Initialize demand response manager.
        
        Args:
            P_max: Maximum facility power (W)
            P_min: Minimum operational power (W)
        """
        self.P_max = P_max
        self.P_min = P_min
        
        # Active DR events
        self.active_events: List[DREvent] = []
        
        # Performance tracking
        self.baseline_power = 0.0
        self.actual_reduction = 0.0
    
    def calculate_available_reduction(self, P_current: float, 
                                     thermal_headroom: float,
                                     deferrable_load: float) -> float:
        """Calculate available load reduction capacity.
        
        Args:
            P_current: Current power consumption (W)
            thermal_headroom: Power reduction from thermal margin (W)
            deferrable_load: Power from deferrable workloads (W)
            
        Returns:
            available_reduction: Available DR capacity (W)
        """
        # Maximum reduction limited by minimum operational power
        max_reduction_power = P_current - self.P_min
        
        # Available reduction from workload deferral and thermal headroom
        available_reduction = min(
            max_reduction_power,
            deferrable_load + thermal_headroom
        )
        
        return max(0, available_reduction)
    
    def optimize_dr_bid(self, price_forecast: np.ndarray,
                       available_reduction: np.ndarray,
                       duration_hours: int = 4) -> Tuple[float, float]:
        """Determine optimal DR bid for economic program.
        
        Args:
            price_forecast: Electricity price forecast ($/kWh)
            available_reduction: Available reduction per hour (W)
            duration_hours: Event duration (hours)
            
        Returns:
            bid_power: Power reduction bid (W)
            expected_revenue: Expected revenue ($)
        """
        # Identify high-price periods (top 20%)
        price_threshold = np.percentile(price_forecast, 80)
        high_price_mask = price_forecast >= price_threshold
        
        # Calculate potential revenue for each hour
        revenue_per_hour = available_reduction * price_forecast / 1000  # Convert W to kW
        
        # Select hours with highest revenue potential
        sorted_indices = np.argsort(revenue_per_hour)[::-1]
        selected_hours = sorted_indices[:duration_hours]
        
        # Bid is minimum available reduction across selected hours
        bid_power = np.min(available_reduction[selected_hours])
        
        # Expected revenue
        expected_revenue = np.sum(bid_power * price_forecast[selected_hours] / 1000)
        
        return bid_power, expected_revenue
    
    def evaluate_dr_event(self, event: DREvent, P_baseline: float,
                         available_reduction: float) -> Dict[str, float]:
        """Evaluate participation in DR event.
        
        Args:
            event: DR event specification
            P_baseline: Baseline power consumption (W)
            available_reduction: Available reduction capacity (W)
            
        Returns:
            evaluation: Dict with revenue, cost, and net benefit
        """
        duration_hours = (event.end_time - event.start_time) / 3600
        
        # Can we meet the target?
        can_participate = available_reduction >= event.target_reduction
        
        if can_participate:
            # Revenue from curtailment
            energy_curtailed_kwh = event.target_reduction * duration_hours / 1000
            revenue = energy_curtailed_kwh * event.price
            
            # No penalty
            penalty_cost = 0.0
            
            # Net benefit
            net_benefit = revenue
        else:
            # Partial participation
            energy_curtailed_kwh = available_reduction * duration_hours / 1000
            revenue = energy_curtailed_kwh * event.price
            
            # Penalty for shortfall
            shortfall_kwh = (event.target_reduction - available_reduction) * duration_hours / 1000
            penalty_cost = shortfall_kwh * event.penalty
            
            # Net benefit (may be negative)
            net_benefit = revenue - penalty_cost
        
        return {
            'can_participate': can_participate,
            'revenue': revenue,
            'penalty': penalty_cost,
            'net_benefit': net_benefit,
            'recommended': net_benefit > 0
        }
    
    def calculate_baseline(self, historical_power: np.ndarray,
                          method: str = 'average') -> float:
        """Calculate baseline power for DR measurement.
        
        Args:
            historical_power: Historical power data (W)
            method: Baseline calculation method ('average', 'median', 'peak')
            
        Returns:
            baseline: Baseline power (W)
        """
        if method == 'average':
            baseline = np.mean(historical_power)
        elif method == 'median':
            baseline = np.median(historical_power)
        elif method == 'peak':
            baseline = np.percentile(historical_power, 95)
        else:
            raise ValueError(f"Unknown baseline method: {method}")
        
        return baseline
    
    def measure_performance(self, P_actual: float, P_baseline: float,
                           target_reduction: float) -> Dict[str, float]:
        """Measure DR event performance.
        
        Args:
            P_actual: Actual power during event (W)
            P_baseline: Baseline power (W)
            target_reduction: Target reduction (W)
            
        Returns:
            performance: Dict with actual reduction and performance ratio
        """
        actual_reduction = P_baseline - P_actual
        
        if target_reduction > 0:
            performance_ratio = actual_reduction / target_reduction
        else:
            performance_ratio = 0.0
        
        return {
            'actual_reduction': actual_reduction,
            'target_reduction': target_reduction,
            'performance_ratio': performance_ratio,
            'meets_target': actual_reduction >= target_reduction
        }


class WorkloadDeferralStrategy:
    """Strategy for identifying and deferring workloads during DR events."""
    
    def __init__(self):
        """Initialize workload deferral strategy."""
        self.deferred_jobs = []
    
    def classify_workload(self, job: Dict, current_time: float) -> str:
        """Classify workload by time-sensitivity.
        
        Args:
            job: Workload job with deadline and compute requirements
            current_time: Current Unix timestamp
            
        Returns:
            classification: 'critical', 'normal', or 'deferrable'
        """
        time_to_deadline = job['deadline'] - current_time
        compute_time = job['compute_hours'] * 3600  # Convert to seconds
        
        slack = time_to_deadline - compute_time
        
        if slack < 3600:  # Less than 1 hour slack
            return 'critical'
        elif slack < 24 * 3600:  # Less than 1 day slack
            return 'normal'
        else:
            return 'deferrable'
    
    def calculate_deferrable_power(self, workload_queue: List[Dict],
                                  current_time: float) -> float:
        """Calculate total power from deferrable workloads.
        
        Args:
            workload_queue: List of pending workload jobs
            current_time: Current Unix timestamp
            
        Returns:
            deferrable_power: Total deferrable power (W)
        """
        deferrable_power = 0.0
        
        for job in workload_queue:
            classification = self.classify_workload(job, current_time)
            
            if classification == 'deferrable':
                deferrable_power += job.get('power_requirement', 0)
        
        return deferrable_power
    
    def defer_workloads(self, workload_queue: List[Dict],
                       target_reduction: float,
                       current_time: float) -> Tuple[List[Dict], float]:
        """Defer workloads to achieve target power reduction.
        
        Args:
            workload_queue: List of pending workload jobs
            target_reduction: Target power reduction (W)
            current_time: Current Unix timestamp
            
        Returns:
            deferred_jobs: List of deferred jobs
            actual_reduction: Actual power reduction achieved (W)
        """
        # Sort jobs by priority (lower priority deferred first)
        sorted_jobs = sorted(workload_queue, 
                           key=lambda j: (self.classify_workload(j, current_time), 
                                        -j.get('priority', 5)))
        
        deferred_jobs = []
        actual_reduction = 0.0
        
        for job in sorted_jobs:
            if actual_reduction >= target_reduction:
                break
            
            classification = self.classify_workload(job, current_time)
            
            if classification in ['deferrable', 'normal']:
                deferred_jobs.append(job)
                actual_reduction += job.get('power_requirement', 0)
        
        self.deferred_jobs = deferred_jobs
        return deferred_jobs, actual_reduction
