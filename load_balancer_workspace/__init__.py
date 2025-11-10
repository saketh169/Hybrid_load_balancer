"""
Load Balancer Simulation Package v2.0.1 - OPTIMIZED

A comprehensive simulation framework for comparing load balancing algorithms
including Round Robin, Ant Colony Optimization (ACO), and Q-Learning Hybrid approaches.
"""

from CC.load_balancer_workspace.server import SimulatedServer, initialize_servers
from CC.load_balancer_workspace.algorithms import (
    round_robin_balancer, aco_balancer, 
    initialize_pheromones, update_pheromones, reset_aco_usage
)
from CC.load_balancer_workspace.hybrid import (
    hybrid_balancer, discretize_state, 
    reset_q_learning, get_q_table_status, get_exploration_rate
)
from CC.load_balancer_workspace.visualization import (
    log_metrics, visualize_assignment_step, 
    on_pick, analyze_and_plot, generate_simulation_id, current_simulation_id
)
from CC.load_balancer_workspace.ui import run_ui

__version__ = "2.0.1"
__author__ = "Load Balancer Simulation Team"

__all__ = [
    # Core components
    'SimulatedServer', 'initialize_servers',
    
    # Algorithms
    'round_robin_balancer', 'aco_balancer',
    'initialize_pheromones', 'update_pheromones', 'reset_aco_usage',
    
    # Hybrid approach
    'hybrid_balancer', 'discretize_state', 
    'reset_q_learning', 'get_q_table_status', 'get_exploration_rate',
    
    # Visualization
    'log_metrics', 'visualize_assignment_step',
    'on_pick', 'analyze_and_plot', 'generate_simulation_id', 'current_simulation_id',
    
    # UI
    'run_ui'
]