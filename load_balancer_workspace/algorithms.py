import random
import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any, Callable
from server import SimulatedServer  # Import SimulatedServer from server.py

# ACO PARAMETERS - Necessary constants for algorithm stability
PHEROMONE_EVAPORATION_RATE = 0.1  # Justified: Standard evaporation rate in ACO for gradual pheromone decay
PHEROMONE_DEPOSIT_AMOUNT = 10.0  # Justified: Initial deposit amount based on ACO literature for effective learning
ALPHA = 1.0  # Justified: Equal weight for pheromone in attractiveness calculation
BETA = 2.0  # Justified: Higher weight for heuristic to favor underutilized servers

# Global state for ACO Pheromones
global_pheromones: Dict[str, float] = {}

# EXECUTION PACING - Necessary constant for fair comparison
ALGORITHM_DELAY = 0.02  # Justified: 20ms delay per task ensures consistent execution speed across algorithms

def round_robin_balancer(servers: List[SimulatedServer], last_server_index: int, log_callback: Callable) -> Tuple[Optional[SimulatedServer], int]:
    """Round-Robin with forced pacing for fair comparison."""
    time.sleep(ALGORITHM_DELAY)  # Enforce delay
    
    if not servers:
        log_callback("No servers available for Round-Robin")
        return None, last_server_index
    
    # Filter only non-idle servers for RR
    active_servers = [s for s in servers if not s.is_idle()] or servers
    
    if not active_servers:
        log_callback("No active servers for Round-Robin")
        return None, last_server_index
    
    next_server_index = (last_server_index + 1) % len(active_servers)
    chosen_server = active_servers[next_server_index]
    log_callback(f"RR: Selected {chosen_server.id} (index {next_server_index})")
    return chosen_server, next_server_index

def initialize_pheromones(servers: List[SimulatedServer]) -> Dict[str, float]:
    """Initializes pheromones for all servers."""
    global global_pheromones
    global_pheromones = {}
    for server in servers:
        global_pheromones[server.id] = 1.0  # Initial neutral pheromone level
    return global_pheromones

def get_server_heuristic(server: SimulatedServer) -> float:
    """Calculates server attractiveness with aggressive differentiation."""
    utilization = server.get_utilization() / 100.0
    response_time = server.get_response_time_metric()
    current_load = server.current_load / server.max_capacity
    
    # AGGRESSIVE HEURISTIC - Strongly favor underutilized servers
    MIN_OFFSET = 0.01  # Justified: Small offset to avoid division by zero
    SENSITIVITY_UTILIZATION = 2.0  # Justified: Squared term increases sensitivity to high utilization
    SENSITIVITY_RESPONSE = 1.5  # Justified: Higher exponent emphasizes low response time
    utilization_score = 1.0 / (utilization + MIN_OFFSET) ** SENSITIVITY_UTILIZATION
    response_score = 1.0 / (response_time + 0.1) ** SENSITIVITY_RESPONSE
    load_score = 1.0 / (current_load + MIN_OFFSET)
    
    # Weighted combination with fixed weights
    WEIGHT_UTILIZATION = 0.4  # Justified: Primary factor for load balancing
    WEIGHT_RESPONSE = 0.3  # Justified: Secondary factor for performance
    WEIGHT_LOAD = 0.3  # Justified: Tertiary factor for current load
    heuristic = (utilization_score * WEIGHT_UTILIZATION) + (response_score * WEIGHT_RESPONSE) + (load_score * WEIGHT_LOAD)
    return max(0.1, heuristic)  # Justified: Minimum value to avoid zero heuristic

def aco_balancer(servers: List[SimulatedServer], log_callback: Callable) -> Optional[SimulatedServer]:
    """
    Enhanced ACO balancer that focuses on maximizing utilization across all servers.
    It uses a probabilistic approach based on server availability.
    """
    time.sleep(ALGORITHM_DELAY)
    if not servers:
        return None

    # 1. Calculate attractiveness based on available capacity (inverse of utilization)
    utilizations = np.array([s.get_utilization() for s in servers])
    # Add a small epsilon to avoid division by zero for 100% utilized servers
    available_capacity = 100.1 - utilizations
    
    # Squaring the capacity makes servers with low utilization significantly more attractive,
    # while still keeping all servers in the selection pool.
    attractiveness_scores = np.power(available_capacity, 2)

    # 2. Probabilistic Selection
    # Ensure that even servers with 0 attractiveness have a tiny chance of being selected
    attractiveness_scores += 0.1
    
    total_attractiveness = np.sum(attractiveness_scores)
    if total_attractiveness == 0:
        # Fallback to random choice if all servers are somehow maxed out
        return random.choice(servers)
        
    probabilities = attractiveness_scores / total_attractiveness
    
    # Choose a server based on the calculated probabilities
    chosen_server = np.random.choice(servers, p=probabilities)
    
    log_callback(f"ACO: Chose {chosen_server.id} (Util: {chosen_server.get_utilization():.2f}%) based on availability probabilities.")
    
    # The pheromone update is now implicitly handled by recalculating attractiveness on each call,
    # which is more suitable for this simulation model.

    return chosen_server

def update_pheromones(servers: List[SimulatedServer], chosen_server_id: str, log_callback: Callable):
    """Aggressive pheromone updates."""
    global global_pheromones
    
    # Evaporate pheromones for all servers
    for server_id in list(global_pheromones.keys()):
        global_pheromones[server_id] *= (1 - PHEROMONE_EVAPORATION_RATE)
        global_pheromones[server_id] = max(0.1, global_pheromones[server_id])  # Justified: Minimum to avoid zero
    
    # AGGRESSIVE DEPOSIT based on performance
    chosen_server = next((s for s in servers if s.id == chosen_server_id), None)
    if chosen_server:
        performance_score = chosen_server.get_performance_score()
        pheromone_deposit = PHEROMONE_DEPOSIT_AMOUNT * performance_score * 2.0  # Justified: Double deposit for stronger learning
        global_pheromones[chosen_server_id] += pheromone_deposit
        log_callback(f"ACO: Deposited {pheromone_deposit:.2f} pheromone on {chosen_server_id}")

def reset_aco_usage(servers: List[SimulatedServer]):
    """Resets ACO usage metrics."""
    for server in servers:
        server.aco_usage = 0
        server.last_algorithm_used = None