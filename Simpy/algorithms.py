import random
import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any, Callable
from server import SimulatedServer  # Import SimulatedServer from server.py

# ACO algorithm parameters
PHEROMONE_EVAPORATION_RATE = 0.1   # Gradual pheromone decay
PHEROMONE_DEPOSIT_AMOUNT = 10.0     # Base deposit amount
ALPHA = 1.0  # Pheromone weight
BETA = 2.0   # Heuristic weight

# Global pheromone state
global_pheromones: Dict[str, float] = {}

# Consistent delay for fair comparison
ALGORITHM_DELAY = 0.02

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
    """Calculate server attractiveness based on utilization, response time, and load."""
    utilization = server.get_utilization() / 100.0
    response_time = server.get_response_time_metric()
    current_load = server.current_load / server.max_capacity
    
    # Calculate component scores
    MIN_OFFSET = 0.01
    SENSITIVITY_UTILIZATION = 2.0
    SENSITIVITY_RESPONSE = 1.5
    utilization_score = 1.0 / (utilization + MIN_OFFSET) ** SENSITIVITY_UTILIZATION
    response_score = 1.0 / (response_time + 0.1) ** SENSITIVITY_RESPONSE
    load_score = 1.0 / (current_load + MIN_OFFSET)
    
    # Weighted combination
    WEIGHT_UTILIZATION = 0.4
    WEIGHT_RESPONSE = 0.3
    WEIGHT_LOAD = 0.3
    heuristic = ((utilization_score * WEIGHT_UTILIZATION) + 
                 (response_score * WEIGHT_RESPONSE) + 
                 (load_score * WEIGHT_LOAD))
    return max(0.1, heuristic)

def aco_balancer(servers: List[SimulatedServer], log_callback: Callable) -> Optional[SimulatedServer]:
    """
    ACO balancer using pheromone trails and server heuristics.
    Balances between exploration and exploitation using probabilistic selection.
    """
    time.sleep(ALGORITHM_DELAY)
    if not servers:
        return None

    # Calculate attractiveness for each server
    attractiveness_scores = []
    
    for server in servers:
        # Get pheromone level for this server
        pheromone = global_pheromones.get(server.id, 1.0)
        
        # Get server heuristic (based on utilization, response time, load)
        heuristic = get_server_heuristic(server)
        
        # ACO formula: attractiveness = (pheromone^ALPHA) * (heuristic^BETA)
        attractiveness = (pheromone ** ALPHA) * (heuristic ** BETA)
        attractiveness_scores.append(attractiveness)
    
    # Convert to numpy array for easier manipulation
    attractiveness_scores = np.array(attractiveness_scores)
    
    # Add small epsilon to avoid division by zero
    attractiveness_scores += 1e-10
    
    # Calculate selection probabilities
    total_attractiveness = np.sum(attractiveness_scores)
    probabilities = attractiveness_scores / total_attractiveness
    
    # Probabilistic selection
    try:
        chosen_server = np.random.choice(servers, p=probabilities)
    except ValueError:
        # Fallback to random selection
        chosen_server = random.choice(servers)
    
    # Update pheromones
    update_pheromones(servers, chosen_server.id, log_callback)
    
    log_callback(f"ACO: Selected {chosen_server.id} (Util: {chosen_server.get_utilization():.2f}%)")
    
    return chosen_server

def update_pheromones(servers: List[SimulatedServer], chosen_server_id: str, log_callback: Callable):
    """Update pheromone trails after server selection."""
    global global_pheromones
    
    # 1. Evaporate pheromones for all servers
    for server_id in list(global_pheromones.keys()):
        global_pheromones[server_id] *= (1 - PHEROMONE_EVAPORATION_RATE)
        global_pheromones[server_id] = max(0.1, global_pheromones[server_id])
    
    # 2. Deposit pheromone on chosen server based on performance
    chosen_server = next((s for s in servers if s.id == chosen_server_id), None)
    if chosen_server:
        performance_score = chosen_server.get_performance_score()
        pheromone_deposit = PHEROMONE_DEPOSIT_AMOUNT * performance_score
        global_pheromones[chosen_server_id] += pheromone_deposit
        
        log_callback(f"ACO: Deposited {pheromone_deposit:.2f} pheromone on {chosen_server_id}")

def reset_aco_usage(servers: List[SimulatedServer]):
    """Resets ACO usage metrics."""
    for server in servers:
        server.aco_usage = 0
        server.last_algorithm_used = None