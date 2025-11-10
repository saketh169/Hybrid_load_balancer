import random
import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any, Callable
from algorithms import round_robin_balancer, aco_balancer  # Adjusted import assuming same directory
from server import SimulatedServer  # Import SimulatedServer from server.py

# Q-learning parameters
LEARNING_RATE = 0.6      # Higher rate for faster load adaptation
DISCOUNT_FACTOR = 0.9    # Emphasize future rewards
EXPLORATION_RATE = 0.8   # Initial exploration rate
EXPLORATION_DECAY = 0.995  # Gradual decay rate
MIN_EXPLORATION = 0.05   # Minimum exploration threshold

ALGORITHM_DELAY = 0.02   # Consistent delay for fair comparision

class QLearningBalancer:
    def __init__(self):
        self.q_table = np.zeros((3, 3, 2))  # [load_state, utilization_balance, algorithm]
        self.exploration_rate = EXPLORATION_RATE
        self.learning_iterations = 0
        self.initialize_q_table()
    
    def initialize_q_table(self):
        """Initialize Q-table with neutral values.
        States: [load(0=low,1=med,2=high), balance(0=poor,1=good,2=exc)]
        Actions: 0=RR, 1=ACO"""
        
        # Initialize with small random values to break symmetry
        # This allows the agent to learn from scratch without bias
        self.q_table = np.random.uniform(0, 0.1, (3, 3, 2))

# Global Q-learning instance
q_learner = QLearningBalancer()

def discretize_state(servers: List[SimulatedServer], num_servers: int) -> Tuple[int, int]:
    """Improved state discretization with utilization balance."""
    if not servers:
        return 0, 0
    
    utilizations = [s.get_utilization() for s in servers]
    avg_utilization = np.mean(utilizations) if utilizations else 0
    util_std = np.std(utilizations) if utilizations else 0
    
    # Load state with fixed thresholds
    LOAD_THRESHOLD_LOW = 25.0  # Justified: Empirical low-load threshold
    LOAD_THRESHOLD_HIGH = 70.0  # Justified: Empirical high-load threshold
    if avg_utilization < LOAD_THRESHOLD_LOW:
        load_state = 0
    elif avg_utilization < LOAD_THRESHOLD_HIGH:
        load_state = 1
    else:
        load_state = 2
    
    # Utilization balance state with fixed thresholds
    BALANCE_THRESHOLD_GOOD = 15.0  # Justified: Good balance threshold based on std dev
    BALANCE_THRESHOLD_POOR = 30.0  # Justified: Poor balance threshold based on std dev
    if util_std < BALANCE_THRESHOLD_GOOD:  # Excellent balance
        balance_state = 2
    elif util_std < BALANCE_THRESHOLD_POOR:  # Good balance
        balance_state = 1
    else:  # Poor balance
        balance_state = 0
    
    return load_state, balance_state

def calculate_reward(server: SimulatedServer, previous_utilization: float, algorithm_used: str, system_utilization: float) -> float:
    """Calculate reward based on algorithm performance and system state."""
    current_utilization = server.get_utilization()
    response_time = server.get_response_time_metric()
    
    # Base reward components
    MIN_RESPONSE_OFFSET = 0.05
    
    # Response time reward (lower is better)
    response_reward = 10.0 / (response_time + MIN_RESPONSE_OFFSET)
    
    # Utilization reward (target 60-80% optimal range)
    UTILIZATION_OPTIMAL_LOW = 60.0
    UTILIZATION_OPTIMAL_HIGH = 80.0
    
    if UTILIZATION_OPTIMAL_LOW <= current_utilization <= UTILIZATION_OPTIMAL_HIGH:
        utilization_reward = 10.0
    elif 40.0 <= current_utilization < UTILIZATION_OPTIMAL_LOW:
        utilization_reward = 5.0
    elif current_utilization > UTILIZATION_OPTIMAL_HIGH:
        utilization_reward = 7.0  # Still good, just over optimal
    else:
        utilization_reward = 2.0
    
    # Load balancing improvement reward
    load_improvement = current_utilization - previous_utilization
    if load_improvement > 5.0:
        improvement_reward = 5.0
    elif load_improvement > 0.0:
        improvement_reward = 2.0
    else:
        improvement_reward = 0.0
    
    # Combined reward with balanced weights
    REWARD_WEIGHTS = [0.3, 0.5, 0.2]  # response, utilization, improvement
    
    total_reward = (
        (response_reward * REWARD_WEIGHTS[0]) + 
        (utilization_reward * REWARD_WEIGHTS[1]) + 
        (improvement_reward * REWARD_WEIGHTS[2])
    )
    
    return total_reward


def update_q_value(state: Tuple[int, int], action: int, reward: float, next_state: Tuple[int, int]):
    """Q-value updates using standard Q-learning."""
    current_q = q_learner.q_table[state[0], state[1], action]
    max_next_q = np.max(q_learner.q_table[next_state[0], next_state[1], :])
    
    # Standard Q-learning update
    new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q - current_q)
    q_learner.q_table[state[0], state[1], action] = new_q
    
    q_learner.learning_iterations += 1
    # Decay exploration gradually
    if q_learner.learning_iterations % 25 == 0:
        q_learner.exploration_rate *= EXPLORATION_DECAY
        q_learner.exploration_rate = max(MIN_EXPLORATION, q_learner.exploration_rate)

def hybrid_balancer(servers: List[SimulatedServer], last_rr_index: int, log_callback: Callable, num_servers: int) -> Tuple[Optional[SimulatedServer], int, str]:
    """Hybrid balancer designed to outperform individual algorithms."""
    time.sleep(ALGORITHM_DELAY)  # Enforce same delay as other algorithms
    
    if not servers:
        return None, last_rr_index, "FAIL"
    
    current_state = discretize_state(servers, num_servers)
    system_utilization = np.mean([s.get_utilization() for s in servers]) if servers else 0
    
    # Store previous state for reward calculation
    previous_server_states = {s.id: s.get_utilization() for s in servers}
    
    # Epsilon-greedy exploration
    if random.random() < q_learner.exploration_rate:
        # Random exploration
        action = random.choice([0, 1])
        algorithm_choice = "ACO" if action == 1 else "RR"
        log_callback(f"Hybrid: Exploration - {algorithm_choice}")
    else:
        # Exploitation - choose best action
        q_values = q_learner.q_table[current_state[0], current_state[1], :]
        action = np.argmax(q_values)
        algorithm_choice = "ACO" if action == 1 else "RR"
        confidence = abs(q_values[0] - q_values[1]) / (max(abs(q_values[0]), abs(q_values[1])) + 0.1)
        log_callback(f"Hybrid: Exploitation - {algorithm_choice} (Conf: {confidence:.2f}, Q-Values: RR={q_values[0]:.1f}, ACO={q_values[1]:.1f})")
    
    # Execute chosen algorithm
    chosen_server = None
    if algorithm_choice == "RR":
        chosen_server, next_rr_index = round_robin_balancer(servers, last_rr_index, log_callback)
        last_rr_index = next_rr_index
    else:
        chosen_server = aco_balancer(servers, log_callback)
    
    if chosen_server:
        # Calculate reward
        previous_util = previous_server_states.get(chosen_server.id, 0)
        reward = calculate_reward(chosen_server, previous_util, algorithm_choice, system_utilization)
        
        # Get next state
        next_state = discretize_state(servers, num_servers)
        
        # Update Q-table
        update_q_value(current_state, action, reward, next_state)
        
        log_callback(f"Hybrid: {algorithm_choice} → {chosen_server.id} | REWARD: {reward:.2f} | Expl: {q_learner.exploration_rate:.3f}")
        return chosen_server, last_rr_index, algorithm_choice
    
    # Fallback
    log_callback("Hybrid: Fallback to RR")
    chosen_server, next_rr_index = round_robin_balancer(servers, last_rr_index, log_callback)
    return chosen_server, next_rr_index, "RR"

def get_q_table_status():
    return q_learner.q_table.copy()

def get_exploration_rate():
    return q_learner.exploration_rate

def get_learning_progress():
    return {
        'iterations': q_learner.learning_iterations,
        'exploration_rate': q_learner.exploration_rate,
        'q_table_variance': np.var(q_learner.q_table)
    }

def reset_q_learning():
    global q_learner
    q_learner = QLearningBalancer()

# def q_table_display():
#     """Prints the current Q-table and exploration rate in a readable format."""
#     q = get_q_table_status()
#     exp = get_exploration_rate()
#     ts = time.strftime("%Y-%m-%d %H:%M:%S")

#     print("\n" + "="*60)
#     print(f"Q-TABLE STATUS at {ts}")
#     print(f"Exploration Rate: {exp:.4f}")
#     print("="*60)

#     load_map = {0: "Low", 1: "Medium", 2: "High"}
#     balance_map = {0: "Poor", 1: "Good", 2: "Excellent"}

#     for load_idx, load_name in load_map.items():
#         print(f"\n--- Load State: {load_name} ---")
#         print(f"{'Balance State':<15} | {'Q(RR)':<12} | {'Q(ACO)':<12} | {'Preference'}")
#         print("-"*60)
#         for bal_idx, bal_name in balance_map.items():
#             rr_val = q[load_idx, bal_idx, 0]
#             aco_val = q[load_idx, bal_idx, 1]
            
#             if rr_val > aco_val:
#                 preference = "RR"
#             elif aco_val > rr_val:
#                 preference = "ACO"
#             else:
#                 preference = "None"

#             print(f"{bal_name:<15} | {rr_val:<12.4f} | {aco_val:<12.4f} | {preference}")
#     print("="*60 + "\n") 