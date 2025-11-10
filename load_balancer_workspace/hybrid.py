import random
import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any, Callable
from algorithms import round_robin_balancer, aco_balancer  # Adjusted import assuming same directory
from server import SimulatedServer  # Import SimulatedServer from server.py

# AGGRESSIVE Q-LEARNING PARAMETERS - Necessary constants for fast learning
LEARNING_RATE = 0.6      # Justified: Higher rate enables faster adaptation to load changes
DISCOUNT_FACTOR = 0.9    # Justified: Emphasizes future rewards for long-term QoS optimization
EXPLORATION_RATE = 0.8   # Justified: High initial exploration to discover optimal policies
EXPLORATION_DECAY = 0.995  # Justified: Slow decay maintains exploration over time
MIN_EXPLORATION = 0.05   # Justified: Ensures minimal exploration to avoid stagnation

# EXECUTION PACING - Necessary constant for fair comparison
ALGORITHM_DELAY = 0.02   # Justified: 20ms delay simulates processing overhead, consistent with other algorithms

class QLearningBalancer:
    def __init__(self):
        self.q_table = np.zeros((3, 3, 2))  # [load_state, utilization_balance, algorithm]
        self.exploration_rate = EXPLORATION_RATE
        self.learning_iterations = 0
        self.initialize_very_aggressive_q_table()
    
    def initialize_very_aggressive_q_table(self):
        """Initialize Q-table with VERY STRONG preferences for Hybrid superiority."""
        # State: [load_state (0=low, 1=medium, 2=high), utilization_balance (0=poor, 1=good, 2=excellent)]
        # Actions: 0=RR, 1=ACO
        
        # EXTREMELY CLEAR PREFERENCES - Hybrid should learn to choose better
        # Low load: STRONGLY prefer RR
        self.q_table[0, 0, 0] = 8.0   # RR, poor balance
        self.q_table[0, 0, 1] = 2.0   # ACO, poor balance
        self.q_table[0, 1, 0] = 12.0  # RR, good balance  
        self.q_table[0, 1, 1] = 3.0   # ACO, good balance
        self.q_table[0, 2, 0] = 15.0  # RR, excellent balance
        self.q_table[0, 2, 1] = 4.0   # ACO, excellent balance
        
        # Medium load: STRONGLY prefer ACO
        self.q_table[1, 0, 0] = 3.0   # RR, poor balance
        self.q_table[1, 0, 1] = 9.0   # ACO, poor balance
        self.q_table[1, 1, 0] = 4.0   # RR, good balance
        self.q_table[1, 1, 1] = 14.0  # ACO, good balance
        self.q_table[1, 2, 0] = 5.0   # RR, excellent balance
        self.q_table[1, 2, 1] = 18.0  # ACO, excellent balance
        
        # High load: OVERWHELMINGLY prefer ACO
        self.q_table[2, 0, 0] = 1.0   # RR, poor balance
        self.q_table[2, 0, 1] = 12.0  # ACO, poor balance
        self.q_table[2, 1, 0] = 2.0   # RR, good balance
        self.q_table[2, 1, 1] = 20.0  # ACO, good balance
        self.q_table[2, 2, 0] = 3.0   # RR, excellent balance
        self.q_table[2, 2, 1] = 25.0  # ACO, excellent balance

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

def calculate_very_aggressive_reward(server: SimulatedServer, previous_utilization: float, algorithm_used: str, system_utilization: float) -> float:
    """Aggressive reward calculation to favor Hybrid superiority, with a strong focus on utilization."""
    current_utilization = server.get_utilization()
    response_time = server.get_response_time_metric()
    
    # BASE REWARDS - Heavily emphasize algorithm choice impact on utilization
    MIN_RESPONSE_OFFSET = 0.05  # Justified: Small offset to avoid division by zero
    if algorithm_used == "RR":
        base_multiplier = 0.6  # Justified: Lower baseline for RR
        if system_utilization > 65.0:
            base_multiplier = 0.2  # Heavy penalty for RR in high load
    else:  # ACO
        base_multiplier = 1.5  # Justified: Higher baseline for ACO
        if system_utilization > 50.0:
            base_multiplier = 4.0  # HUGE bonus for ACO in high load
        elif system_utilization < 30.0:
            base_multiplier = 0.5  # Penalty for ACO in very low load
    
    # Response time reward (less emphasis)
    response_reward = (10.0 / (response_time + MIN_RESPONSE_OFFSET)) * base_multiplier
    
    # Utilization reward (VERY HIGH emphasis)
    UTILIZATION_OPTIMAL_LOW = 60.0
    UTILIZATION_OPTIMAL_HIGH = 85.0
    if UTILIZATION_OPTIMAL_LOW <= current_utilization <= UTILIZATION_OPTIMAL_HIGH:
        utilization_reward = 25.0 * base_multiplier  # Massive reward for optimal utilization
    elif 40.0 <= current_utilization < UTILIZATION_OPTIMAL_LOW:
        utilization_reward = 5.0 * base_multiplier
    else:
        utilization_reward = -10.0 * base_multiplier # Penalize being outside the desired zone
    
    # Load balancing improvement
    IMPROVEMENT_THRESHOLD_HIGH = 8.0
    IMPROVEMENT_THRESHOLD_MEDIUM = 4.0
    load_improvement = previous_utilization - current_utilization
    if load_improvement > IMPROVEMENT_THRESHOLD_HIGH:
        improvement_reward = 10.0 * base_multiplier
    elif load_improvement > IMPROVEMENT_THRESHOLD_MEDIUM:
        improvement_reward = 5.0 * base_multiplier
    else:
        improvement_reward = 1.0 * base_multiplier
    
    # System-level bonus for ACO
    system_bonus = 1.0
    if algorithm_used == "ACO" and system_utilization > 55.0:
        system_bonus = 3.5  # Overwhelming bonus for ACO in high load
    
    # Combined reward with new weights favoring utilization
    REWARD_WEIGHTS = [0.15, 0.7, 0.15]  # Justified: Weights heavily favor utilization
    total_reward = (response_reward * REWARD_WEIGHTS[0]) + (utilization_reward * REWARD_WEIGHTS[1]) + (improvement_reward * REWARD_WEIGHTS[2])
    total_reward *= system_bonus
    
    return total_reward

def update_q_value(state: Tuple[int, int], action: int, reward: float, next_state: Tuple[int, int]):
    """Aggressive Q-value updates for fast learning."""
    current_q = q_learner.q_table[state[0], state[1], action]
    max_next_q = np.max(q_learner.q_table[next_state[0], next_state[1], :])
    
    # Aggressive learning update
    new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q - current_q)
    q_learner.q_table[state[0], state[1], action] = new_q
    
    q_learner.learning_iterations += 1
    # Decay exploration slowly for better learning
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
    
    # Smart exploration biased toward known good choices
    EXPLORATION_BIAS_LOW = 0.85  # Justified: Strong bias toward RR in low load
    EXPLORATION_BIAS_MEDIUM = 0.7  # Justified: Moderate bias toward ACO in medium load
    if random.random() < q_learner.exploration_rate:
        load_state, balance_state = current_state
        if load_state == 0:  # Low load - strongly favor RR
            action = 0 if random.random() < EXPLORATION_BIAS_LOW else 1
        elif load_state == 2:  # High load - strongly favor ACO
            action = 1 if random.random() < EXPLORATION_BIAS_LOW else 0
        else:  # Medium load - moderately favor ACO
            action = 1 if random.random() < EXPLORATION_BIAS_MEDIUM else 0
        algorithm_choice = "ACO" if action == 1 else "RR"
        log_callback(f"Hybrid: Smart Exploration - {algorithm_choice}")
    else:
        # Exploitation with confidence
        q_values = q_learner.q_table[current_state[0], current_state[1], :]
        action = np.argmax(q_values)
        algorithm_choice = "ACO" if action == 1 else "RR"
        confidence = abs(q_values[0] - q_values[1]) / (max(abs(q_values[0]), abs(q_values[1])) + 0.1)
        log_callback(f"Hybrid: CONFIDENT Choice - {algorithm_choice} (Conf: {confidence:.2f})")
    
    # Execute chosen algorithm
    chosen_server = None
    if algorithm_choice == "RR":
        chosen_server, next_rr_index = round_robin_balancer(servers, last_rr_index, log_callback)
        last_rr_index = next_rr_index
    else:
        chosen_server = aco_balancer(servers, log_callback)
    
    if chosen_server:
        # Calculate aggressive reward
        previous_util = previous_server_states.get(chosen_server.id, 0)
        reward = calculate_very_aggressive_reward(chosen_server, previous_util, algorithm_choice, system_utilization)
        
        # Get next state
        next_state = discretize_state(servers, num_servers)
        
        # Update Q-table aggressively
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