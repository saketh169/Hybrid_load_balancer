import random
import simpy
from typing import Optional, List

class SimulatedServer:
    """Enhanced server with SimPy integration for discrete event simulation."""
    
    def __init__(self, env: simpy.Environment, server_id: str, capacity_units: int = 100):
        self.env = env
        self.id = server_id
        self.max_capacity = capacity_units
        self.current_load = 0
        self.busy_time = 0
        self.total_tasks_processed = 0
        self.rr_usage = 0
        self.aco_usage = 0
        self.last_task_load = 0
        self.last_algorithm_used: Optional[str] = None
        self.zero_load_cycles = 0
        self.response_time_accumulator = 0
        self.overload_penalty_count = 0
        self.log_message: Optional[str] = None  # For logging overload events
        # SimPy resource for managing concurrent tasks
        self.resource = simpy.Resource(env, capacity=1)

    def add_task_load(self, task_load: float, algorithm_used: str):
        """Add task load as a SimPy process."""
        self.current_load += task_load
        self.last_task_load = task_load
        self.last_algorithm_used = algorithm_used
        
        # Calculate response time based on algorithm and load
        utilization = self.current_load / self.max_capacity
        base_processing = 0.5
        
        # Algorithm-specific modifiers
        if algorithm_used == "RR":
            load_penalty = utilization * 2.0
            algorithm_efficiency = 1.0
        elif algorithm_used == "ACO": 
            load_penalty = utilization * 1.3
            algorithm_efficiency = 0.85
        else:  # Hybrid
            load_penalty = utilization * 1.1
            algorithm_efficiency = 0.75
        
        # Apply overload penalty above 90% utilization
        overload_penalty = (utilization - 0.9) * 3.0 if utilization > 0.9 else 0
        if utilization > 0.9:
            self.overload_penalty_count += 1
        
        random_variation = random.uniform(0.9, 1.1)
        response_time = (base_processing + load_penalty + overload_penalty) * random_variation * algorithm_efficiency
        
        # Start the task processing process
        self.env.process(self._process_task(response_time))
        
        self.response_time_accumulator += response_time
        self.busy_time += response_time
        self.total_tasks_processed += 1
        
        if algorithm_used == "RR":
            self.rr_usage += 1
        elif algorithm_used == "ACO":
            self.aco_usage += 1
            
        self.zero_load_cycles = 0
        
        # Free server if utilization reaches 100%
        if self.get_utilization() >= 100.0:
            self.current_load = 0
            self.log_message = f"Server {self.id} freed due to 100% utilization."

    def reduce_load(self, task_load: float):
        """Reduce server load."""
        self.current_load = max(0, self.current_load - task_load)
        if self.current_load == 0:
            self.zero_load_cycles += 1
        else:
            self.zero_load_cycles = 0

    def get_utilization(self) -> float:
        return min(100.0, (self.current_load / self.max_capacity) * 100)

    def get_response_time_metric(self) -> float:
        if self.total_tasks_processed == 0:
            return 1.0
        return self.response_time_accumulator / self.total_tasks_processed

    def is_idle(self) -> bool:
        return self.current_load == 0 and self.total_tasks_processed > 0 and self.zero_load_cycles >= 3

    def _process_task(self, response_time):
        """SimPy process for task execution."""
        with self.resource.request() as request:
            yield request
            yield self.env.timeout(response_time)

    def get_performance_score(self) -> float:
        """Enhanced performance score with overload penalties."""
        utilization = self.get_utilization() / 100.0
        response_time = self.get_response_time_metric()
        
        overload_penalty = 1.0
        if utilization > 0.9:
            overload_penalty = 0.5  # Heavy penalty for overload
        elif utilization > 0.8:
            overload_penalty = 0.8  # Moderate penalty for high utilization
        
        score = (1.0 / (response_time + 0.1)) * (1.0 - abs(0.7 - utilization)) * overload_penalty
        return max(0.1, score)


def initialize_servers(env: simpy.Environment, num_servers: int) -> List[SimulatedServer]:
    if not (20 <= num_servers <= 50):
        raise ValueError("Number of servers must be between 20 and 50")
    print(f"--- INITIALIZING: {num_servers} Servers for this run ---")
    return [SimulatedServer(env, f"Server-{i+1}") for i in range(num_servers)]