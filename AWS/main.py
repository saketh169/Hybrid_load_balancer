import simpy
import random
import time
import os
import threading
import json
from typing import List, Optional
import csv
from datetime import datetime

from server import initialize_servers, SimulatedServer
from algorithms import round_robin_balancer, aco_balancer, initialize_pheromones, update_pheromones
from hybrid import hybrid_balancer, reset_q_learning, get_q_table_status, get_exploration_rate
from workload_generator import workload_generator
from visualization import log_metrics, generate_simulation_id, current_simulation_id, analyze_and_plot_cli

# Define the log file path for structured JSON data
current_dir = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(current_dir, "q_table_log.jsonl")

def log_message(message: str):
    """Simple console logging."""
    print(message)

def log_q_table_update():
    """Log the Q-table whenever it is updated."""
    try:
        q_table = get_q_table_status()
        exploration_rate = get_exploration_rate()
        
        # Maps for creating the descriptive JSON structure
        load_map = {0: "Low Load", 1: "Medium Load", 2: "High Load"}
        balance_map = {0: "Poor Balance", 1: "Good Balance", 2: "Excellent Balance"}
        
        # Manually build the q_table_view string for custom formatting
        q_table_lines = []
        for load_idx, load_name in load_map.items():
            balance_parts = []
            for bal_idx, bal_name in balance_map.items():
                rr_val = round(q_table[load_idx, bal_idx, 0], 4)
                aco_val = round(q_table[load_idx, bal_idx, 1], 4)
                # Use compact JSON for the innermost object
                balance_json = json.dumps({
                    "Q(RR)": rr_val, "Q(ACO)": aco_val
                })
                balance_parts.append(f'"{bal_name}": {balance_json}')
            
            # Join all balance states for the current load state into a single line
            q_table_lines.append(f'    "{load_name}": {{ {", ".join(balance_parts)} }}')

        # Assemble the final log string
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        exp_rate_str = f'{round(exploration_rate, 4)}'
        q_table_view_str = ",\n".join(q_table_lines)

        log_string = (
            "{\n"
            f'  "timestamp": "{timestamp}",\n'
            f'  "exploration_rate": {exp_rate_str},\n'
            '  "q_table_view": {\n'
            f"{q_table_view_str}\n"
            "  }\n"
            "}\n"
        )

        # Append the custom-formatted string to the file
        with open(LOG_FILE, "a") as f:
            f.write(log_string)
            f.write("---\n")  # Add a separator between entries

    except Exception as e:
        print(f"Error logging Q-table: {e}")

def clear_previous_metrics():
    """Clear all previous metrics, graphs, and log files."""
    # Ensure simulation_results directory exists
    if not os.path.exists('simulation_results'):
        os.makedirs('simulation_results')
        
    # Clear metrics.csv
    metrics_file = os.path.join('simulation_results', 'all_metrics_None.csv')
    if os.path.exists(metrics_file):
        os.remove(metrics_file)
    
    # Clear summary file
    summary_file = os.path.join('simulation_results', 'summary_none.txt')
    if os.path.exists(summary_file):
        os.remove(summary_file)
    
    # DO NOT clear Q-table log here - it's handled separately in run_simulation()
        
    print("\n🧹 Cleared previous metrics and logs")

def run_simulation(num_servers: int, num_tasks: int, algorithm: str, generate_new_workload: bool):
    """Run the load balancer simulation with the specified parameters."""
    
    # Store these values as last run parameters in a file to track changes
    try:
        with open('last_run_params.txt', 'r') as f:
            last_params = f.read().split(',')
            last_servers, last_tasks = map(int, last_params)
    except:
        last_servers, last_tasks = None, None
    
    # Only clear metrics if servers or tasks count changed, or if generating new workload
    if (last_servers != num_servers or last_tasks != num_tasks or generate_new_workload):
        clear_previous_metrics()
        # Save new parameters
        with open('last_run_params.txt', 'w') as f:
            f.write(f"{num_servers},{num_tasks}")
    
    # Clear Q-table log if running Hybrid algorithm
    if algorithm == "Hybrid":
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
        log_message("🧹 Cleared Q-table log for Hybrid simulation")
    
    # Validate inputs
    if not (20 <= num_servers <= 50):
        raise ValueError("Number of servers must be between 20 and 50")
    if num_tasks <= 0:
        raise ValueError("Number of tasks must be positive")

    # Generate or use existing workload
    if generate_new_workload or not workload_generator.has_workload():
        workload = workload_generator.generate_workload(num_tasks)
        log_message(f"📊 New workload generated: {len(workload)} tasks")
        log_message(f"📈 Load range: {min(workload):.1f} - {max(workload):.1f}")
    else:
        workload = workload_generator.get_current_workload()
        if len(workload) != num_tasks:
            log_message("Workload size mismatch. Generating new workload.")
            workload = workload_generator.generate_workload(num_tasks)
    
    # Initialize simulation
    env = simpy.Environment()
    servers = initialize_servers(env, num_servers)
    initialize_pheromones(servers)
    
    # Generate simulation ID and set it as current
    global current_simulation_id
    current_simulation_id = generate_simulation_id()
    
    log_message(f"\n🚀 Starting {algorithm} simulation")
    log_message(f"📊 Simulation ID: {current_simulation_id}")
    log_message(f"🖥️  {num_servers} servers, {num_tasks} tasks")
    
    # Initialize tracking variables
    last_rr_index = -1
    task_count = 0
    current_workload_index = 0

    # Main simulation loop
    while task_count < num_tasks and current_workload_index < len(workload):
        # Get next task from workload
        task_load = workload[current_workload_index]
        current_workload_index += 1
        task_count += 1
        
        # Select server based on algorithm
        chosen_server = None
        strategy_used = algorithm
        
        if algorithm == "RoundRobin":
            chosen_server, last_rr_index = round_robin_balancer(
                servers, last_rr_index, log_message)
            strategy_used = "RR"
        elif algorithm == "ACO":
            chosen_server = aco_balancer(servers, log_message)
            strategy_used = "ACO"
        elif algorithm == "Hybrid":
            chosen_server, last_rr_index, strategy_used = hybrid_balancer(
                servers, last_rr_index, log_message, num_servers)
            # Log Q-table only for Hybrid algorithm
            log_q_table_update()
        
        if chosen_server:
            # Add task to server and run simulation step
            chosen_server.add_task_load(task_load, strategy_used)
            env.run(until=env.now + 1)
            log_metrics(algorithm, chosen_server, task_load)
            
            # Simulate task completion
            completion_rate = random.uniform(0.3, 0.8)
            chosen_server.reduce_load(task_load * completion_rate)
            
            # Update ACO pheromones if applicable
            if strategy_used in ["ACO", "Hybrid"]:
                update_pheromones(servers, chosen_server.id, log_message)
        
        # Print progress every 10%
        if task_count % (num_tasks // 10) == 0:
            print(f"Progress: {(task_count/num_tasks)*100:.1f}% complete")
            print_server_stats(servers)

    # Final analysis and save metrics
    log_message("\n✅ Simulation complete!")
    analyze_and_plot_cli(algorithm, num_tasks, num_servers)  # Pass both input task count and server count

def print_server_stats(servers: List[SimulatedServer]):
    """Print current server statistics."""
    utilizations = [s.get_utilization() for s in servers]
    response_times = [s.get_response_time_metric() for s in servers]
    
    avg_util = sum(utilizations) / len(utilizations)
    avg_response = sum(response_times) / len(response_times)
    max_util = max(utilizations)
    min_util = min(utilizations)
    active_servers = len([s for s in servers if s.current_load > 0])
    
    print(f"\nCurrent Statistics:")
    print(f"Average Utilization: {avg_util:.1f}%")
    print(f"Utilization Range: {min_util:.1f}% - {max_util:.1f}%")
    print(f"Average Response Time: {avg_response:.2f}")
    print(f"Active Servers: {active_servers}")

def main():
    """Main function to run the simulation with command line inputs."""
    print("Welcome to Load Balancer Simulator")
    print("----------------------------------")
    
    # Initialize AWS services
    from aws_integrations import initialize_aws_services, create_dynamodb_table
    dynamodb, cloudwatch = initialize_aws_services()
    if dynamodb:
        table = create_dynamodb_table(dynamodb)
        if not table:
            print("⚠️ Warning: DynamoDB table creation failed. Will continue without AWS integration.")
    
    algorithm_map = {
        "1": "RoundRobin",
        "2": "ACO",
        "3": "Hybrid"
    }

    while True:
        try:
            print("\n=== New Simulation ===")
            # Get user inputs
            num_servers = int(input("Enter number of servers (20-50): "))
            num_tasks = int(input("Enter number of tasks: "))
            
            print("\nAvailable algorithms:")
            print("1. RoundRobin")
            print("2. ACO")
            print("3. Hybrid")
            algo_choice = input("Choose algorithm (1-3): ")
            
            if algo_choice not in algorithm_map:
                raise ValueError("Invalid algorithm choice")
            
            algorithm = algorithm_map[algo_choice]
            
            # Set the current algorithm in visualization module
            from visualization import set_current_algorithm
            set_current_algorithm(algorithm)
            
            generate_workload = input("Generate new workload? (y/n): ").lower() == 'y'
            
            # Run simulation
            run_simulation(num_servers, num_tasks, algorithm, generate_workload)
            
            # Ask if user wants to run another simulation
            run_again = input("\nRun another simulation? (y/n): ").lower()
            if run_again != 'y':
                print("\nThank you for using Load Balancer Simulator!")
                break

        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            print("Let's try again...")
        
        print("-" * 50)

if __name__ == "__main__":
    main()