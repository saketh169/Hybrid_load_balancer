import sys
import os
import threading
import time
import json

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

# Moved imports to the top level to prevent threading deadlocks
from hybrid import get_q_table_status, get_exploration_rate
from ui import run_ui

# Define the log file path for structured JSON data
LOG_FILE = os.path.join(current_dir, "q_table_log.jsonl")

def _q_table_logger(interval: float):
    """A function that logs the Q-table in a semi-compact, readable JSON format."""
    while True:
        time.sleep(interval)
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
                f.write("---\n") # Add a separator between entries

            # Console display is now commented out as requested.
            # print(q_table_display())
                
        except Exception as e:
            print(f"Error logging Q-table: {e}")

def main():
    """Main entry point."""
    try:
        # Clear the log file at the start of a new session
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)

        # Start the background thread to log the Q-table
        q_table_thread = threading.Thread(target=_q_table_logger, args=(5.0,), daemon=True)
        q_table_thread.start()
        
        print("🚀 Starting Load Balancer Simulation...")
        print("📊 Algorithms: Round Robin, ACO, Hybrid (Q-Learning)")
        print("💡 Use the UI to configure and run simulations")
        print("📈 Workload Generator: Generate once, test all algorithms with same load!")
        print(f"ℹ️  Q-table will be logged every 5 seconds to {LOG_FILE}")
        
        run_ui()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()