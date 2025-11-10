import json
import time
import os
import csv
try:
    from hybrid import get_q_table_status, get_exploration_rate
except ImportError as e:
    exit(1)

# Define the log file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(current_dir, "q_table_log.jsonl")
SIMULATION_RESULTS_DIR = os.path.join(current_dir, "simulation_results")
CSV_OUTPUT_FILE = os.path.join(SIMULATION_RESULTS_DIR, "q_table_log.csv")

def log_q_table_update():
    """Log the Q-table whenever it is updated."""
    try:
        q_table = get_q_table_status()
        exploration_rate = get_exploration_rate()
        
        load_map = {0: "Low Load", 1: "Medium Load", 2: "High Load"}
        balance_map = {0: "Poor Balance", 1: "Good Balance", 2: "Excellent Balance"}
        
        q_table_lines = []
        for load_idx, load_name in load_map.items():
            balance_parts = []
            for bal_idx, bal_name in balance_map.items():
                rr_val = round(q_table[load_idx, bal_idx, 0], 4)
                aco_val = round(q_table[load_idx, bal_idx, 1], 4)
                balance_json = json.dumps({
                    "Q(RR)": rr_val, "Q(ACO)": aco_val
                })
                balance_parts.append(f'"{bal_name}": {balance_json}')
            
            q_table_lines.append(f'    "{load_name}": {{ {", ".join(balance_parts)} }}')

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

        with open(LOG_FILE, "a") as f:
            f.write(log_string)
            f.write("---\n")

    except Exception:
        pass

def convert_jsonl_to_csv():
    """Convert all entries from q_table_log.jsonl to a CSV file with only Good Balance and Excellent Balance data in simulation_results, deleting and recreating if it exists."""
    try:
        # Check if JSONL file exists
        if not os.path.exists(LOG_FILE):
            return

        # Check if JSONL file is empty
        if os.path.getsize(LOG_FILE) == 0:
            return

        # Ensure simulation_results directory exists
        if not os.path.exists(SIMULATION_RESULTS_DIR):
            os.makedirs(SIMULATION_RESULTS_DIR)

        # Delete existing CSV file if it exists
        if os.path.exists(CSV_OUTPUT_FILE):
            try:
                os.remove(CSV_OUTPUT_FILE)
            except OSError:
                return

        # Define CSV headers - only Good Balance and Excellent Balance columns
        headers = [
            "Timestamp", "Exploration_Rate",
            "Low_Load_Good_Balance_Q(RR)", "Low_Load_Good_Balance_Q(ACO)",
            "Low_Load_Excellent_Balance_Q(RR)", "Low_Load_Excellent_Balance_Q(ACO)",
            "Medium_Load_Good_Balance_Q(RR)", "Medium_Load_Good_Balance_Q(ACO)",
            "Medium_Load_Excellent_Balance_Q(RR)", "Medium_Load_Excellent_Balance_Q(ACO)",
            "High_Load_Good_Balance_Q(RR)", "High_Load_Good_Balance_Q(ACO)",
            "High_Load_Excellent_Balance_Q(RR)", "High_Load_Excellent_Balance_Q(ACO)"
        ]

        # List to store all rows
        rows = []
        entry_count = 0

        # Read the JSONL file
        with open(LOG_FILE, "r") as f:
            current_entry = []
            for line in f:
                line = line.strip()
                if line == "---":
                    if current_entry:
                        try:
                            json_str = "".join(current_entry)
                            entry = json.loads(json_str)
                            entry_count += 1
                            row = {
                                "Timestamp": entry["timestamp"],
                                "Exploration_Rate": float(entry["exploration_rate"])
                            }
                            for load_state in ["Low Load", "Medium Load", "High Load"]:
                                for balance_state in ["Good Balance", "Excellent Balance"]:
                                    q_values = entry["q_table_view"][load_state][balance_state]
                                    row[f"{load_state.replace(' ', '_')}_{balance_state.replace(' ', '_')}_Q(RR)"] = q_values["Q(RR)"]
                                    row[f"{load_state.replace(' ', '_')}_{balance_state.replace(' ', '_')}_Q(ACO)"] = q_values["Q(ACO)"]
                            rows.append(row)
                        except json.JSONDecodeError:
                            pass
                        current_entry = []
                else:
                    current_entry.append(line)

        # Check if any valid rows were parsed
        if not rows:
            return

        # Write to new CSV file
        with open(CSV_OUTPUT_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        print(f"all entries $ {entry_count} entries are saved to q_table_log.csv")

    except Exception:
        pass

def main():
    """Main function to run the conversion when the script is executed."""
    convert_jsonl_to_csv()

if __name__ == "__main__":
    main()