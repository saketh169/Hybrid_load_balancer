import tkinter as tk
from tkinter import ttk, messagebox
import random
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import simpy

from server import initialize_servers
from algorithms import round_robin_balancer, aco_balancer, initialize_pheromones, update_pheromones
from hybrid import hybrid_balancer, reset_q_learning, get_q_table_status, get_exploration_rate
from visualization import log_metrics, visualize_assignment_step, on_pick, analyze_and_plot, generate_simulation_id, current_simulation_id
from workload_generator import workload_generator

class LoadBalancerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Load Balancer Simulation")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')
        
        # Simulation state
        self.simulation_running = False
        self.simulation_paused = False
        self.task_count = 0
        self.servers = []
        self.num_tasks = 0
        self.num_servers = 0
        self.approach_name = ""
        self.env = None
        self.last_rr_index = -1
        self.last_update_time = 0
        self.current_workload_index = 0  # Track current position in workload
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface."""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel - Controls and info
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        
        # Right panel - Visualizations
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side='right', fill='both', expand=True)
        
        self.setup_controls(left_panel)
        self.setup_visualizations(right_panel)
    
    def setup_controls(self, parent):
        """Setup control panel with workload generator."""
        # Title
        title = ttk.Label(parent, text="Load Balancer Simulator", 
                         font=('Arial', 16, 'bold'), foreground='#34495e')
        title.pack(pady=(0, 20))
        
        # Input frame
        input_frame = ttk.LabelFrame(parent, text="Simulation Parameters", padding=10)
        input_frame.pack(fill='x', pady=(0, 10))
        
        # Server input
        ttk.Label(input_frame, text="Number of Servers (20-50):").grid(row=0, column=0, sticky='w', pady=2)
        self.servers_entry = ttk.Entry(input_frame, width=15)
        self.servers_entry.grid(row=0, column=1, sticky='w', pady=2)
        self.servers_entry.insert(0, "25")
        
        # Tasks input
        ttk.Label(input_frame, text="Number of Tasks:").grid(row=1, column=0, sticky='w', pady=2)
        self.tasks_entry = ttk.Entry(input_frame, width=15)
        self.tasks_entry.grid(row=1, column=1, sticky='w', pady=2)
        self.tasks_entry.insert(0, "500")
        
        # Algorithm selection
        ttk.Label(input_frame, text="Algorithm:").grid(row=2, column=0, sticky='w', pady=2)
        self.algorithm_var = tk.StringVar(value="Hybrid")
        algorithm_combo = ttk.Combobox(input_frame, textvariable=self.algorithm_var,
                                      values=["RoundRobin", "ACO", "Hybrid"], state="readonly", width=12)
        algorithm_combo.grid(row=2, column=1, sticky='w', pady=2)
        
        # WORKLOAD GENERATOR SECTION
        workload_frame = ttk.LabelFrame(input_frame, text="Workload Generator", padding=5)
        workload_frame.grid(row=4, column=0, columnspan=2, sticky='we', pady=10)
        
        # Workload controls
        ttk.Button(workload_frame, text="Generate New Workload", 
                  command=self.generate_workload).pack(side='left', padx=2, pady=2)
        
        ttk.Button(workload_frame, text="Use Same Workload", 
                  command=self.use_same_workload).pack(side='left', padx=2, pady=2)
        
        self.workload_status = ttk.Label(workload_frame, text="No workload generated", 
                                        foreground='red', font=('Arial', 8))
        self.workload_status.pack(side='left', padx=10, pady=2)
        
        # Control buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=10)
        
        self.start_btn = ttk.Button(button_frame, text="Start Simulation", command=self.start_simulation)
        self.start_btn.pack(side='left', padx=2)
        
        self.pause_btn = ttk.Button(button_frame, text="Pause", command=self.pause_simulation, state='disabled')
        self.pause_btn.pack(side='left', padx=2)
        
        self.resume_btn = ttk.Button(button_frame, text="Resume", command=self.resume_simulation, state='disabled')
        self.resume_btn.pack(side='left', padx=2)
        
        # Clear Table button
        self.clear_btn = ttk.Button(button_frame, text="Clear Table", command=self.clear_table)
        self.clear_btn.pack(side='left', padx=2)
        
        # Stats frame
        stats_frame = ttk.LabelFrame(parent, text="Live Statistics", padding=10)
        stats_frame.pack(fill='x', pady=10)
        
        self.stats_text = tk.Text(stats_frame, height=8, width=30, font=('Consolas', 9))
        self.stats_text.pack(fill='both')
        
        # Log frame
        log_frame = ttk.LabelFrame(parent, text="Simulation Log", padding=10)
        log_frame.pack(fill='both', expand=True, pady=10)
        
        self.log_text = tk.Text(log_frame, height=15, width=30, font=('Consolas', 8))
        scrollbar = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
    
    def setup_visualizations(self, parent):
        """Setup visualization panels."""
        # Create notebook for tabs
        notebook = ttk.Notebook(parent)
        notebook.pack(fill='both', expand=True)
        
        # Real-time tab
        realtime_frame = ttk.Frame(notebook)
        notebook.add(realtime_frame, text="Real-time Monitoring")
        
        # Scatter plot
        scatter_frame = ttk.Frame(realtime_frame)
        scatter_frame.pack(fill='both', expand=True, pady=5)
        
        self.scatter_fig = plt.Figure(figsize=(8, 4))
        self.scatter_canvas = FigureCanvasTkAgg(self.scatter_fig, scatter_frame)
        self.scatter_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Bar chart
        bar_frame = ttk.Frame(realtime_frame)
        bar_frame.pack(fill='both', expand=True, pady=5)
        
        self.bar_fig = plt.Figure(figsize=(8, 4))
        self.bar_canvas = FigureCanvasTkAgg(self.bar_fig, bar_frame)
        self.bar_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Results tab
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Results & Analysis")
        
        self.results_fig = plt.Figure(figsize=(8, 6))
        self.results_canvas = FigureCanvasTkAgg(self.results_fig, results_frame)
        self.results_canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        
        # Servers table
        table_frame = ttk.LabelFrame(parent, text="Server Details", padding=5)
        table_frame.pack(fill='x', pady=5)
        
        columns = ("ID", "Load", "Utilization", "Response", "Algorithm", "Tasks")
        self.servers_table = ttk.Treeview(table_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.servers_table.heading(col, text=col)
            self.servers_table.column(col, width=80)
        
        scrollbar_table = ttk.Scrollbar(table_frame, orient='vertical', command=self.servers_table.yview)
        self.servers_table.configure(yscrollcommand=scrollbar_table.set)
        self.servers_table.pack(side='left', fill='both', expand=True)
        scrollbar_table.pack(side='right', fill='y')
        
        # Connect pick event
        self.scatter_canvas.mpl_connect("pick_event", 
                                       lambda event: on_pick(event, self.servers, self.servers_table, self.log_message))
    
    def generate_workload(self):
        """Generate a new workload that will be same for all algorithms."""
        try:
            num_tasks = int(self.tasks_entry.get())
            if num_tasks <= 0:
                messagebox.showerror("Error", "Number of tasks must be positive")
                return
            
            workload = workload_generator.generate_workload(num_tasks)
            self.workload_status.config(text=f"Workload ready: {len(workload)} tasks", foreground='green')
            self.log_message(f"📊 New workload generated: {len(workload)} tasks")
            self.log_message(f"📈 Load range: {min(workload):.1f} - {max(workload):.1f}")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid number of tasks")
    
    def use_same_workload(self):
        """Use the same workload for next simulation."""
        if not workload_generator.has_workload():
            messagebox.showwarning("Warning", "No workload generated yet. Generate workload first.")
            return
        
        workload = workload_generator.get_current_workload()
        self.tasks_entry.delete(0, tk.END)
        self.tasks_entry.insert(0, str(len(workload)))
        self.workload_status.config(text=f"Using same workload: {len(workload)} tasks", foreground='blue')
        self.log_message(f"🔄 Using same workload: {len(workload)} tasks")
    
    def log_message(self, message):
        """Add message to log."""
        self.log_text.insert('end', f"{message}\n")
        self.log_text.see('end')
        self.root.update_idletasks()
    
    def update_stats(self):
        """Update statistics display."""
        if not self.servers:
            return
        
        utilizations = [s.get_utilization() for s in self.servers]
        response_times = [s.get_response_time_metric() for s in self.servers]
        
        avg_util = sum(utilizations) / len(utilizations)
        avg_response = sum(response_times) / len(response_times)
        max_util = max(utilizations)
        min_util = min(utilizations)
        
        stats_text = f"Tasks: {self.task_count}/{self.num_tasks}\n"
        stats_text += f"Servers: {len(self.servers)}\n"
        stats_text += f"Avg Utilization: {avg_util:.1f}%\n"
        stats_text += f"Avg Response: {avg_response:.2f}\n"
        stats_text += f"Utilization Range: {min_util:.1f}%-{max_util:.1f}%\n"
        stats_text += f"Active Servers: {len([s for s in self.servers if s.current_load > 0])}\n"
        
        if self.approach_name == "Hybrid":
            exploration_rate = get_exploration_rate()
            stats_text += f"\nQ-Learning Exploration: {exploration_rate:.3f}"
        
        self.stats_text.delete('1.0', 'end')
        self.stats_text.insert('1.0', stats_text)
    
    def clear_table(self):
        """Clear the servers table and reset display."""
        for item in self.servers_table.get_children():
            self.servers_table.delete(item)
        self.log_text.delete('1.0', 'end')
        self.stats_text.delete('1.0', 'end')
        
        # Clear plots
        self.scatter_fig.clear()
        self.bar_fig.clear()
        self.results_fig.clear()
        self.scatter_canvas.draw()
        self.bar_canvas.draw()
        self.results_canvas.draw()
        
        self.log_message("🗑️ Table cleared. Ready for new simulation.")
    
    def start_simulation(self):
        """Start the simulation."""
        try:
            self.num_servers = int(self.servers_entry.get())
            self.num_tasks = int(self.tasks_entry.get())
            self.approach_name = self.algorithm_var.get()
            
            if not (20 <= self.num_servers <= 50):
                messagebox.showerror("Error", "Number of servers must be between 20 and 50")
                return
            
            if self.num_tasks <= 0:
                messagebox.showerror("Error", "Number of tasks must be positive")
                return
            
            # Check if workload is available
            if not workload_generator.has_workload():
                # Auto-generate workload if not exists
                self.generate_workload()
            
            # Verify workload size matches
            workload = workload_generator.get_current_workload()
            if len(workload) != self.num_tasks:
                messagebox.showwarning("Workload Mismatch", 
                                     f"Workload has {len(workload)} tasks, but simulation needs {self.num_tasks}. Generating new workload.")
                workload = workload_generator.generate_workload(self.num_tasks)
            
            # Generate unique simulation ID
            global current_simulation_id
            current_simulation_id = generate_simulation_id()
            
            # Clear previous data
            self.clear_table()
            
            # Reset simulation state
            self.simulation_running = True
            self.simulation_paused = False
            self.task_count = 0
            self.current_workload_index = 0  # Reset workload index
            self.last_rr_index = -1
            self.last_update_time = time.time()
            
            # Initialize SimPy environment and servers
            self.env = simpy.Environment()
            self.servers = initialize_servers(self.env, self.num_servers)
            initialize_pheromones(self.servers)
            # Q-table is now persistent and should NOT be reset automatically
            # if self.approach_name == "Hybrid":
            #     reset_q_learning()
            
            self.log_message(f"🚀 Starting {self.approach_name} simulation")
            self.log_message(f"📊 Simulation ID: {current_simulation_id}")
            self.log_message(f"🖥️  {self.num_servers} servers, {self.num_tasks} tasks")
            self.log_message(f"📈 Using pre-generated workload")
            self.log_message(f"💾 Results will be saved automatically")
            
            # Enable/disable buttons
            self.start_btn.config(state='disabled')
            self.pause_btn.config(state='normal')
            self.resume_btn.config(state='disabled')
            self.clear_btn.config(state='disabled')
            
            # Start simulation loop
            self.run_simulation_step()
            
        except ValueError as e:
            messagebox.showerror("Error", "Please enter valid numbers")
    
    def pause_simulation(self):
        """Pause the simulation."""
        self.simulation_paused = True
        self.pause_btn.config(state='disabled')
        self.resume_btn.config(state='normal')
        self.log_message("⏸️ Simulation paused")
    
    def resume_simulation(self):
        """Resume the simulation."""
        self.simulation_paused = False
        self.pause_btn.config(state='normal')
        self.resume_btn.config(state='disabled')
        self.log_message("▶️ Simulation resumed")
        self.run_simulation_step()
    
    def run_simulation_step(self):
        """Run one simulation step using pre-generated workload."""
        if not self.simulation_running or self.simulation_paused:
            return
        
        current_time = time.time()
        tasks_processed = 0
        max_tasks_per_cycle = 50
        
        # Get the pre-generated workload
        workload = workload_generator.get_current_workload()
        
        # Process tasks
        while (self.task_count < self.num_tasks and 
               tasks_processed < max_tasks_per_cycle and
               current_time - self.last_update_time < 0.5):
            
            # Use pre-generated task load instead of random
            task_load = workload[self.current_workload_index]
            self.current_workload_index += 1
            self.task_count += 1
            tasks_processed += 1
            
            # Select server based on algorithm
            chosen_server = None
            strategy_used = self.approach_name
            
            if self.approach_name == "RoundRobin":
                chosen_server, self.last_rr_index = round_robin_balancer(
                    self.servers, self.last_rr_index, self.log_message)
                strategy_used = "RR"
            elif self.approach_name == "ACO":
                chosen_server = aco_balancer(self.servers, self.log_message)
                strategy_used = "ACO"
            elif self.approach_name == "Hybrid":
                chosen_server, self.last_rr_index, strategy_used = hybrid_balancer(
                    self.servers, self.last_rr_index, self.log_message, self.num_servers)
            
            if chosen_server:
                # Add task to server and run simulation step
                chosen_server.add_task_load(task_load, strategy_used)
                self.env.run(until=self.env.now + 1)  # Run simulation for one time unit
                log_metrics(self.approach_name, chosen_server, task_load)
                
                # Simulate task completion
                completion_rate = random.uniform(0.3, 0.8)
                chosen_server.reduce_load(task_load * completion_rate)
                
                # Update ACO pheromones if applicable
                if strategy_used in ["ACO", "Hybrid"]:
                    update_pheromones(self.servers, chosen_server.id, self.log_message)
            
            current_time = time.time()
            
            # Break if we've used all workload
            if self.current_workload_index >= len(workload):
                break
        
        # Update visualizations if needed
        if current_time - self.last_update_time >= 0.5 or self.task_count >= self.num_tasks:
            visualize_assignment_step(
                self.servers, self.approach_name, self.task_count, self.num_tasks,
                self.scatter_fig, self.scatter_canvas, self.bar_fig, self.bar_canvas,
                self.servers_table, self.log_message
            )
            self.update_stats()
            self.last_update_time = current_time
        
        # Check if simulation completed
        if self.task_count >= self.num_tasks or self.current_workload_index >= len(workload):
            self.simulation_running = False
            self.log_message("✅ Simulation completed!")
            self.finalize_simulation()
            return
        
        # Schedule next step
        self.root.after(10, self.run_simulation_step)
    
    def finalize_simulation(self):
        """Finalize simulation and show results."""
        self.start_btn.config(state='normal')
        self.pause_btn.config(state='disabled')
        self.resume_btn.config(state='disabled')
        self.clear_btn.config(state='normal')
        
        # Generate final analysis and save all metrics
        analyze_and_plot(self.approach_name, self.results_fig, self.results_canvas, self.log_message)
        
        self.log_message("\n🎯 SIMULATION COMPLETE!")
        self.log_message("📁 All metrics and comparison plots saved in 'simulation_results' folder")
        self.log_message("✅ Run other algorithms to see comprehensive comparison!")

def run_ui():
    """Launch the UI application."""
    root = tk.Tk()
    app = LoadBalancerUI(root)
    root.mainloop()