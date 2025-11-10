import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D
import numpy as np
import os
from datetime import datetime
from typing import List
from CC.load_balancer_workspace.server import SimulatedServer

# Global metrics storage for all algorithms
all_metrics_log = []
current_simulation_id = None

def generate_simulation_id():
    """Generate unique simulation ID with timestamp."""
    return f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def log_metrics(approach, server, task_load):
    """Log comprehensive metrics for analysis."""
    timestamp = time.time()
    metrics_entry = {
        'LogID': f"{timestamp:.6f}".replace('.', ''),
        'Timestamp': timestamp,
        'Approach': approach,
        'ServerID': server.id,
        'ResponseTime': float(server.get_response_time_metric()),
        'Utilization': float(server.get_utilization()),
        'TaskLoad': float(task_load),
        'TotalTasks': server.total_tasks_processed,
        'AlgorithmUsed': server.last_algorithm_used or "None",
        'SimulationID': current_simulation_id
    }
    all_metrics_log.append(metrics_entry)
    return metrics_entry

def save_comprehensive_metrics():
    """Save all metrics to CSV and generate comparison plots."""
    if not all_metrics_log:
        return None
    
    results_dir = "simulation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    df_all = pd.DataFrame(all_metrics_log)
    csv_filename = f"{results_dir}/all_metrics_{current_simulation_id}.csv"
    df_all.to_csv(csv_filename, index=False)
    
    plot_filename = generate_comparison_plots(df_all, results_dir)
    
    return csv_filename, plot_filename

def aggregate_milestone_metrics(servers: List[SimulatedServer], approach: str, simulation_id: str) -> dict:
    """Aggregate milestone metrics (response time and utilization) across all servers for a given approach."""
    milestone_data = {0.25: {'utilization': [], 'response_time': []},
                     0.50: {'utilization': [], 'response_time': []},
                     0.75: {'utilization': [], 'response_time': []},
                     1.00: {'utilization': [], 'response_time': []}}

    for server in servers:
        milestones = server.get_milestone_metrics()
        for milestone in milestones:
            milestone_data[milestone]['utilization'].append(milestones[milestone]['utilization'])
            milestone_data[milestone]['response_time'].append(milestones[milestone]['response_time'])

    result = {}
    for milestone in milestone_data:
        util_values = [v for v in milestone_data[milestone]['utilization'] if v > 0]
        resp_values = [v for v in milestone_data[milestone]['response_time'] if v > 0]
        result[milestone] = {
            'avg_utilization': sum(util_values) / len(util_values) if util_values else 0.0,
            'avg_response_time': sum(resp_values) / len(resp_values) if resp_values else 0.0
        }
    return result

def generate_comparison_plots(df, results_dir):
    """Generate 5 specific plots for analysis."""
    approaches = df['Approach'].unique()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    fig = plt.figure(figsize=(18, 10))
    
    ax1 = plt.subplot(2, 3, 1)
    response_means = [df[df['Approach'] == app]['ResponseTime'].mean() for app in approaches]
    response_stds = [df[df['Approach'] == app]['ResponseTime'].std() for app in approaches]
    
    bars1 = ax1.bar(approaches, response_means, yerr=response_stds, capsize=8, 
                   color=colors[:len(approaches)], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_title('1. Average Response Time\n(Lower is Better)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Response Time (seconds)', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, mean, std in zip(bars1, response_means, response_stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.05,
                f'{mean:.2f}s\n(±{std:.2f})', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    ax2 = plt.subplot(2, 3, 2)
    utilization_means = [df[df['Approach'] == app]['Utilization'].mean() for app in approaches]
    utilization_stds = [df[df['Approach'] == app]['Utilization'].std() for app in approaches]
    
    bars2 = ax2.bar(approaches, utilization_means, yerr=utilization_stds, capsize=8,
                   color=colors[:len(approaches)], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_title('2. Average Utilization\n(60-80% Optimal)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Utilization (%)', fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, mean, std in zip(bars2, utilization_means, utilization_stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 2,
                f'{mean:.1f}%\n(±{std:.1f})', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    ax3 = plt.subplot(2, 3, 3)
    performance_scores = []
    for app in approaches:
        app_data = df[df['Approach'] == app]
        avg_response = app_data['ResponseTime'].mean()
        avg_utilization = app_data['Utilization'].mean()
        util_std = app_data['Utilization'].std()
        
        response_score = 10.0 / (avg_response + 0.1)
        utilization_score = avg_utilization / 100.0
        balance_score = 1.0 / (util_std + 0.1)
        
        performance_score = (response_score * 0.4) + (utilization_score * 0.4) + (balance_score * 0.2)
        performance_scores.append(performance_score * 100)
    
    bars3 = ax3.bar(approaches, performance_scores, color=colors[:len(approaches)], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_title('3. Overall Performance Score\n(Higher is Better)', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Performance Score', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, score in zip(bars3, performance_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax4 = plt.subplot(2, 3, 4)
    hybrid_data = df[df['Approach'] == 'Hybrid']
    if not hybrid_data.empty:
        algo_counts = hybrid_data['AlgorithmUsed'].value_counts()
        algo_counts = algo_counts[algo_counts.index != 'None']
        if not algo_counts.empty:
            wedges, texts, autotexts = ax4.pie(algo_counts.values, 
                                              labels=algo_counts.index, 
                                              autopct='%1.1f%%',
                                              colors=colors[:len(algo_counts)], 
                                              startangle=90,
                                              explode=[0.05] * len(algo_counts))
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
            
            for text in texts:
                text.set_fontweight('bold')
                text.set_fontsize=10
                
            ax4.set_title('4. Hybrid: Algorithm Choice %\n(Decision Distribution)', 
                         fontweight='bold', fontsize=12)
    else:
        ax4.text(0.5, 0.5, 'No Hybrid Data\nAvailable', 
                ha='center', va='center', fontweight='bold', fontsize=12)
        ax4.set_title('4. Algorithm Choice Distribution', fontweight='bold', fontsize=12)
    
    ax5 = plt.subplot(2, 3, 5)
    for i, app in enumerate(approaches):
        app_data = df[df['Approach'] == app]
        if len(app_data) > 5:
            app_data_sorted = app_data.sort_values('Timestamp')
            if len(app_data_sorted) > 20:
                sampled = app_data_sorted.iloc[::max(1, len(app_data_sorted)//15)]
            else:
                sampled = app_data_sorted
            
            time_normalized = sampled['Timestamp'] - sampled['Timestamp'].min()
            
            ax5.plot(time_normalized, sampled['ResponseTime'], 
                    marker='o', linewidth=2, label=app, color=colors[i], 
                    alpha=0.7, markersize=4)
    
    ax5.set_title('5. Response Time Trend Over Time', fontweight='bold', fontsize=12)
    ax5.set_xlabel('Time Progression', fontweight='bold')
    ax5.set_ylabel('Response Time (seconds)', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(f'LOAD BALANCER PERFORMANCE ANALYSIS\nSimulation ID: {current_simulation_id}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.subplots_adjust(top=0.90, bottom=0.10)
    
    plot_filename = f"{results_dir}/analysis_{current_simulation_id}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return plot_filename

def generate_summary_statistics(df: pd.DataFrame, results_dir: str, servers: List[SimulatedServer] = None):
    """Generate detailed summary statistics."""
    summary_filename = f"{results_dir}/summary_{current_simulation_id}.txt"
    
    with open(summary_filename, 'w') as f:
        f.write("LOAD BALANCER SIMULATION - COMPREHENSIVE SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Simulation ID: {current_simulation_id}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        approaches = df['Approach'].unique()
        
        for approach in approaches:
            app_data = df[df['Approach'] == approach]
            
            f.write(f"ALGORITHM: {approach}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Tasks Processed: {app_data['TotalTasks'].sum():,}\n")
            f.write(f"Unique Servers Used: {app_data['ServerID'].nunique()}\n")
            f.write(f"Average Response Time: {app_data['ResponseTime'].mean():.3f} seconds\n")
            f.write(f"Response Time Std Dev: {app_data['ResponseTime'].std():.3f} seconds\n")
            f.write(f"Average Utilization: {app_data['Utilization'].mean():.2f}%\n")
            f.write(f"Utilization Std Dev: {app_data['Utilization'].std():.2f}%\n")
            f.write(f"Average Task Load: {app_data['TaskLoad'].mean():.2f}\n")
            f.write(f"Min/Max Utilization: {app_data['Utilization'].min():.1f}% / {app_data['Utilization'].max():.1f}%\n")
            f.write(f"Min/Max Response Time: {app_data['ResponseTime'].min():.3f}s / {app_data['ResponseTime'].max():.3f}s\n")
            
            if approach == 'Hybrid':
                algo_usage = app_data['AlgorithmUsed'].value_counts()
                f.write("\nAlgorithm Usage Breakdown:\n")
                for algo, count in algo_usage.items():
                    if algo != 'None':
                        percentage = (count / len(app_data)) * 100
                        f.write(f"  {algo}: {count} tasks ({percentage:.1f}%)\n")
            
            f.write("\n" + "=" * 60 + "\n\n")

def visualize_assignment_step(servers, approach_name, task_count, num_tasks, 
                            scatter_fig, scatter_canvas, bar_fig, bar_canvas, 
                            tree, update_ui_callback):
    """Update live visualizations with current state."""
    if not servers:
        return
    
    active_servers = servers
    
    server_ids = [s.id for s in active_servers]
    utilizations = [s.get_utilization() for s in active_servers]
    response_times = [s.get_response_time_metric() for s in active_servers]
    task_loads = [s.last_task_load for s in active_servers]
    
    scatter_fig.clear()
    ax1 = scatter_fig.add_subplot(111)
    
    def get_algorithm_color(server):
        algo = server.last_algorithm_used
        if algo == "RR": return 'blue'
        elif algo == "ACO": return 'red'
        else: return 'gray'
    
    colors = [get_algorithm_color(s) for s in active_servers]
    fixed_size = 50
    
    scatter = ax1.scatter(response_times, utilizations, s=fixed_size, c=colors, 
                         alpha=0.7, edgecolors='black', linewidth=0.5, picker=True)
    
    for i, (server_id, util, resp) in enumerate(zip(server_ids, utilizations, response_times)):
        if util > 85 or resp > np.mean(response_times) * 1.3 or i % 4 == 0:
            ax1.annotate(server_id, (resp, util), 
                        xytext=(5, 5), textcoords='offset points', fontsize=7, alpha=0.8)
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='RR'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='ACO'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Idle/None')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    if response_times and utilizations:
        x_min = max(0.1, min(response_times) * 0.8)
        x_max = max(response_times) * 1.2
        y_min = max(0, min(utilizations) - 5)
        y_max = 105
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
    else:
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 105)
    
    ax1.set_xlabel('Response Time (seconds) - Lower is Better')
    ax1.set_ylabel('Utilization (%) - 60-80% Optimal')
    ax1.set_title(f'{approach_name} - Task {task_count}/{num_tasks}\nServer Performance Scatter')
    ax1.grid(True, alpha=0.3)
    
    bar_fig.clear()
    ax2 = bar_fig.add_subplot(111)
    
    x_positions = np.arange(len(server_ids))
    bars = ax2.bar(x_positions, utilizations, color=colors, alpha=0.7, width=0.6)
    ax2.set_xlabel('Server ID')
    ax2.set_ylabel('Utilization (%)')
    ax2.set_title('Server Utilization Distribution')
    ax2.set_ylim(0, 105)
    
    if server_ids:
        ax2.set_xlim(-0.5, len(server_ids) - 0.5)
        ax2.set_xticks(x_positions)
        rotation = 45 if len(server_ids) > 15 else 0
        fontsize = 6 if len(server_ids) > 20 else 8
        ax2.set_xticklabels(server_ids, rotation=rotation, ha='right' if rotation else 'center', 
                           fontsize=fontsize)
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, utilization, server_id in zip(bars, utilizations, server_ids):
        height = bar.get_height()
        if height > 85 or height < 20 or server_ids.index(server_id) % 6 == 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{utilization:.0f}%', ha='center', va='bottom', 
                    fontsize=6, rotation=0, alpha=0.8, fontweight='bold')
    
    for item in tree.get_children():
        tree.delete(item)
        
    for server in active_servers:
        tree.insert("", "end", values=(
            server.id,
            f"{server.last_task_load:.1f}",
            f"{server.get_utilization():.1f}%",
            f"{server.get_response_time_metric():.2f}",
            server.last_algorithm_used or "None",
            server.total_tasks_processed
        ))
    
    avg_util = np.mean(utilizations) if utilizations else 0
    avg_response = np.mean(response_times) if response_times else 0
    util_std = np.std(utilizations) if utilizations else 0
    resp_std = np.std(response_times) if response_times else 0
    
    optimal_util_servers = len([u for u in utilizations if 60 <= u <= 80])
    overloaded_servers = len([u for u in utilizations if u > 85])
    
    update_ui_callback(
        f"📊 Progress: {task_count}/{num_tasks} | "
        f"⚡ Avg Response: {avg_response:.2f}s (±{resp_std:.2f}) | "
        f"💾 Avg Util: {avg_util:.1f}% (±{util_std:.1f})\n"
        f"🎯 Optimal Servers: {optimal_util_servers}/{len(servers)} | "
        f"⚠️ Overloaded: {overloaded_servers} | "
        f"🔄 Active: {len([s for s in servers if s.current_load > 0])}"
    )
    
    scatter_fig.tight_layout()
    bar_fig.tight_layout()
    scatter_canvas.draw()
    bar_canvas.draw()

def on_pick(event, servers, tree, update_ui_callback):
    """Handle scatter plot point clicks."""
    if event.artist:
        ind = event.ind[0]
        if ind < len(servers):
            server = servers[ind]
            update_ui_callback(
                f"🔍 Server Details: {server.id}\n"
                f"   📊 Utilization: {server.get_utilization():.1f}%\n"
                f"   ⚡ Response Time: {server.get_response_time_metric():.2f}s\n"
                f"   📦 Current Load: {server.current_load:.1f}/{server.max_capacity}\n"
                f"   🔄 Last Task: {server.last_task_load:.1f}\n"
                f"   📈 Total Tasks: {server.total_tasks_processed}\n"
                f"   🎯 Last Algorithm: {server.last_algorithm_used or 'None'}\n"
                f"   ⏱️ Busy Time: {server.busy_time:.1f}s"
            )

def analyze_and_plot(approach_name, final_fig, final_canvas, update_ui_callback, servers=None):
    """Display final analysis with performance bar chart."""
    if not all_metrics_log:
        update_ui_callback("No metrics data available for analysis.")
        return
    
    result = save_comprehensive_metrics()
    
    if result:
        csv_filename, plot_filename = result
        update_ui_callback("📊 RESULTS SAVED:")
        update_ui_callback("=" * 50)
        update_ui_callback(f"📁 Metrics CSV: {csv_filename}")
        update_ui_callback(f"🖼️ Analysis Plots: {plot_filename}")
        update_ui_callback("=" * 50)
    
    df = pd.DataFrame(all_metrics_log)
    results_dir = "simulation_results"
    generate_summary_statistics(df, results_dir, servers)
    
    current_sim_metrics = df[df['SimulationID'] == current_simulation_id]
    
    if current_sim_metrics.empty:
        update_ui_callback("No metrics data for current simulation.")
        return
    
    final_fig.clear()
    ax = final_fig.add_subplot(111)
    
    approaches = current_sim_metrics['Approach'].unique()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(approaches)]
    
    metrics_data = {}
    for app in approaches:
        app_data = current_sim_metrics[current_sim_metrics['Approach'] == app]
        metrics_data[app] = {
            'response': app_data['ResponseTime'].mean(),
            'utilization': app_data['Utilization'].mean(),
            'balance': 1.0 / (app_data['Utilization'].std() + 0.1),
            'throughput': app_data['TotalTasks'].sum()
        }
    
    max_response = max(metrics_data[app]['response'] for app in approaches)
    response_scores = [1.0 - (metrics_data[app]['response'] / max_response) for app in approaches]
    utilization_scores = [metrics_data[app]['utilization'] / 100.0 for app in approaches]
    performance_scores = [(resp * 0.5 + util * 0.5) * 100 for resp, util in zip(response_scores, utilization_scores)]
    
    bars = ax.bar(approaches, performance_scores, color=colors, 
                 alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Load Balancing Approach')
    ax.set_ylabel('Performance Score')
    ax.set_title('Final Performance Comparison\n(Higher is Better)', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)
    
    for bar, app, score in zip(bars, approaches, performance_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{score:.1f}\n({metrics_data[app]["response"]:.2f}s)', 
               ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    final_fig.tight_layout()
    final_canvas.draw()
    
    update_ui_callback("🎯 SIMULATION COMPLETE - KEY METRICS:")
    update_ui_callback("=" * 50)
    for app in approaches:
        data = metrics_data[app]
        update_ui_callback(
            f"{app}:\n"
            f"  ⚡ Avg Response: {data['response']:.3f}s\n"
            f"  💾 Avg Utilization: {data['utilization']:.1f}%\n"
            f"  📈 Total Throughput: {data['throughput']:,} tasks\n"
            f"  🎯 Performance Score: {performance_scores[list(approaches).index(app)]:.1f}"
        )
    update_ui_callback("=" * 50)
    update_ui_callback("✅ Check 'simulation_results' folder for analysis!")