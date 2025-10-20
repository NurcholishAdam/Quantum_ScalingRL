#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Trend Over Edit Cycles Visualization
Tracks how the agent improves over time through RL retraining and heuristic updates
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict


def plot_performance_trend(edit_history: List[Dict],
                          output_file: str = 'performance_trend.png'):
    """
    Create line plot showing performance improvement over edit cycles
    
    Args:
        edit_history: List of edit cycle results
        output_file: Output filename for the plot
    """
    cycles = list(range(1, len(edit_history) + 1))
    performance_deltas = [e['performance_delta'] for e in edit_history]
    rewards = [e.get('reward', 0.5 + e['performance_delta']) for e in edit_history]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Performance Delta
    ax1.plot(cycles, performance_deltas, 'o-', linewidth=2, markersize=6,
            color='#3498db', label='Performance Delta')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(cycles, 0, performance_deltas, alpha=0.3, color='#3498db')
    
    # Add moving average
    window = 3
    if len(performance_deltas) >= window:
        moving_avg = np.convolve(performance_deltas, np.ones(window)/window, mode='valid')
        ax1.plot(range(window, len(cycles)+1), moving_avg, 'r--', linewidth=2,
                label=f'{window}-Cycle Moving Average')
    
    ax1.set_xlabel('Edit Cycle', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Performance Delta', fontsize=12, fontweight='bold')
    ax1.set_title('Performance Delta Over Edit Cycles', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Cumulative Reward
    ax2.plot(cycles, rewards, 'o-', linewidth=2, markersize=6,
            color='#2ecc71', label='Reward')
    ax2.fill_between(cycles, min(rewards), rewards, alpha=0.3, color='#2ecc71')
    
    # Add trend line
    z = np.polyfit(cycles, rewards, 2)
    p = np.poly1d(z)
    ax2.plot(cycles, p(cycles), 'r--', linewidth=2, label='Trend')
    
    ax2.set_xlabel('Edit Cycle', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Reward', fontsize=12, fontweight='bold')
    ax2.set_title('Reward Progression Over Edit Cycles', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Performance trend saved to {output_file}")
    plt.close()


def plot_backend_usage_over_time(edit_history: List[Dict],
                                 output_file: str = 'backend_usage_trend.png'):
    """
    Create stacked area chart showing backend usage over time
    
    Args:
        edit_history: List of edit cycle results
        output_file: Output filename for the plot
    """
    cycles = list(range(1, len(edit_history) + 1))
    backends = list(set(e['backend'] for e in edit_history))
    
    # Count backend usage in windows
    window_size = 5
    backend_counts = {b: [] for b in backends}
    
    for i in range(len(edit_history)):
        start = max(0, i - window_size + 1)
        window = edit_history[start:i+1]
        total = len(window)
        for backend in backends:
            count = sum(1 for e in window if e['backend'] == backend)
            backend_counts[backend].append(count / total)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    ax.stackplot(cycles, *[backend_counts[b] for b in backends],
                labels=backends, colors=colors[:len(backends)], alpha=0.8)
    
    ax.set_xlabel('Edit Cycle', fontsize=12, fontweight='bold')
    ax.set_ylabel('Backend Usage Proportion', fontsize=12, fontweight='bold')
    ax.set_title(f'Backend Usage Over Time\n({window_size}-Cycle Rolling Window)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Backend usage trend saved to {output_file}")
    plt.close()


def plot_learning_curve_with_retraining(edit_history: List[Dict],
                                        retrain_intervals: List[int],
                                        output_file: str = 'learning_curve.png'):
    """
    Create learning curve with retraining markers
    
    Args:
        edit_history: List of edit cycle results
        retrain_intervals: List of cycle numbers where retraining occurred
        output_file: Output filename for the plot
    """
    cycles = list(range(1, len(edit_history) + 1))
    rewards = [e.get('reward', 0.5 + e['performance_delta']) for e in edit_history]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot rewards
    ax.plot(cycles, rewards, 'o-', linewidth=2, markersize=5,
           color='#3498db', alpha=0.7, label='Reward')
    
    # Add retraining markers
    for retrain_cycle in retrain_intervals:
        if retrain_cycle <= len(cycles):
            ax.axvline(x=retrain_cycle, color='red', linestyle='--', 
                      alpha=0.7, linewidth=2)
            ax.text(retrain_cycle, max(rewards) * 0.95, 'Retrain',
                   rotation=90, va='top', ha='right', fontsize=9,
                   color='red', fontweight='bold')
    
    # Add confidence band
    window = 5
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        moving_std = [np.std(rewards[max(0, i-window):i+1]) 
                     for i in range(window-1, len(rewards))]
        x_avg = range(window, len(cycles)+1)
        ax.plot(x_avg, moving_avg, 'g-', linewidth=3, label='Moving Average')
        ax.fill_between(x_avg, 
                       np.array(moving_avg) - np.array(moving_std),
                       np.array(moving_avg) + np.array(moving_std),
                       alpha=0.2, color='green', label='±1 Std Dev')
    
    ax.set_xlabel('Edit Cycle', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=12, fontweight='bold')
    ax.set_title('Learning Curve with RL Retraining Events', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Learning curve saved to {output_file}")
    plt.close()


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)
    
    # Generate sample edit history
    edit_history = []
    for i in range(30):
        # Simulate improving performance
        base_reward = 0.65 + 0.01 * i + 0.05 * np.random.randn()
        performance_delta = base_reward - 0.5
        edit_history.append({
            'edit_id': f'edit_{i}',
            'backend': np.random.choice(['ibm', 'russian']),
            'performance_delta': performance_delta,
            'reward': base_reward
        })
    
    # Retraining every 10 cycles
    retrain_intervals = [10, 20, 30]
    
    plot_performance_trend(edit_history)
    plot_backend_usage_over_time(edit_history)
    plot_learning_curve_with_retraining(edit_history, retrain_intervals)
