#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reward vs Batch Size Scaling Visualization
Visualizes how reward scales with batch size across different model sizes
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple


def plot_reward_vs_batch_size(batch_sizes: List[int], 
                              rewards: List[float],
                              model_sizes: List[float],
                              output_file: str = 'reward_vs_batch_size.png'):
    """
    Create scatter plot showing reward vs batch size colored by model size
    
    Args:
        batch_sizes: List of batch sizes used
        rewards: List of corresponding rewards
        model_sizes: List of model size proxies
        output_file: Output filename for the plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    scatter = ax.scatter(batch_sizes, rewards, c=model_sizes, 
                        s=100, alpha=0.6, cmap='viridis', edgecolors='black')
    
    # Add trend line
    z = np.polyfit(batch_sizes, rewards, 2)
    p = np.poly1d(z)
    x_trend = np.linspace(min(batch_sizes), max(batch_sizes), 100)
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend')
    
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=12, fontweight='bold')
    ax.set_title('Reward vs Batch Size Scaling\n(Colored by Model Size)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend()
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Model Size Proxy', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Reward vs batch size saved to {output_file}")
    plt.close()


def plot_scaling_law_validation(model_sizes: List[float],
                                optimal_batch_sizes: List[int],
                                output_file: str = 'scaling_law_validation.png'):
    """
    Validate batch_size ∝ √(model_size) scaling law
    
    Args:
        model_sizes: List of model size proxies
        optimal_batch_sizes: List of computed optimal batch sizes
        output_file: Output filename for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot actual data
    ax.scatter(model_sizes, optimal_batch_sizes, s=100, alpha=0.7, 
              label='Actual', color='#3498db', edgecolors='black')
    
    # Plot theoretical scaling law
    base_batch = optimal_batch_sizes[0] / np.sqrt(model_sizes[0])
    theoretical = [base_batch * np.sqrt(m) for m in model_sizes]
    ax.plot(model_sizes, theoretical, 'r--', linewidth=2, 
           label='Theoretical: batch ∝ √(model_size)')
    
    ax.set_xlabel('Model Size Proxy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Optimal Batch Size', fontsize=12, fontweight='bold')
    ax.set_title('Scaling Law Validation\nbatch_size ∝ √(model_size)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Scaling law validation saved to {output_file}")
    plt.close()


def plot_compute_efficiency_heatmap(batch_sizes: List[int],
                                   model_sizes: List[float],
                                   efficiencies: np.ndarray,
                                   output_file: str = 'compute_efficiency_heatmap.png'):
    """
    Create heatmap of compute efficiency across batch sizes and model sizes
    
    Args:
        batch_sizes: List of batch sizes
        model_sizes: List of model sizes
        efficiencies: 2D array of compute efficiencies
        output_file: Output filename for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(efficiencies, cmap='RdYlGn', aspect='auto', 
                   interpolation='nearest')
    
    ax.set_xticks(np.arange(len(batch_sizes)))
    ax.set_yticks(np.arange(len(model_sizes)))
    ax.set_xticklabels(batch_sizes)
    ax.set_yticklabels([f'{m:.2f}' for m in model_sizes])
    
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model Size Proxy', fontsize=12, fontweight='bold')
    ax.set_title('Compute Efficiency Heatmap\n(Reward per Second)', 
                 fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Efficiency (reward/sec)', fontsize=11, fontweight='bold')
    
    # Add text annotations
    for i in range(len(model_sizes)):
        for j in range(len(batch_sizes)):
            text = ax.text(j, i, f'{efficiencies[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Compute efficiency heatmap saved to {output_file}")
    plt.close()


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)
    
    # Generate sample data
    batch_sizes = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    model_sizes = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    rewards = [0.70 + 0.05 * np.sqrt(b) + 0.02 * np.random.randn() 
               for b in batch_sizes]
    
    plot_reward_vs_batch_size(batch_sizes, rewards, model_sizes)
    
    # Scaling law validation
    optimal_batch_sizes = [int(8 * np.sqrt(m)) for m in model_sizes]
    plot_scaling_law_validation(model_sizes, optimal_batch_sizes)
    
    # Compute efficiency heatmap
    efficiencies = np.random.uniform(5, 12, (len(model_sizes), len(batch_sizes)))
    plot_compute_efficiency_heatmap(batch_sizes, model_sizes, efficiencies)
