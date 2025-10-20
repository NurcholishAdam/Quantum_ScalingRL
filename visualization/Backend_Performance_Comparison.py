#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backend Performance Comparison Visualization
Compares IBM vs Russian backends across languages using mean reward and standard deviation
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


def plot_backend_performance_comparison(backend_performance: Dict[str, List[float]], 
                                       output_file: str = 'backend_comparison.png'):
    """
    Create bar chart comparing backend performance with error bars
    
    Args:
        backend_performance: Dict mapping backend names to reward lists
        output_file: Output filename for the plot
    """
    backends = list(backend_performance.keys())
    means = [np.mean(backend_performance[b]) if backend_performance[b] else 0 
             for b in backends]
    stds = [np.std(backend_performance[b]) if backend_performance[b] else 0 
            for b in backends]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(backends))
    width = 0.6
    
    bars = ax.bar(x, means, width, yerr=stds, capsize=10,
                  color=['#3498db', '#e74c3c'], alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Backend', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
    ax.set_title('Backend Performance Comparison\n(IBM vs Russian)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(backends)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mean:.3f}±{std:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Backend comparison saved to {output_file}")
    plt.close()


def plot_backend_performance_by_language(learned_heuristics: Dict[str, Dict],
                                        backend_performance: Dict[str, List[float]],
                                        output_file: str = 'backend_by_language.png'):
    """
    Create grouped bar chart showing backend performance per language
    
    Args:
        learned_heuristics: Dict mapping languages to heuristic info
        backend_performance: Dict mapping backend names to reward lists
        output_file: Output filename for the plot
    """
    languages = list(learned_heuristics.keys())
    backends = list(backend_performance.keys())
    
    # Organize data by language and backend
    data = {backend: [] for backend in backends}
    for lang in languages:
        preferred = learned_heuristics[lang]['preferred_backend']
        avg_reward = learned_heuristics[lang]['avg_reward']
        data[preferred].append(avg_reward)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(languages))
    width = 0.35
    
    # Create bars for each backend
    for i, backend in enumerate(backends):
        rewards = [learned_heuristics[lang]['avg_reward'] 
                  if learned_heuristics[lang]['preferred_backend'] == backend 
                  else 0 for lang in languages]
        offset = width * (i - len(backends)/2 + 0.5)
        ax.bar(x + offset, rewards, width, label=backend, alpha=0.8)
    
    ax.set_xlabel('Language', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
    ax.set_title('Backend Performance by Language', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(languages)
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Backend by language saved to {output_file}")
    plt.close()


if __name__ == '__main__':
    # Example usage
    backend_performance = {
        'ibm': [0.807, 0.785, 0.820, 0.795, 0.830],
        'russian': [0.825, 0.810, 0.840, 0.815, 0.835]
    }
    
    learned_heuristics = {
        'ru': {'preferred_backend': 'ibm', 'avg_reward': 0.807},
        'zh': {'preferred_backend': 'russian', 'avg_reward': 0.814},
        'es': {'preferred_backend': 'russian', 'avg_reward': 0.853},
        'fr': {'preferred_backend': 'russian', 'avg_reward': 0.842},
        'en': {'preferred_backend': 'russian', 'avg_reward': 0.803}
    }
    
    plot_backend_performance_comparison(backend_performance)
    plot_backend_performance_by_language(learned_heuristics, backend_performance)
