#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Lingual Backend Preference Visualization
Shows which backend is preferred per language based on learned heuristics
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict


def plot_backend_preference_pie(learned_heuristics: Dict[str, Dict],
                                output_file: str = 'backend_preference_pie.png'):
    """
    Create pie chart showing overall backend preference distribution
    
    Args:
        learned_heuristics: Dict mapping languages to heuristic info
        output_file: Output filename for the plot
    """
    backend_counts = {}
    for lang, heuristic in learned_heuristics.items():
        backend = heuristic['preferred_backend']
        backend_counts[backend] = backend_counts.get(backend, 0) + 1
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    explode = [0.05] * len(backend_counts)
    
    wedges, texts, autotexts = ax.pie(backend_counts.values(), 
                                       labels=backend_counts.keys(),
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       colors=colors[:len(backend_counts)],
                                       explode=explode,
                                       shadow=True)
    
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    ax.set_title('Backend Preference Distribution\nAcross Languages', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Backend preference pie chart saved to {output_file}")
    plt.close()


def plot_language_backend_matrix(learned_heuristics: Dict[str, Dict],
                                 output_file: str = 'language_backend_matrix.png'):
    """
    Create matrix visualization showing language-backend preferences with rewards
    
    Args:
        learned_heuristics: Dict mapping languages to heuristic info
        output_file: Output filename for the plot
    """
    languages = list(learned_heuristics.keys())
    backends = list(set(h['preferred_backend'] for h in learned_heuristics.values()))
    
    # Create matrix
    matrix = np.zeros((len(languages), len(backends)))
    for i, lang in enumerate(languages):
        backend = learned_heuristics[lang]['preferred_backend']
        j = backends.index(backend)
        matrix[i, j] = learned_heuristics[lang]['avg_reward']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(matrix, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(backends)))
    ax.set_yticks(np.arange(len(languages)))
    ax.set_xticklabels(backends, fontsize=11)
    ax.set_yticklabels(languages, fontsize=11)
    
    ax.set_xlabel('Backend', fontsize=12, fontweight='bold')
    ax.set_ylabel('Language', fontsize=12, fontweight='bold')
    ax.set_title('Language-Backend Preference Matrix\n(Colored by Average Reward)', 
                 fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Reward', fontsize=11, fontweight='bold')
    
    # Add text annotations
    for i in range(len(languages)):
        for j in range(len(backends)):
            if matrix[i, j] > 0:
                text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                              ha="center", va="center", 
                              color="white" if matrix[i, j] > 0.5 else "black",
                              fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Language-backend matrix saved to {output_file}")
    plt.close()


def plot_backend_preference_bars(learned_heuristics: Dict[str, Dict],
                                 output_file: str = 'backend_preference_bars.png'):
    """
    Create horizontal bar chart showing backend preferences with rewards
    
    Args:
        learned_heuristics: Dict mapping languages to heuristic info
        output_file: Output filename for the plot
    """
    languages = list(learned_heuristics.keys())
    rewards = [learned_heuristics[lang]['avg_reward'] for lang in languages]
    backends = [learned_heuristics[lang]['preferred_backend'] for lang in languages]
    
    # Color by backend
    backend_colors = {'ibm': '#3498db', 'russian': '#e74c3c', 
                     'google': '#2ecc71', 'ionq': '#f39c12'}
    colors = [backend_colors.get(b, '#95a5a6') for b in backends]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(languages))
    bars = ax.barh(y_pos, rewards, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(languages, fontsize=11)
    ax.set_xlabel('Average Reward', fontsize=12, fontweight='bold')
    ax.set_title('Backend Preference by Language\n(Colored by Preferred Backend)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels and backend names
    for i, (bar, reward, backend) in enumerate(zip(bars, rewards, backends)):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{reward:.3f} ({backend})',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=backend, edgecolor='black')
                      for backend, color in backend_colors.items()
                      if backend in backends]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Backend preference bars saved to {output_file}")
    plt.close()


if __name__ == '__main__':
    # Example usage
    learned_heuristics = {
        'ru': {'preferred_backend': 'ibm', 'avg_reward': 0.807, 'edit_count': 5},
        'zh': {'preferred_backend': 'russian', 'avg_reward': 0.814, 'edit_count': 4},
        'es': {'preferred_backend': 'russian', 'avg_reward': 0.853, 'edit_count': 2},
        'fr': {'preferred_backend': 'russian', 'avg_reward': 0.842, 'edit_count': 2},
        'en': {'preferred_backend': 'russian', 'avg_reward': 0.803, 'edit_count': 2}
    }
    
    plot_backend_preference_pie(learned_heuristics)
    plot_language_backend_matrix(learned_heuristics)
    plot_backend_preference_bars(learned_heuristics)
