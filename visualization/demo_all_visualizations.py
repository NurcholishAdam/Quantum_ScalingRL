#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: All Quantum-Scaling RL Visualizations
Demonstrates all four visualization modules with sample data
"""
import sys
sys.path.append('..')
import numpy as np
from Backend_Performance_Comparison import (
    plot_backend_performance_comparison,
    plot_backend_performance_by_language
)
from Reward_vs_BatchSize_Scaling import (
    plot_reward_vs_batch_size,
    plot_scaling_law_validation,
    plot_compute_efficiency_heatmap
)
from Cross_Lingual_Backend_Preference import (
    plot_backend_preference_pie,
    plot_language_backend_matrix,
    plot_backend_preference_bars
)
from Performance_Trend_Over_Edit_Cycles import (
    plot_performance_trend,
    plot_backend_usage_over_time,
    plot_learning_curve_with_retraining
)


def generate_sample_data():
    """Generate realistic sample data for all visualizations"""
    np.random.seed(42)
    
    # Backend performance data
    backend_performance = {
        'ibm': [0.807, 0.785, 0.820, 0.795, 0.830],
        'russian': [0.825, 0.810, 0.840, 0.815, 0.835, 0.820, 0.845, 0.830, 0.825, 0.838]
    }
    
    # Learned heuristics
    learned_heuristics = {
        'ru': {'preferred_backend': 'ibm', 'avg_reward': 0.807, 'edit_count': 5},
        'zh': {'preferred_backend': 'russian', 'avg_reward': 0.814, 'edit_count': 4},
        'es': {'preferred_backend': 'russian', 'avg_reward': 0.853, 'edit_count': 2},
        'fr': {'preferred_backend': 'russian', 'avg_reward': 0.842, 'edit_count': 2},
        'en': {'preferred_backend': 'russian', 'avg_reward': 0.803, 'edit_count': 2}
    }
    
    # Batch size scaling data
    batch_sizes = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    model_sizes = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    rewards = [0.70 + 0.05 * np.sqrt(b) + 0.02 * np.random.randn() 
               for b in batch_sizes]
    optimal_batch_sizes = [int(8 * np.sqrt(m)) for m in model_sizes]
    
    # Compute efficiency heatmap
    efficiencies = np.random.uniform(5, 12, (len(model_sizes), len(batch_sizes)))
    
    # Edit history
    edit_history = []
    for i in range(30):
        base_reward = 0.65 + 0.01 * i + 0.05 * np.random.randn()
        performance_delta = base_reward - 0.5
        edit_history.append({
            'edit_id': f'edit_{i}',
            'backend': 'russian' if i > 5 else np.random.choice(['ibm', 'russian']),
            'performance_delta': performance_delta,
            'reward': base_reward
        })
    
    retrain_intervals = [10, 20, 30]
    
    return {
        'backend_performance': backend_performance,
        'learned_heuristics': learned_heuristics,
        'batch_sizes': batch_sizes,
        'model_sizes': model_sizes,
        'rewards': rewards,
        'optimal_batch_sizes': optimal_batch_sizes,
        'efficiencies': efficiencies,
        'edit_history': edit_history,
        'retrain_intervals': retrain_intervals
    }


def main():
    print("=" * 80)
    print("Quantum-Scaling RL Visualization Demo")
    print("=" * 80)
    print()
    
    # Generate sample data
    print("Generating sample data...")
    data = generate_sample_data()
    print("✓ Sample data generated")
    print()
    
    # Module 1: Backend Performance Comparison
    print("=" * 80)
    print("Module 1: Backend Performance Comparison")
    print("=" * 80)
    plot_backend_performance_comparison(
        data['backend_performance'],
        'output/backend_comparison.png'
    )
    plot_backend_performance_by_language(
        data['learned_heuristics'],
        data['backend_performance'],
        'output/backend_by_language.png'
    )
    print()
    
    # Module 2: Reward vs Batch Size Scaling
    print("=" * 80)
    print("Module 2: Reward vs Batch Size Scaling")
    print("=" * 80)
    plot_reward_vs_batch_size(
        data['batch_sizes'],
        data['rewards'],
        data['model_sizes'],
        'output/reward_vs_batch_size.png'
    )
    plot_scaling_law_validation(
        data['model_sizes'],
        data['optimal_batch_sizes'],
        'output/scaling_law_validation.png'
    )
    plot_compute_efficiency_heatmap(
        data['batch_sizes'],
        data['model_sizes'],
        data['efficiencies'],
        'output/compute_efficiency_heatmap.png'
    )
    print()
    
    # Module 3: Cross-Lingual Backend Preference
    print("=" * 80)
    print("Module 3: Cross-Lingual Backend Preference")
    print("=" * 80)
    plot_backend_preference_pie(
        data['learned_heuristics'],
        'output/backend_preference_pie.png'
    )
    plot_language_backend_matrix(
        data['learned_heuristics'],
        'output/language_backend_matrix.png'
    )
    plot_backend_preference_bars(
        data['learned_heuristics'],
        'output/backend_preference_bars.png'
    )
    print()
    
    # Module 4: Performance Trend Over Edit Cycles
    print("=" * 80)
    print("Module 4: Performance Trend Over Edit Cycles")
    print("=" * 80)
    plot_performance_trend(
        data['edit_history'],
        'output/performance_trend.png'
    )
    plot_backend_usage_over_time(
        data['edit_history'],
        'output/backend_usage_trend.png'
    )
    plot_learning_curve_with_retraining(
        data['edit_history'],
        data['retrain_intervals'],
        'output/learning_curve.png'
    )
    print()
    
    print("=" * 80)
    print("All Visualizations Complete!")
    print("=" * 80)
    print()
    print("Generated 10 visualization files in output/ directory:")
    print("  1. backend_comparison.png")
    print("  2. backend_by_language.png")
    print("  3. reward_vs_batch_size.png")
    print("  4. scaling_law_validation.png")
    print("  5. compute_efficiency_heatmap.png")
    print("  6. backend_preference_pie.png")
    print("  7. language_backend_matrix.png")
    print("  8. backend_preference_bars.png")
    print("  9. performance_trend.png")
    print(" 10. backend_usage_trend.png")
    print(" 11. learning_curve.png")


if __name__ == '__main__':
    import os
    os.makedirs('output', exist_ok=True)
    main()
