# Quantum-Scaling RL Visualization Modules

Four visualization modules for analyzing Quantum-Scaling RL Hybrid Agent performance.

## Modules Overview

### 1. Backend Performance Comparison
Compares IBM vs Russian backends across languages with mean reward and standard deviation.

**Visualizations**: Bar charts with error bars, grouped bars per language

### 2. Reward vs Batch Size Scaling
Shows how reward scales with batch size across different model sizes.

**Visualizations**: Scatter plots, scaling law validation, efficiency heatmaps

### 3. Cross-Lingual Backend Preference
Displays backend preferences per language based on learned heuristics.

**Visualizations**: Pie charts, language-backend matrices, horizontal bars

### 4. Performance Trend Over Edit Cycles
Tracks agent improvement over time through RL retraining and heuristic updates.

**Visualizations**: Line plots with moving average, stacked area charts, learning curves

## Quick Start

```bash
# Run demo (generates 11 visualizations)
cd agent/visualizations
python demo_all_visualizations.py
```

## Usage Example

```python
from Backend_Performance_Comparison import plot_backend_performance_comparison

backend_performance = {
    'ibm': [0.807, 0.785, 0.820],
    'russian': [0.825, 0.810, 0.840]
}

plot_backend_performance_comparison(backend_performance, 'output.png')
```

## Integration

```python
from quantum_scaling_rl_hybrid import QuantumScalingRLHybrid
from visualizations.Backend_Performance_Comparison import plot_backend_performance_comparison

agent = QuantumScalingRLHybrid()
# ... run edit cycles ...
stats = agent.get_statistics()
plot_backend_performance_comparison(stats['backend_performance'])
```

## Dependencies

```bash
pip install matplotlib numpy
```

## Files

- `Backend_Performance_Comparison.py` - Backend comparison charts
- `Reward_vs_BatchSize_Scaling.py` - Batch size scaling analysis
- `Cross_Lingual_Backend_Preference.py` - Language preference visualization
- `Performance_Trend_Over_Edit_Cycles.py` - Performance trend tracking
- `demo_all_visualizations.py` - Complete demo script

## Output

All visualizations are 300 DPI PNG files with professional styling, clear labels, and color-coded data.
