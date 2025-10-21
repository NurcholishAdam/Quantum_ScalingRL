# Quantum-Scaling RL Hybrid Agent

A self-improving hybrid agent that integrates quantum optimization with reinforcement learning for multilingual semantic graph editing.

## Quick Start

```python
from quantum_scaling_rl_hybrid import QuantumScalingRLHybrid, QuantumRLConfig

# Initialize agent
config = QuantumRLConfig(backends=['ibm', 'russian'])
agent = QuantumScalingRLHybrid(config)

# Run edit cycle
result = agent.run_edit_cycle(edit, corpus)
print(f"Performance: {result.performance_delta:.3f}")
```

## Run Demo

```bash
# Simple demo (no quantum dependencies)
python agent/demo_quantum_scaling_rl_simple.py

# Full demo (requires qiskit)
pip install qiskit qiskit-machine-learning
python agent/demo_quantum_scaling_rl.py

# Visualization demo
python agent/visualizations/demo_all_visualizations.py
```

## Architecture: 5-Stage Pipeline

1. **Quantum Optimization** - QAOA traversal, QSVM hallucination detection, QEC correction
2. **RLHF Adaptation** - KL-regularized PPO, backend selection learning
3. **ScalingRL Budgeting** - Batch sizing (∝ √model_size), reward shaping, compute tracking
4. **Feedback Loop** - Reflector, curator, RL retraining
5. **Benchmarking & Visualization** - Performance metrics and visual analytics

## Key Features

- ✅ Self-improving: Learns optimal backends per language
- ✅ Multilingual: Adapts strategies for each language (ru, zh, es, fr, en)
- ✅ Compute-efficient: Optimizes batch sizes and resources
- ✅ Benchmarking: Tracks IBM vs Russian backend performance
- ✅ **NEW**: Comprehensive visualization suite (4 modules, 11 charts)

## Visualization Modules

**Location**: `agent/visualizations/`

1. **Backend Performance Comparison** - IBM vs Russian backend analysis
2. **Reward vs Batch Size Scaling** - Validates batch_size ∝ √(model_size)
3. **Cross-Lingual Backend Preference** - Language-specific backend preferences
4. **Performance Trend Over Edit Cycles** - Learning curves and improvement tracking

```bash
# Generate all visualizations
cd agent/visualizations
python demo_all_visualizations.py
# Output: 11 high-resolution PNG charts in output/ directory
```

## Files

### Core Implementation
- `quantum_scaling_rl_hybrid.py` - Main implementation (450+ lines)
- `demo_quantum_scaling_rl_simple.py` - Simple demo (tested & working)
- `demo_quantum_scaling_rl.py` - Full demo (requires qiskit)
- `test_quantum_scaling_rl.py` - Test suite (13 tests)

### Visualization Modules
- `visualizations/Backend_Performance_Comparison.py`
- `visualizations/Reward_vs_BatchSize_Scaling.py`
- `visualizations/Cross_Lingual_Backend_Preference.py`
- `visualizations/Performance_Trend_Over_Edit_Cycles.py`
- `visualizations/demo_all_visualizations.py`

### Documentation
- `QUANTUM_SCALING_RL_ARCHITECTURE.md` - Complete 5-stage architecture
- `QUANTUM_SCALING_RL_HYBRID_DOCUMENTATION.md` - Full technical docs
- `QUANTUM_SCALING_RL_QUICK_REFERENCE.md` - Quick reference
- `QUANTUM_SCALING_RL_IMPLEMENTATION_SUMMARY.md` - Implementation summary

## Demo Results

```
Total Edits: 15
Performance Trend: improving

Backend Performance:
  ibm:     Mean Reward: 0.807 ± 0.022
  russian: Mean Reward: 0.825 ± 0.024

Learned Heuristics:
  ru: Preferred Backend: ibm (0.807)
  zh: Preferred Backend: russian (0.814)
  es: Preferred Backend: russian (0.853)
  fr: Preferred Backend: russian (0.842)
  en: Preferred Backend: russian (0.803)
```

## Performance Metrics

### Quantum Metrics
- QAOA Coherence: 0.6-0.9
- QEC Logical Error: 0.001-0.01
- QSVM Valid Prob: 0.7-0.95

### RL Metrics
- Final Reward: 0.75-0.88
- Edit Reliability: 0.99-1.0
- KL Penalty: 0.0-0.01

### Scaling Metrics
- Compute Efficiency: 6-11 reward/sec
- Optimal Batch Size: 8-16
- Performance Trend: Improving

## Dependencies

```bash
# Core (required)
pip install numpy

# Visualization (required for charts)
pip install matplotlib

# Quantum (optional, for full functionality)
pip install qiskit qiskit-machine-learning torch transformers
```

## Integration

### With Quantum Modules
- `qaoa_traversal.py` - Semantic graph optimization
- `qsvm_hallucination.py` - Hallucination detection
- `repair_qec_extension.py` - Error correction

### With RLHF System
- `rlhf/reward_model.py` - Reward model manager
- `rlhf/rl_trainer.py` - RL training config

### With Scaling Laws
- `scaling_laws/scaling_measurement_framework.py` - Scaling analysis

## Usage with Visualizations

```python
from quantum_scaling_rl_hybrid import QuantumScalingRLHybrid
from visualizations.Backend_Performance_Comparison import plot_backend_performance_comparison

# Run agent
agent = QuantumScalingRLHybrid()
for i in range(30):
    result = agent.run_edit_cycle(edit, corpus)

# Get statistics
stats = agent.get_statistics()

# Visualize results
plot_backend_performance_comparison(
    stats['backend_performance'],
    'backend_comparison.png'
)
```

## License

MIT License
