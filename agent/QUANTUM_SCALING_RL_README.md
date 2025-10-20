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
```

## Architecture

**4-Step Integration:**

1. **Quantum Optimization** - QAOA traversal, QSVM hallucination detection, QEC correction
2. **RLHF Adaptation** - KL-regularized PPO, backend selection learning
3. **ScalingRL Budgeting** - Batch sizing, reward shaping, compute tracking
4. **Feedback Loop** - Reflector, curator, RL retraining

## Key Features

- ✅ Self-improving: Learns optimal backends per language
- ✅ Multilingual: Adapts strategies for each language
- ✅ Compute-efficient: Optimizes batch sizes and resources
- ✅ Benchmarking: Tracks IBM vs Russian backend performance

## Files

- `quantum_scaling_rl_hybrid.py` - Main implementation (450+ lines)
- `demo_quantum_scaling_rl_simple.py` - Simple demo (works without qiskit)
- `demo_quantum_scaling_rl.py` - Full demo (requires qiskit)
- `test_quantum_scaling_rl.py` - Test suite
- `QUANTUM_SCALING_RL_HYBRID_DOCUMENTATION.md` - Complete docs
- `QUANTUM_SCALING_RL_QUICK_REFERENCE.md` - Quick reference

## Demo Results

```
Total Edits: 15
Performance Trend: improving

Backend Performance:
  ibm:     Mean Reward: 0.807
  russian: Mean Reward: 0.825

Learned Heuristics:
  ru: Preferred Backend: ibm (0.807)
  zh: Preferred Backend: russian (0.814)
  es: Preferred Backend: russian (0.853)
```

## Documentation

- Full docs: `QUANTUM_SCALING_RL_HYBRID_DOCUMENTATION.md`
- Quick reference: `QUANTUM_SCALING_RL_QUICK_REFERENCE.md`
- Implementation summary: `QUANTUM_SCALING_RL_IMPLEMENTATION_SUMMARY.md`

## Dependencies

```bash
# Core (required)
pip install numpy

# Quantum (optional, for full functionality)
pip install qiskit qiskit-machine-learning torch transformers
```

## Integration

Uses existing quantum modules:
- `qaoa_traversal.py` - Semantic graph optimization
- `qsvm_hallucination.py` - Hallucination detection
- `repair_qec_extension.py` - Error correction

Integrates with:
- RLHF system (`rlhf/reward_model.py`, `rlhf/rl_trainer.py`)
- Scaling laws (`scaling_laws/scaling_measurement_framework.py`)

## License

MIT License
