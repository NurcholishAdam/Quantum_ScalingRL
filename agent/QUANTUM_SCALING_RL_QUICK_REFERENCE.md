# Quantum-Scaling RL Hybrid Agent - Quick Reference

## Installation

```bash
# Install dependencies
pip install qiskit qiskit-machine-learning torch transformers numpy scikit-learn networkx

# Navigate to agent directory
cd agent
```

## Quick Start

```python
from quantum_scaling_rl_hybrid import QuantumScalingRLHybrid, QuantumRLConfig
import numpy as np

# 1. Initialize agent
agent = QuantumScalingRLHybrid()

# 2. Prepare corpus
corpus = [
    {
        'id': 'doc_1',
        'lang': 'en',
        'text': 'Sample text',
        'embedding': np.random.randn(768)
    }
]

# 3. Prepare edit
edit = {
    'id': 'edit_1',
    'language': 'en',
    'start_node': 'doc_1',
    'end_node': 'doc_2',
    'embedding': np.random.randn(768),
    'label': 1
}

# 4. Run edit cycle
result = agent.run_edit_cycle(edit, corpus)

# 5. View results
print(f"Performance: {result.performance_delta:.3f}")
print(f"Backend: {result.backend}")
```

## Key Components

### 1. Quantum Optimization
```python
# QAOA semantic traversal
quantum_result = agent.quantum_optimize_edit(edit, corpus, 'ibm')
print(quantum_result['quantum_metrics']['qaoa_coherence'])

# QSVM hallucination detection
# (requires trained classifier)
print(quantum_result['quantum_metrics']['qsvm_valid_prob'])

# QEC error correction
print(quantum_result['quantum_metrics']['qec_logical_error_rate'])
```

### 2. RLHF Adaptation
```python
# Adapt backend based on feedback
rlhf_result = agent.rlhf_adapt_backend(edit, quantum_metrics, 'ibm')
print(f"Reward: {rlhf_result['reward']:.3f}")
print(f"Recommended: {rlhf_result['backend_recommendation']}")
```

### 3. Scaling RL Budgeting
```python
# Optimize compute allocation
scaling_result = agent.scaling_rl_budget(edit, quantum_metrics, rl_metrics)
print(f"Optimal batch size: {scaling_result['scaling_metrics']['optimal_batch_size']}")
print(f"Compute efficiency: {scaling_result['scaling_metrics']['compute_efficiency']:.3f}")
```

### 4. Statistics
```python
# Get comprehensive statistics
stats = agent.get_statistics()
print(f"Total edits: {stats['total_edits']}")
print(f"Trend: {stats['performance_trend']}")
print(f"Backend performance: {stats['backend_performance']}")
```

## Configuration Options

```python
config = QuantumRLConfig(
    # Quantum parameters
    qaoa_depth=2,              # QAOA circuit depth
    qsvm_feature_dim=8,        # QSVM feature dimension
    qec_code_distance=5,       # Surface code distance
    
    # RL parameters
    learning_rate=1e-5,        # Learning rate
    batch_size=8,              # Base batch size
    ppo_epochs=4,              # PPO epochs
    clip_epsilon=0.2,          # PPO clipping
    kl_coef=0.1,               # KL coefficient
    
    # Scaling parameters
    compute_budget=1.0,        # Compute budget
    batch_size_scaling=True,   # Enable batch scaling
    reward_shaping=True,       # Enable reward shaping
    
    # Backends
    backends=['ibm', 'russian']
)

agent = QuantumScalingRLHybrid(config)
```

## Training QSVM Classifier

```python
# Prepare training data
training_edits = [
    {'embedding': np.random.randn(768), 'label': 0},  # hallucinated
    {'embedding': np.random.randn(768), 'label': 1},  # valid
    # ... more edits
]

X_train = np.array([e['embedding'] for e in training_edits])
y_train = np.array([e['label'] for e in training_edits])

# Preprocess and train
X_train = agent.qsvm_classifier._reduce_dimensions(X_train)
X_train = agent.qsvm_classifier.scaler.fit_transform(X_train)
agent.qsvm_classifier.train_qsvm(X_train, y_train)
```

## Running the Demo

```bash
python demo_quantum_scaling_rl.py
```

Output includes:
- Agent initialization
- Corpus generation
- QSVM training
- 15 edit cycles with metrics
- Final statistics and learned heuristics

## Running Tests

```bash
python test_quantum_scaling_rl.py
```

Tests cover:
- Initialization
- Quantum optimization
- RLHF adaptation
- Scaling budgeting
- Complete edit cycles
- Backend recommendation
- Performance trends
- Statistics generation

## Key Metrics

### Quantum Metrics
- `qaoa_coherence`: Semantic coherence (0-1)
- `qaoa_latency_ms`: Optimization time
- `qsvm_valid_prob`: Valid edit probability (0-1)
- `qec_logical_error_rate`: Error rate (0-1)
- `qec_success`: Correction success (bool)

### RL Metrics
- `edit_reliability_delta`: Reliability (0-1)
- `latency_reduction`: Latency improvement (0-1)
- `contributor_agreement_score`: Agreement (0-1)
- `final_reward`: Total reward (0-1)
- `kl_penalty`: Backend switching penalty (≥0)

### Scaling Metrics
- `optimal_batch_size`: Computed batch size
- `compute_efficiency`: Reward per second
- `shaped_reward`: Variance-adjusted reward
- `estimated_gpu_time_to_target_ms`: Time to target

## Common Patterns

### Multi-Language Processing
```python
languages = ['en', 'ru', 'zh', 'es', 'fr']
for lang in languages:
    edit = {'language': lang, ...}
    result = agent.run_edit_cycle(edit, corpus)
    print(f"{lang}: {result.performance_delta:.3f}")
```

### Backend Comparison
```python
backends = ['ibm', 'russian']
for backend in backends:
    result = agent.run_edit_cycle(edit, corpus, backend)
    print(f"{backend}: {result.rl_metrics['final_reward']:.3f}")
```

### Performance Monitoring
```python
for i in range(100):
    result = agent.run_edit_cycle(edit, corpus)
    if i % 10 == 0:
        stats = agent.get_statistics()
        print(f"Cycle {i}: Trend = {stats['performance_trend']}")
```

## Troubleshooting

### QSVM Not Trained
```python
# Error: Model not trained
# Solution: Train before using
agent.qsvm_classifier.train_qsvm(X_train, y_train)
```

### Low Performance
```python
# Check statistics
stats = agent.get_statistics()
print(stats['backend_performance'])

# Adjust configuration
config.learning_rate = 5e-6  # Lower learning rate
config.kl_coef = 0.05        # Reduce KL penalty
```

### High Compute Time
```python
# Reduce quantum parameters
config.qaoa_depth = 1
config.qec_code_distance = 3

# Disable scaling features
config.batch_size_scaling = False
config.reward_shaping = False
```

## Integration Examples

### With Research Agent
```python
from agent.research_agent import ResearchAgent

research_agent = ResearchAgent()
research_agent.quantum_rl_module = agent
```

### With LIMIT-GRAPH
```python
from extensions.LIMIT-GRAPH.agents.graph_reasoner import GraphReasoner

reasoner = GraphReasoner()
reasoner.quantum_optimizer = agent.qaoa_traversal
```

### With Semantic Graph
```python
from semantic_graph.ai_research_agent_integration import SemanticGraphIntegration

integration = SemanticGraphIntegration()
integration.quantum_rl_agent = agent
```

## Performance Tips

1. **Batch Processing**: Process multiple edits together
2. **Caching**: Cache QAOA results for similar paths
3. **Parallel Backends**: Run multiple backends in parallel
4. **Incremental Training**: Update QSVM incrementally
5. **Heuristic Warmup**: Pre-populate heuristics from historical data

## Next Steps

1. Read full documentation: `QUANTUM_SCALING_RL_HYBRID_DOCUMENTATION.md`
2. Run demo: `python demo_quantum_scaling_rl.py`
3. Run tests: `python test_quantum_scaling_rl.py`
4. Integrate with your system
5. Monitor performance and adjust configuration

## Support

For issues or questions:
- Check documentation
- Review test cases
- Examine demo code
- Inspect statistics output
