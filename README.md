# Quantum-Scaling RL Hybrid Agent

## Overview

The Quantum-Scaling RL Hybrid Agent integrates quantum optimization modules with reinforcement learning and scaling laws to create a self-improving system for multilingual semantic graph editing. The agent combines:

1. **Quantum Optimization**: QAOA traversal, QSVM hallucination detection, QEC error correction
2. **RLHF Adaptation**: Reinforcement learning for backend selection and heuristic learning
3. **ScalingRL Budgeting**: Compute-efficient resource allocation based on scaling laws
4. **Feedback Loop**: Self-improving cycle with reflector, curator, and retraining

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Quantum-Scaling RL Hybrid                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐      ┌──────────────────┐           │
│  │ Quantum Modules  │      │  RLHF Adaptation │           │
│  ├──────────────────┤      ├──────────────────┤           │
│  │ • QAOA Traversal │      │ • Reward Model   │           │
│  │ • QSVM Classifier│──────│ • PPO Training   │           │
│  │ • QEC Extension  │      │ • KL Regulation  │           │
│  └──────────────────┘      └──────────────────┘           │
│           │                         │                      │
│           └─────────┬───────────────┘                      │
│                     │                                      │
│           ┌─────────▼──────────┐                          │
│           │ ScalingRL Budgeting│                          │
│           ├────────────────────┤                          │
│           │ • Batch Sizing     │                          │
│           │ • Reward Shaping   │                          │
│           │ • Compute Tracking │                          │
│           └─────────┬──────────┘                          │
│                     │                                      │
│           ┌─────────▼──────────┐                          │
│           │   Feedback Loop    │                          │
│           ├────────────────────┤                          │
│           │ • Reflector        │                          │
│           │ • Curator          │                          │
│           │ • RL Retraining    │                          │
│           └────────────────────┘                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Quantum Optimization

#### QAOA Semantic Traversal
- **Purpose**: Optimize semantic graph paths for multilingual citation walks
- **Input**: Corpus with embeddings, start/end nodes
- **Output**: Optimized path with coherence score
- **Metrics**: Coherence score, latency, cross-lingual detection

#### QSVM Hallucination Detection
- **Purpose**: Kernel-based classification of valid vs hallucinated edits
- **Input**: Edit embeddings
- **Output**: Hallucination probability
- **Metrics**: AUROC, precision, recall, F1 score

#### QEC Surface Code Extension
- **Purpose**: Quantum error correction for fault-tolerant edits
- **Input**: Edit data
- **Output**: Corrected edit with syndrome information
- **Metrics**: Logical error rate, correction success rate

### 2. RLHF Adaptation

#### Reward Signals
- **Edit Reliability Delta**: `1.0 - logical_error_rate`
- **Latency Reduction**: `1.0 / (1.0 + latency_ms / 100)`
- **Contributor Agreement Score**: QSVM valid probability

#### KL-Regularized PPO
- Base reward combines three signals (weighted 0.4, 0.3, 0.3)
- KL penalty prevents excessive backend switching
- Final reward: `base_reward - kl_coef * |reward - historical_mean|`

#### Heuristic Learning
- Learns preferred backends per language
- Tracks average rewards and edit counts
- Updates preferences based on performance

### 3. ScalingRL Budgeting

#### Batch Size Scaling
- Proportional to model size: `batch_size * sqrt(model_size_proxy)`
- Based on "The Art of Scaling RL Compute" insights
- Optimizes throughput vs quality tradeoff

#### Low-Variance Reward Shaping
- Reduces variance for multilingual edits
- Shaped reward: `reward / (1.0 + variance)`
- Stabilizes training across languages

#### Compute Efficiency Tracking
- Monitors total quantum + RL time
- Calculates efficiency: `reward / compute_time`
- Predicts GPU time to reach performance targets

### 4. Feedback Loop

#### Reflector Module
- Analyzes performance delta
- Evaluates quantum, RL, and scaling quality
- Identifies improvement opportunities

#### Curator Module
- Updates learned heuristics
- Reinforces successful backends
- Maintains language-specific preferences

#### RL Agent Retraining
- Triggers retraining every N edits
- Incorporates new feedback
- Adapts to changing patterns

## Usage

### Basic Usage

```python
from quantum_scaling_rl_hybrid import QuantumScalingRLHybrid, QuantumRLConfig

# Initialize agent
config = QuantumRLConfig(
    qaoa_depth=2,
    qsvm_feature_dim=8,
    qec_code_distance=5,
    learning_rate=1e-5,
    batch_size=8,
    kl_coef=0.1,
    backends=['ibm', 'russian']
)

agent = QuantumScalingRLHybrid(config)

# Prepare data
corpus = [
    {
        'id': 'doc_1',
        'lang': 'en',
        'text': 'Sample text',
        'embedding': np.random.randn(768)
    },
    # ... more documents
]

edit = {
    'id': 'edit_1',
    'language': 'en',
    'start_node': 'doc_1',
    'end_node': 'doc_2',
    'embedding': np.random.randn(768),
    'label': 1  # 0=hallucinated, 1=valid
}

# Run edit cycle
result = agent.run_edit_cycle(edit, corpus)

# Access results
print(f"Performance Delta: {result.performance_delta}")
print(f"Backend: {result.backend}")
print(f"Quantum Metrics: {result.quantum_metrics}")
print(f"RL Metrics: {result.rl_metrics}")
print(f"Scaling Metrics: {result.scaling_metrics}")
```

### Training QSVM Classifier

```python
# Prepare training data
training_edits = [...]  # List of edits with embeddings and labels
X_train = np.array([e['embedding'] for e in training_edits])
y_train = np.array([e['label'] for e in training_edits])

# Train classifier
X_train = agent.qsvm_classifier._reduce_dimensions(X_train)
X_train = agent.qsvm_classifier.scaler.fit_transform(X_train)
agent.qsvm_classifier.train_qsvm(X_train, y_train)
```

### Getting Statistics

```python
stats = agent.get_statistics()

print(f"Total Edits: {stats['total_edits']}")
print(f"Performance Trend: {stats['performance_trend']}")
print(f"Backend Performance: {stats['backend_performance']}")
print(f"Learned Heuristics: {stats['learned_heuristics']}")
print(f"QEC Stats: {stats['quantum_stats']}")
```

## Configuration

### QuantumRLConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `qaoa_depth` | int | 2 | QAOA circuit depth (p parameter) |
| `qsvm_feature_dim` | int | 8 | Feature dimension for QSVM (power of 2) |
| `qec_code_distance` | int | 5 | Surface code distance (3, 5, or 7) |
| `learning_rate` | float | 1e-5 | RL learning rate |
| `batch_size` | int | 8 | Base batch size for training |
| `ppo_epochs` | int | 4 | PPO update epochs |
| `clip_epsilon` | float | 0.2 | PPO clipping parameter |
| `kl_coef` | float | 0.1 | KL divergence coefficient |
| `compute_budget` | float | 1.0 | Total compute budget |
| `batch_size_scaling` | bool | True | Enable batch size scaling |
| `reward_shaping` | bool | True | Enable reward shaping |
| `backends` | List[str] | ['ibm', 'russian'] | Available quantum backends |

## Performance Metrics

### Quantum Metrics
- **QAOA Coherence**: Semantic coherence of optimized path (0-1)
- **QAOA Latency**: Path optimization time (ms)
- **QSVM Hallucination Prob**: Probability edit is hallucinated (0-1)
- **QSVM Valid Prob**: Probability edit is valid (0-1)
- **QEC Syndromes**: Number of error syndromes detected
- **QEC Corrections**: Number of corrections applied
- **QEC Logical Error Rate**: Post-correction error rate (0-1)
- **QEC Success**: Whether correction succeeded (bool)

### RL Metrics
- **Edit Reliability Delta**: Reliability improvement (0-1)
- **Latency Reduction**: Normalized latency improvement (0-1)
- **Contributor Agreement Score**: Agreement with human feedback (0-1)
- **Base Reward**: Combined reward before KL penalty (0-1)
- **KL Penalty**: Penalty for backend switching (≥0)
- **Final Reward**: Total reward after penalties (0-1)

### Scaling Metrics
- **Optimal Batch Size**: Computed optimal batch size
- **Reward Variance**: Historical reward variance
- **Shaped Reward**: Variance-adjusted reward
- **Compute Efficiency**: Reward per second
- **Total Compute Time**: Total processing time (ms)
- **Estimated GPU Time to Target**: Predicted time to reach target performance (ms)

## Self-Improving Loop

The agent implements a continuous improvement cycle:

1. **Edit Cycle**: Process edit with quantum optimization
2. **Adaptation**: Learn from feedback and adjust backends
3. **Budgeting**: Optimize compute allocation
4. **Reflection**: Analyze performance and update heuristics
5. **Repeat**: Next edit benefits from learned patterns

### Learning Dynamics

- **Backend Selection**: Learns which backends work best for each language
- **Heuristic Refinement**: Continuously updates edit strategies
- **Compute Optimization**: Adapts batch sizes and resource allocation
- **Performance Tracking**: Monitors trends and triggers retraining

## Benchmarking

### IBM vs Russian Backend Comparison

The agent tracks performance across backends:

```python
stats = agent.get_statistics()
for backend, perf in stats['backend_performance'].items():
    print(f"{backend}: {perf['mean_reward']:.3f} ± {perf['std_reward']:.3f}")
```

### Cross-Lingual Performance

Per-language heuristics show adaptation:

```python
for lang, heuristic in stats['learned_heuristics'].items():
    print(f"{lang}: {heuristic['preferred_backend']} ({heuristic['avg_reward']:.3f})")
```

## Integration with Existing Systems

### With AI Research Agent

```python
from agent.quantum_scaling_rl_hybrid import create_hybrid_agent
from agent.research_agent import ResearchAgent

# Create hybrid agent
hybrid = create_hybrid_agent()

# Integrate with research agent
research_agent = ResearchAgent()
research_agent.quantum_rl_module = hybrid
```

### With LIMIT-GRAPH

```python
from extensions.LIMIT-GRAPH.agents.graph_reasoner import GraphReasoner

# Use hybrid agent for graph optimization
reasoner = GraphReasoner()
reasoner.quantum_optimizer = hybrid.qaoa_traversal
reasoner.hallucination_detector = hybrid.qsvm_classifier
```

## Running the Demo

```bash
cd agent
python demo_quantum_scaling_rl.py
```

The demo will:
1. Initialize the hybrid agent
2. Generate sample multilingual corpus
3. Train QSVM classifier
4. Run 15 edit cycles
5. Display comprehensive statistics
6. Show learned heuristics and performance trends

## Future Enhancements

1. **Advanced RL Algorithms**: DPO, REINFORCE variants
2. **Multi-Backend Ensembles**: Combine predictions from multiple backends
3. **Adaptive QEC**: Dynamic code distance based on error rates
4. **Hierarchical RL**: Multi-level policy optimization
5. **Transfer Learning**: Share heuristics across related languages
6. **Real-Time Adaptation**: Online learning during inference

## References

- QAOA: Farhi et al., "A Quantum Approximate Optimization Algorithm"
- QSVM: Havlíček et al., "Supervised learning with quantum-enhanced feature spaces"
- Surface Codes: Fowler et al., "Surface codes: Towards practical large-scale quantum computation"
- PPO: Schulman et al., "Proximal Policy Optimization Algorithms"
- Scaling Laws: Hilton et al., "The Art of Scaling RL Compute"

## License

MIT License - See LICENSE file for details
