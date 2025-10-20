#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: Quantum-Scaling RL Hybrid Agent
Demonstrates the self-improving loop with quantum optimization and RL adaptation
"""
import numpy as np
from quantum_scaling_rl_hybrid import QuantumScalingRLHybrid, QuantumRLConfig


def generate_sample_corpus(num_docs: int = 20) -> list:
    """Generate sample multilingual corpus"""
    languages = ['en', 'ru', 'zh', 'es', 'fr']
    corpus = []
    
    for i in range(num_docs):
        corpus.append({
            'id': f'doc_{i}',
            'lang': np.random.choice(languages),
            'text': f'Sample document {i} with semantic content',
            'embedding': np.random.randn(768)  # Simulated embedding
        })
    
    return corpus


def generate_sample_edit(edit_id: int, corpus: list) -> dict:
    """Generate sample REPAIR edit"""
    doc_ids = [doc['id'] for doc in corpus]
    
    return {
        'id': f'edit_{edit_id}',
        'language': np.random.choice(['en', 'ru', 'zh', 'es', 'fr']),
        'start_node': np.random.choice(doc_ids),
        'end_node': np.random.choice(doc_ids),
        'embedding': np.random.randn(768),
        'label': np.random.choice([0, 1]),  # 0=hallucinated, 1=valid
        'text': f'Edit {edit_id}: Modify semantic relationship'
    }


def main():
    print("=" * 80)
    print("Quantum-Scaling RL Hybrid Agent Demo")
    print("=" * 80)
    print()
    
    # Initialize hybrid agent
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
    print("✓ Hybrid agent initialized")
    print(f"  - QAOA depth: {config.qaoa_depth}")
    print(f"  - QSVM feature dim: {config.qsvm_feature_dim}")
    print(f"  - QEC code distance: {config.qec_code_distance}")
    print(f"  - Backends: {config.backends}")
    print()
    
    # Generate sample data
    corpus = generate_sample_corpus(20)
    print(f"✓ Generated corpus with {len(corpus)} documents")
    print(f"  - Languages: {set(doc['lang'] for doc in corpus)}")
    print()
    
    # Train QSVM classifier (simplified)
    print("Training QSVM classifier...")
    training_edits = [generate_sample_edit(i, corpus) for i in range(50)]
    X_train = np.array([e['embedding'] for e in training_edits])
    y_train = np.array([e['label'] for e in training_edits])
    
    X_train = agent.qsvm_classifier._reduce_dimensions(X_train)
    X_train = agent.qsvm_classifier.scaler.fit_transform(X_train)
    agent.qsvm_classifier.train_qsvm(X_train, y_train)
    print("✓ QSVM classifier trained")
    print()
    
    # Run edit cycles
    print("=" * 80)
    print("Running Edit Cycles")
    print("=" * 80)
    print()
    
    num_cycles = 15
    for i in range(num_cycles):
        print(f"--- Edit Cycle {i+1}/{num_cycles} ---")
        
        # Generate edit
        edit = generate_sample_edit(i, corpus)
        print(f"Edit ID: {edit['id']}, Language: {edit['language']}")
        
        # Run cycle
        result = agent.run_edit_cycle(edit, corpus)
        
        # Display results
        print(f"Backend: {result.backend}")
        print(f"Performance Delta: {result.performance_delta:+.3f}")
        print(f"Quantum Metrics:")
        print(f"  - QAOA Coherence: {result.quantum_metrics.get('qaoa_coherence', 0):.3f}")
        print(f"  - QEC Logical Error: {result.quantum_metrics.get('qec_logical_error_rate', 0):.4f}")
        print(f"  - QSVM Valid Prob: {result.quantum_metrics.get('qsvm_valid_prob', 0):.3f}")
        print(f"RL Metrics:")
        print(f"  - Final Reward: {result.rl_metrics.get('final_reward', 0):.3f}")
        print(f"  - Edit Reliability: {result.rl_metrics.get('edit_reliability_delta', 0):.3f}")
        print(f"  - KL Penalty: {result.rl_metrics.get('kl_penalty', 0):.4f}")
        print(f"Scaling Metrics:")
        print(f"  - Compute Efficiency: {result.scaling_metrics.get('compute_efficiency', 0):.3f}")
        print(f"  - Optimal Batch Size: {result.scaling_metrics.get('optimal_batch_size', 0)}")
        print()
    
    # Display final statistics
    print("=" * 80)
    print("Final Statistics")
    print("=" * 80)
    print()
    
    stats = agent.get_statistics()
    
    print(f"Total Edits: {stats['total_edits']}")
    print(f"Performance Trend: {stats['performance_trend']}")
    print()
    
    print("Backend Performance:")
    for backend, perf in stats['backend_performance'].items():
        print(f"  {backend}:")
        print(f"    - Mean Reward: {perf['mean_reward']:.3f}")
        print(f"    - Std Reward: {perf['std_reward']:.3f}")
        print(f"    - Edit Count: {perf['edit_count']}")
    print()
    
    print("Learned Heuristics:")
    for lang, heuristic in stats['learned_heuristics'].items():
        print(f"  {lang}:")
        print(f"    - Preferred Backend: {heuristic.get('preferred_backend', 'N/A')}")
        print(f"    - Avg Reward: {heuristic.get('avg_reward', 0):.3f}")
        print(f"    - Edit Count: {heuristic.get('edit_count', 0)}")
    print()
    
    print("QEC Statistics:")
    qec_stats = stats['quantum_stats']
    print(f"  - Total Edits: {qec_stats.get('total_edits', 0)}")
    print(f"  - Syndromes Detected: {qec_stats.get('syndromes_detected', 0)}")
    print(f"  - Corrections Applied: {qec_stats.get('corrections_applied', 0)}")
    print(f"  - Successful Corrections: {qec_stats.get('successful_corrections', 0)}")
    if 'correction_rate' in qec_stats:
        print(f"  - Correction Rate: {qec_stats['correction_rate']:.2%}")
    print()
    
    print("Recent Performance (last 5 edits):")
    for edit_info in stats['recent_performance'][-5:]:
        print(f"  {edit_info['edit_id']}: {edit_info['performance_delta']:+.3f} ({edit_info['backend']})")
    print()
    
    print("=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    print()
    print("Key Insights:")
    print("1. Quantum modules optimize semantic paths and detect hallucinations")
    print("2. RLHF adapts backend selection based on multilingual feedback")
    print("3. Scaling laws optimize compute budgets and batch sizes")
    print("4. Feedback loop creates self-improving behavior")
    print()
    print("The agent learns which backends work best for each language")
    print("and continuously improves edit quality through the RL loop.")


if __name__ == '__main__':
    main()
