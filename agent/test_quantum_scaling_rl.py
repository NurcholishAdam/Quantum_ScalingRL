#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for Quantum-Scaling RL Hybrid Agent
"""
import unittest
import numpy as np
from quantum_scaling_rl_hybrid import (
    QuantumScalingRLHybrid,
    QuantumRLConfig,
    EditCycleResult
)


class TestQuantumScalingRLHybrid(unittest.TestCase):
    """Test cases for the hybrid agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = QuantumRLConfig(
            qaoa_depth=1,  # Reduced for faster tests
            qsvm_feature_dim=8,
            qec_code_distance=3,
            learning_rate=1e-5,
            batch_size=4,
            backends=['ibm', 'russian']
        )
        self.agent = QuantumScalingRLHybrid(self.config)
        
        # Sample corpus
        self.corpus = [
            {
                'id': f'doc_{i}',
                'lang': np.random.choice(['en', 'ru', 'zh']),
                'text': f'Document {i}',
                'embedding': np.random.randn(768)
            }
            for i in range(10)
        ]
        
        # Sample edit
        self.edit = {
            'id': 'test_edit_1',
            'language': 'en',
            'start_node': 'doc_0',
            'end_node': 'doc_5',
            'embedding': np.random.randn(768),
            'label': 1,
            'text': 'Test edit'
        }
    
    def test_initialization(self):
        """Test agent initialization"""
        self.assertIsNotNone(self.agent)
        self.assertEqual(self.agent.config.qaoa_depth, 1)
        self.assertEqual(len(self.agent.config.backends), 2)
        self.assertEqual(len(self.agent.backend_performance), 2)
    
    def test_quantum_optimize_edit(self):
        """Test quantum optimization step"""
        result = self.agent.quantum_optimize_edit(
            self.edit,
            self.corpus,
            'ibm'
        )
        
        self.assertIn('optimized_edit', result)
        self.assertIn('quantum_metrics', result)
        
        metrics = result['quantum_metrics']
        self.assertIn('qaoa_coherence', metrics)
        self.assertIn('qec_logical_error_rate', metrics)
        self.assertIn('total_quantum_time_ms', metrics)
    
    def test_rlhf_adapt_backend(self):
        """Test RLHF adaptation step"""
        quantum_metrics = {
            'qec_logical_error_rate': 0.05,
            'qaoa_latency_ms': 50,
            'qsvm_valid_prob': 0.8
        }
        
        result = self.agent.rlhf_adapt_backend(
            self.edit,
            quantum_metrics,
            'ibm'
        )
        
        self.assertIn('reward', result)
        self.assertIn('rl_metrics', result)
        self.assertIn('backend_recommendation', result)
        
        self.assertGreater(result['reward'], 0)
        self.assertLess(result['reward'], 1)
    
    def test_scaling_rl_budget(self):
        """Test scaling RL budgeting step"""
        quantum_metrics = {'total_quantum_time_ms': 100}
        rl_metrics = {'final_reward': 0.7, 'adaptation_time_ms': 50}
        
        result = self.agent.scaling_rl_budget(
            self.edit,
            quantum_metrics,
            rl_metrics
        )
        
        self.assertIn('scaling_metrics', result)
        self.assertIn('compute_budget_remaining', result)
        
        metrics = result['scaling_metrics']
        self.assertIn('optimal_batch_size', metrics)
        self.assertIn('compute_efficiency', metrics)
    
    def test_run_edit_cycle(self):
        """Test complete edit cycle"""
        # Train QSVM first
        training_edits = [
            {
                'embedding': np.random.randn(768),
                'label': np.random.choice([0, 1])
            }
            for _ in range(20)
        ]
        X_train = np.array([e['embedding'] for e in training_edits])
        y_train = np.array([e['label'] for e in training_edits])
        X_train = self.agent.qsvm_classifier._reduce_dimensions(X_train)
        X_train = self.agent.qsvm_classifier.scaler.fit_transform(X_train)
        self.agent.qsvm_classifier.train_qsvm(X_train, y_train)
        
        # Run cycle
        result = self.agent.run_edit_cycle(self.edit, self.corpus, 'ibm')
        
        self.assertIsInstance(result, EditCycleResult)
        self.assertEqual(result.backend, 'ibm')
        self.assertIsNotNone(result.quantum_metrics)
        self.assertIsNotNone(result.rl_metrics)
        self.assertIsNotNone(result.scaling_metrics)
    
    def test_backend_recommendation(self):
        """Test backend recommendation logic"""
        # Initially should return default
        backend = self.agent._recommend_backend(self.edit)
        self.assertIn(backend, self.config.backends)
        
        # After learning, should use heuristics
        self.agent.learned_heuristics['en'] = {
            'preferred_backend': 'russian',
            'avg_reward': 0.8,
            'edit_count': 5
        }
        
        backend = self.agent._recommend_backend(self.edit)
        self.assertEqual(backend, 'russian')
    
    def test_performance_trend_calculation(self):
        """Test performance trend calculation"""
        # Insufficient data
        trend = self.agent._calculate_performance_trend()
        self.assertEqual(trend, "insufficient_data")
        
        # Add improving trend
        for i in range(5):
            self.agent.edit_history.append(
                EditCycleResult(
                    edit_id=f'edit_{i}',
                    backend='ibm',
                    quantum_metrics={},
                    rl_metrics={},
                    scaling_metrics={},
                    performance_delta=0.2,
                    timestamp='2024-01-01'
                )
            )
        
        trend = self.agent._calculate_performance_trend()
        self.assertEqual(trend, "improving")
    
    def test_statistics_generation(self):
        """Test statistics generation"""
        # Run a few cycles
        for i in range(3):
            edit = self.edit.copy()
            edit['id'] = f'edit_{i}'
            self.agent.run_edit_cycle(edit, self.corpus)
        
        stats = self.agent.get_statistics()
        
        self.assertIn('total_edits', stats)
        self.assertIn('backend_performance', stats)
        self.assertIn('learned_heuristics', stats)
        self.assertIn('performance_trend', stats)
        self.assertIn('quantum_stats', stats)
        self.assertIn('recent_performance', stats)
        
        self.assertEqual(stats['total_edits'], 3)
    
    def test_feedback_loop_update(self):
        """Test feedback loop update"""
        result = EditCycleResult(
            edit_id='test_edit',
            backend='ibm',
            quantum_metrics={'qaoa_coherence': 0.8},
            rl_metrics={'final_reward': 0.7},
            scaling_metrics={'compute_efficiency': 0.6},
            performance_delta=0.1,
            timestamp='2024-01-01'
        )
        
        feedback = self.agent.feedback_loop_update(result)
        
        self.assertIn('reflection', feedback)
        self.assertIn('curator_updates', feedback)
        self.assertIn('retrain_signal', feedback)
        self.assertIn('feedback_loop_time_ms', feedback)
    
    def test_batch_size_scaling(self):
        """Test batch size scaling logic"""
        # Small edit
        small_edit = {'text': 'short'}
        quantum_metrics = {}
        rl_metrics = {'final_reward': 0.5}
        
        result = self.agent.scaling_rl_budget(small_edit, quantum_metrics, rl_metrics)
        small_batch = result['scaling_metrics']['optimal_batch_size']
        
        # Large edit
        large_edit = {'text': 'x' * 10000}
        result = self.agent.scaling_rl_budget(large_edit, quantum_metrics, rl_metrics)
        large_batch = result['scaling_metrics']['optimal_batch_size']
        
        # Larger edits should get larger batches
        self.assertGreaterEqual(large_batch, small_batch)
    
    def test_reward_shaping(self):
        """Test reward shaping for multilingual edits"""
        # Add some history
        self.agent.backend_performance['ibm'] = [0.5, 0.6, 0.7, 0.5, 0.6]
        self.agent.learned_heuristics['en'] = {
            'preferred_backend': 'ibm',
            'avg_reward': 0.6,
            'edit_count': 5
        }
        
        quantum_metrics = {}
        rl_metrics = {'final_reward': 0.7}
        
        result = self.agent.scaling_rl_budget(self.edit, quantum_metrics, rl_metrics)
        
        self.assertIn('shaped_reward', result['scaling_metrics'])
        self.assertIn('reward_variance', result['scaling_metrics'])
    
    def test_kl_penalty_calculation(self):
        """Test KL penalty for backend switching"""
        # Add history
        self.agent.backend_performance['ibm'] = [0.6] * 10
        
        quantum_metrics = {
            'qec_logical_error_rate': 0.05,
            'qaoa_latency_ms': 50,
            'qsvm_valid_prob': 0.8
        }
        
        result = self.agent.rlhf_adapt_backend(self.edit, quantum_metrics, 'ibm')
        
        self.assertIn('kl_penalty', result['rl_metrics'])
        self.assertGreaterEqual(result['rl_metrics']['kl_penalty'], 0)


class TestQuantumRLConfig(unittest.TestCase):
    """Test configuration class"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = QuantumRLConfig()
        
        self.assertEqual(config.qaoa_depth, 2)
        self.assertEqual(config.qsvm_feature_dim, 8)
        self.assertEqual(config.qec_code_distance, 5)
        self.assertEqual(config.learning_rate, 1e-5)
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(len(config.backends), 2)
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = QuantumRLConfig(
            qaoa_depth=3,
            qsvm_feature_dim=16,
            backends=['ibm', 'russian', 'google']
        )
        
        self.assertEqual(config.qaoa_depth, 3)
        self.assertEqual(config.qsvm_feature_dim, 16)
        self.assertEqual(len(config.backends), 3)


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
