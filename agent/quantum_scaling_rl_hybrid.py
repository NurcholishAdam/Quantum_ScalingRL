# -*- coding: utf-8 -*-
"""
Quantum-Scaling RL Hybrid Agent
Integrates quantum optimization (QAOA, QSVM, QEC) with scaling RL for self-improving multilingual edits
"""
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Quantum modules
import sys
sys.path.append('quantum_integration/quantum limit graph v2.3.0/src')
from graph.qaoa_traversal import QAOASemanticTraversal
from evaluation.qsvm_hallucination import QSVMHallucinationClassifier
sys.path.append('quantum_integration/quantum-limit-graph-v2.4.0/src')
from agent.repair_qec_extension import REPAIRQECExtension

# RLHF modules
from rlhf.reward_model import RewardModelManager
from rlhf.rl_trainer import RLTrainingConfig

# Scaling laws
from scaling_laws.scaling_measurement_framework import ScalingLawMeasurement, ScalingDimension


@dataclass
class QuantumRLConfig:
    """Configuration for Quantum-Scaling RL Hybrid"""
    # Quantum parameters
    qaoa_depth: int = 2
    qsvm_feature_dim: int = 8
    qec_code_distance: int = 5
    
    # RL parameters
    learning_rate: float = 1e-5
    batch_size: int = 8
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    kl_coef: float = 0.1
    
    # Scaling parameters
    compute_budget: float = 1.0
    batch_size_scaling: bool = True
    reward_shaping: bool = True
    
    # Backend parameters
    backends: List[str] = None
    
    def __post_init__(self):
        if self.backends is None:
            self.backends = ['ibm', 'russian']


@dataclass
class EditCycleResult:
    """Result from one edit cycle"""
    edit_id: str
    backend: str
    quantum_metrics: Dict[str, float]
    rl_metrics: Dict[str, float]
    scaling_metrics: Dict[str, float]
    performance_delta: float
    timestamp: str


class QuantumScalingRLHybrid:
    """Hybrid agent integrating quantum optimization with scaling RL"""
    
    def __init__(self, config: QuantumRLConfig = None):
        self.config = config or QuantumRLConfig()
        self.logger = logging.getLogger("QuantumScalingRLHybrid")
        self.logger.setLevel(logging.INFO)
        
        # Initialize quantum modules
        self.qaoa_traversal = QAOASemanticTraversal(p=self.config.qaoa_depth)
        self.qsvm_classifier = QSVMHallucinationClassifier(feature_dimension=self.config.qsvm_feature_dim)
        self.qec_extension = REPAIRQECExtension(code_distance=self.config.qec_code_distance)
        
        # Initialize RLHF components
        self.reward_manager = RewardModelManager()
        self.rl_config = RLTrainingConfig(
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            ppo_epochs=self.config.ppo_epochs,
            clip_epsilon=self.config.clip_epsilon
        )
        
        # Initialize scaling measurement
        self.scaling_framework = ScalingLawMeasurement()
        
        # State tracking
        self.edit_history: List[EditCycleResult] = []
        self.backend_performance: Dict[str, List[float]] = {b: [] for b in self.config.backends}
        self.learned_heuristics: Dict[str, Any] = {}
        
        self.logger.info("Quantum-Scaling RL Hybrid Agent initialized")
    
    def quantum_optimize_edit(
        self,
        edit: Dict,
        corpus: List[Dict],
        backend: str
    ) -> Dict[str, Any]:
        """
        Step 1: Quantum Optimization
        Uses QAOA for semantic graph optimization, QSVM for hallucination detection, QEC for correction
        """
        start_time = time.time()
        quantum_metrics = {}
        
        # 1. QAOA Semantic Graph Optimization
        if 'start_node' in edit and 'end_node' in edit:
            traversal_result = self.qaoa_traversal.traverse_semantic_path(
                corpus,
                edit['start_node'],
                edit['end_node']
            )
            quantum_metrics['qaoa_coherence'] = traversal_result['coherence_score']
            quantum_metrics['qaoa_latency_ms'] = traversal_result['latency_ms']
            quantum_metrics['cross_lingual'] = traversal_result['cross_lingual']
            edit['optimized_path'] = traversal_result['path']
        
        # 2. QSVM Hallucination Detection
        if 'embedding' in edit and 'label' in edit:
            # Prepare for classification
            test_edits = [edit]
            X = np.array([e['embedding'] for e in test_edits])
            X = self.qsvm_classifier._reduce_dimensions(X)
            X = self.qsvm_classifier.scaler.transform(X) if hasattr(self.qsvm_classifier.scaler, 'mean_') else X
            
            # Predict hallucination
            if self.qsvm_classifier.model is not None:
                prediction = self.qsvm_classifier.predict(X)[0]
                proba = self.qsvm_classifier.predict_proba(X)[0]
                quantum_metrics['qsvm_hallucination_prob'] = proba[0]
                quantum_metrics['qsvm_valid_prob'] = proba[1]
                edit['hallucination_detected'] = prediction == 0
            else:
                quantum_metrics['qsvm_hallucination_prob'] = 0.0
                quantum_metrics['qsvm_valid_prob'] = 1.0
                edit['hallucination_detected'] = False
        
        # 3. QEC Surface Code Correction
        qec_result = self.qec_extension.apply_qec(edit, backend)
        quantum_metrics['qec_syndromes'] = len(qec_result.syndromes_detected)
        quantum_metrics['qec_corrections'] = len(qec_result.corrections_applied)
        quantum_metrics['qec_logical_error_rate'] = qec_result.logical_error_rate
        quantum_metrics['qec_success'] = qec_result.correction_success
        edit = qec_result.corrected_edit
        
        quantum_metrics['total_quantum_time_ms'] = (time.time() - start_time) * 1000
        
        return {
            'optimized_edit': edit,
            'quantum_metrics': quantum_metrics
        }
    
    def rlhf_adapt_backend(
        self,
        edit: Dict,
        quantum_metrics: Dict,
        backend: str
    ) -> Dict[str, Any]:
        """
        Step 2: RLHF Adaptation
        Uses RL to adapt backend selection and learn edit heuristics from feedback
        """
        start_time = time.time()
        rl_metrics = {}
        
        # Calculate reward signals
        edit_reliability = 1.0 - quantum_metrics.get('qec_logical_error_rate', 0.1)
        latency_reduction = 1.0 / (1.0 + quantum_metrics.get('qaoa_latency_ms', 100) / 100)
        contributor_agreement = quantum_metrics.get('qsvm_valid_prob', 0.5)
        
        # Combined reward with KL regularization
        base_reward = (
            0.4 * edit_reliability +
            0.3 * latency_reduction +
            0.3 * contributor_agreement
        )
        
        # KL penalty for backend switching
        kl_penalty = 0.0
        if self.backend_performance[backend]:
            historical_perf = np.mean(self.backend_performance[backend][-10:])
            kl_penalty = self.config.kl_coef * abs(base_reward - historical_perf)
        
        reward = base_reward - kl_penalty
        
        rl_metrics['edit_reliability_delta'] = edit_reliability
        rl_metrics['latency_reduction'] = latency_reduction
        rl_metrics['contributor_agreement_score'] = contributor_agreement
        rl_metrics['base_reward'] = base_reward
        rl_metrics['kl_penalty'] = kl_penalty
        rl_metrics['final_reward'] = reward
        
        # Update backend performance history
        self.backend_performance[backend].append(reward)
        
        # Learn edit heuristics
        language = edit.get('language', 'en')
        if language not in self.learned_heuristics:
            self.learned_heuristics[language] = {
                'preferred_backend': backend,
                'avg_reward': reward,
                'edit_count': 1
            }
        else:
            heuristic = self.learned_heuristics[language]
            heuristic['edit_count'] += 1
            heuristic['avg_reward'] = (
                (heuristic['avg_reward'] * (heuristic['edit_count'] - 1) + reward) /
                heuristic['edit_count']
            )
            # Update preferred backend if this one performs better
            if reward > heuristic['avg_reward']:
                heuristic['preferred_backend'] = backend
        
        rl_metrics['adaptation_time_ms'] = (time.time() - start_time) * 1000
        
        return {
            'reward': reward,
            'rl_metrics': rl_metrics,
            'backend_recommendation': self._recommend_backend(edit)
        }
    
    def scaling_rl_budget(
        self,
        edit: Dict,
        quantum_metrics: Dict,
        rl_metrics: Dict
    ) -> Dict[str, Any]:
        """
        Step 3: ScalingRL Budgeting
        Applies insights from scaling laws to optimize compute allocation
        """
        start_time = time.time()
        scaling_metrics = {}
        
        # Calculate model size proxy (based on edit complexity)
        edit_complexity = len(str(edit)) / 1000  # Rough proxy
        model_size_proxy = max(1.0, edit_complexity)
        
        # Batch size proportional to model size (scaling law insight)
        if self.config.batch_size_scaling:
            optimal_batch_size = int(self.config.batch_size * np.sqrt(model_size_proxy))
            scaling_metrics['optimal_batch_size'] = optimal_batch_size
        else:
            scaling_metrics['optimal_batch_size'] = self.config.batch_size
        
        # Low-variance reward shaping for multilingual edits
        if self.config.reward_shaping:
            language = edit.get('language', 'en')
            if language in self.learned_heuristics:
                historical_variance = np.var(self.backend_performance.get(
                    self.learned_heuristics[language]['preferred_backend'], [0.5]
                ))
                shaped_reward = rl_metrics['final_reward'] / (1.0 + historical_variance)
                scaling_metrics['reward_variance'] = historical_variance
                scaling_metrics['shaped_reward'] = shaped_reward
            else:
                scaling_metrics['shaped_reward'] = rl_metrics['final_reward']
        
        # Track compute efficiency
        total_compute_time = (
            quantum_metrics.get('total_quantum_time_ms', 0) +
            rl_metrics.get('adaptation_time_ms', 0)
        )
        compute_efficiency = rl_metrics['final_reward'] / (total_compute_time / 1000 + 1e-6)
        scaling_metrics['compute_efficiency'] = compute_efficiency
        scaling_metrics['total_compute_time_ms'] = total_compute_time
        
        # GPU time prediction for performance targets
        target_reward = 0.9
        current_reward = rl_metrics['final_reward']
        if current_reward < target_reward:
            # Estimate additional compute needed (simplified)
            reward_gap = target_reward - current_reward
            estimated_gpu_time = total_compute_time * (reward_gap / current_reward)
            scaling_metrics['estimated_gpu_time_to_target_ms'] = estimated_gpu_time
        else:
            scaling_metrics['estimated_gpu_time_to_target_ms'] = 0.0
        
        scaling_metrics['budgeting_time_ms'] = (time.time() - start_time) * 1000
        
        return {
            'scaling_metrics': scaling_metrics,
            'compute_budget_remaining': self.config.compute_budget - (total_compute_time / 1000)
        }
    
    def feedback_loop_update(
        self,
        edit_result: EditCycleResult
    ) -> Dict[str, Any]:
        """
        Step 4: Feedback Loop
        Reflector analyzes performance, curator updates heuristics, RL agent retrains
        """
        start_time = time.time()
        
        # Reflector: Analyze performance
        reflection = {
            'performance_delta': edit_result.performance_delta,
            'quantum_quality': np.mean(list(edit_result.quantum_metrics.values())),
            'rl_quality': edit_result.rl_metrics.get('final_reward', 0.5),
            'scaling_efficiency': edit_result.scaling_metrics.get('compute_efficiency', 0.5)
        }
        
        # Curator: Update heuristics
        backend = edit_result.backend
        if reflection['performance_delta'] > 0:
            # Positive performance - reinforce this backend
            if backend in self.learned_heuristics:
                self.learned_heuristics[backend]['reinforcement_count'] = \
                    self.learned_heuristics[backend].get('reinforcement_count', 0) + 1
        
        # RL Agent: Retrain signal (simplified - would trigger actual retraining)
        retrain_signal = {
            'should_retrain': len(self.edit_history) % 10 == 0,  # Retrain every 10 edits
            'new_feedback_count': 1,
            'performance_trend': self._calculate_performance_trend()
        }
        
        feedback_time = (time.time() - start_time) * 1000
        
        return {
            'reflection': reflection,
            'curator_updates': len(self.learned_heuristics),
            'retrain_signal': retrain_signal,
            'feedback_loop_time_ms': feedback_time
        }
    
    def run_edit_cycle(
        self,
        edit: Dict,
        corpus: List[Dict],
        backend: Optional[str] = None
    ) -> EditCycleResult:
        """
        Complete edit cycle: quantum optimize -> RLHF adapt -> scaling budget -> feedback loop
        """
        # Select backend
        if backend is None:
            backend = self._recommend_backend(edit)
        
        self.logger.info(f"Running edit cycle with backend: {backend}")
        
        # Step 1: Quantum Optimization
        quantum_result = self.quantum_optimize_edit(edit, corpus, backend)
        
        # Step 2: RLHF Adaptation
        rlhf_result = self.rlhf_adapt_backend(
            quantum_result['optimized_edit'],
            quantum_result['quantum_metrics'],
            backend
        )
        
        # Step 3: ScalingRL Budgeting
        scaling_result = self.scaling_rl_budget(
            quantum_result['optimized_edit'],
            quantum_result['quantum_metrics'],
            rlhf_result['rl_metrics']
        )
        
        # Calculate performance delta
        performance_delta = rlhf_result['reward'] - 0.5  # Baseline is 0.5
        
        # Create result
        cycle_result = EditCycleResult(
            edit_id=edit.get('id', f"edit_{len(self.edit_history)}"),
            backend=backend,
            quantum_metrics=quantum_result['quantum_metrics'],
            rl_metrics=rlhf_result['rl_metrics'],
            scaling_metrics=scaling_result['scaling_metrics'],
            performance_delta=performance_delta,
            timestamp=datetime.now().isoformat()
        )
        
        # Step 4: Feedback Loop
        feedback_result = self.feedback_loop_update(cycle_result)
        
        # Store history
        self.edit_history.append(cycle_result)
        
        self.logger.info(
            f"Edit cycle complete - Performance delta: {performance_delta:.3f}, "
            f"Backend: {backend}, Reward: {rlhf_result['reward']:.3f}"
        )
        
        return cycle_result
    
    def _recommend_backend(self, edit: Dict) -> str:
        """Recommend backend based on learned heuristics"""
        language = edit.get('language', 'en')
        
        if language in self.learned_heuristics:
            return self.learned_heuristics[language]['preferred_backend']
        
        # Default: choose backend with best overall performance
        best_backend = max(
            self.config.backends,
            key=lambda b: np.mean(self.backend_performance[b]) if self.backend_performance[b] else 0.5
        )
        return best_backend
    
    def _calculate_performance_trend(self) -> str:
        """Calculate recent performance trend"""
        if len(self.edit_history) < 5:
            return "insufficient_data"
        
        recent_deltas = [r.performance_delta for r in self.edit_history[-5:]]
        trend = np.mean(recent_deltas)
        
        if trend > 0.1:
            return "improving"
        elif trend < -0.1:
            return "declining"
        else:
            return "stable"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            'total_edits': len(self.edit_history),
            'backend_performance': {
                backend: {
                    'mean_reward': np.mean(perfs) if perfs else 0.0,
                    'std_reward': np.std(perfs) if perfs else 0.0,
                    'edit_count': len(perfs)
                }
                for backend, perfs in self.backend_performance.items()
            },
            'learned_heuristics': self.learned_heuristics,
            'performance_trend': self._calculate_performance_trend(),
            'quantum_stats': self.qec_extension.get_statistics(),
            'recent_performance': [
                {
                    'edit_id': r.edit_id,
                    'backend': r.backend,
                    'performance_delta': r.performance_delta,
                    'timestamp': r.timestamp
                }
                for r in self.edit_history[-10:]
            ]
        }


def create_hybrid_agent(config: QuantumRLConfig = None) -> QuantumScalingRLHybrid:
    """Factory function to create hybrid agent"""
    return QuantumScalingRLHybrid(config)
