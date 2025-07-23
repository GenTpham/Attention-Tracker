"""
Enhanced Attention Detector for Prompt Injection Detection
Advanced features: adaptive threshold, multiple aggregation strategies, performance monitoring
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import time
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from enhanced_attention_model import EnhancedAttentionModel, MemoryOptimizer

class AttentionProcessor:
    """Advanced attention processing with multiple aggregation strategies"""
    
    @staticmethod
    def process_attention_map(
        attention_map: List[torch.Tensor], 
        token_ranges: Tuple[Tuple[int, int], Tuple[int, int]], 
        method: str = "normalize_sum"
    ) -> np.ndarray:
        """Process attention map with various aggregation methods"""
        
        inst_range, data_range = token_ranges
        heatmap = np.zeros((len(attention_map), attention_map[0].shape[1]))
        
        for layer_idx, attn_layer in enumerate(attention_map):
            try:
                attn_layer = attn_layer.to(torch.float32).numpy()
                
                # Extract attention to instruction and data
                inst_attn = np.sum(attn_layer[0, :, -1, inst_range[0]:inst_range[1]], axis=1)
                data_attn = np.sum(attn_layer[0, :, -1, data_range[0]:data_range[1]], axis=1)
                
                if "normalize" in method:
                    epsilon = 1e-8
                    total_attn = inst_attn + data_attn + epsilon
                    heatmap[layer_idx, :] = inst_attn / total_attn
                elif "ratio" in method:
                    epsilon = 1e-8
                    heatmap[layer_idx, :] = inst_attn / (data_attn + epsilon)
                elif "difference" in method:
                    heatmap[layer_idx, :] = inst_attn - data_attn
                else:  # raw sum
                    heatmap[layer_idx, :] = inst_attn
                    
            except Exception as e:
                print(f"⚠️ Error processing layer {layer_idx}: {e}")
                continue
        
        return np.nan_to_num(heatmap, nan=0.0)
    
    @staticmethod
    def calculate_attention_score(heatmap: np.ndarray, important_heads: List[List[int]]) -> float:
        """Calculate focus score from processed heatmap"""
        try:
            if len(important_heads) == 0:
                return 0.0
            
            scores = []
            for layer, head in important_heads:
                if layer < heatmap.shape[0] and head < heatmap.shape[1]:
                    scores.append(heatmap[layer, head])
            
            return float(np.mean(scores)) if scores else 0.0
            
        except Exception as e:
            print(f"❌ Error calculating attention score: {e}")
            return 0.0

class AdaptiveThresholder:
    """Adaptive threshold management with multiple calibration strategies"""
    
    def __init__(self, initial_threshold: float = 0.5):
        self.threshold = initial_threshold
        self.calibration_history = []
        self.performance_history = []
        
    def calibrate_from_examples(
        self, 
        pos_scores: List[float], 
        neg_scores: List[float], 
        method: str = "statistical"
    ) -> float:
        """Calibrate threshold using positive and negative examples"""
        
        if not pos_scores and not neg_scores:
            return self.threshold
        
        if method == "statistical":
            if pos_scores and neg_scores:
                # Statistical separation method
                pos_mean, pos_std = np.mean(pos_scores), np.std(pos_scores)
                neg_mean, neg_std = np.mean(neg_scores), np.std(neg_scores)
                
                # Find optimal threshold between distributions
                self.threshold = (pos_mean - 2 * pos_std + neg_mean + 2 * neg_std) / 2
                
            elif pos_scores:
                # Conservative threshold based on positive examples
                pos_mean, pos_std = np.mean(pos_scores), np.std(pos_scores)
                self.threshold = pos_mean - 2 * pos_std
                
        elif method == "percentile":
            if pos_scores and neg_scores:
                # Use percentile-based threshold
                pos_25 = np.percentile(pos_scores, 25)
                neg_75 = np.percentile(neg_scores, 75)
                self.threshold = (pos_25 + neg_75) / 2
        
        self.calibration_history.append({
            'method': method,
            'threshold': self.threshold,
            'pos_samples': len(pos_scores) if pos_scores else 0,
            'neg_samples': len(neg_scores) if neg_scores else 0
        })
        
        return self.threshold
    
    def update_from_performance(self, metrics: Dict[str, float]):
        """Update threshold based on performance metrics"""
        self.performance_history.append(metrics)
        
        # Simple adaptive adjustment based on FPR/FNR
        if len(self.performance_history) >= 3:
            recent_fpr = np.mean([m.get('fpr', 0) for m in self.performance_history[-3:]])
            recent_fnr = np.mean([m.get('fnr', 0) for m in self.performance_history[-3:]])
            
            # Adjust threshold to balance FPR and FNR
            if recent_fpr > 0.1 and recent_fnr < 0.05:  # Too many false positives
                self.threshold -= 0.01
            elif recent_fnr > 0.1 and recent_fpr < 0.05:  # Too many false negatives
                self.threshold += 0.01

class EnhancedAttentionDetector:
    """Enhanced Attention Detector with advanced features"""
    
    def __init__(
        self,
        model: EnhancedAttentionModel,
        pos_examples: Optional[List[str]] = None,
        neg_examples: Optional[List[str]] = None,
        use_token: Union[str, int] = "first",
        instruction: str = "Say xxxxxx",
        threshold: float = 0.5,
        attention_method: str = "normalize_sum",
        adaptive_threshold: bool = True
    ):
        self.name = "enhanced_attention_detector"
        self.model = model
        self.important_heads = model.important_heads
        self.instruction = instruction
        self.use_token = use_token
        self.attention_method = attention_method
        self.adaptive_threshold = adaptive_threshold
        
        # Initialize adaptive thresholder
        self.thresholder = AdaptiveThresholder(threshold)
        
        # Performance tracking
        self.detection_stats = {
            'total_detections': 0,
            'total_queries': 0,
            'detection_times': [],
            'score_distribution': {'positive': [], 'negative': []},
            'accuracy_over_time': []
        }
        
        # Calibrate threshold if examples provided
        if pos_examples or neg_examples:
            self._calibrate_threshold(pos_examples, neg_examples)
        
        print(f"🛡️ Enhanced detector initialized")
        print(f"   Threshold: {self.thresholder.threshold:.4f}")
        print(f"   Attention method: {self.attention_method}")
        print(f"   Use token: {self.use_token}")
    
    def _calibrate_threshold(
        self, 
        pos_examples: Optional[List[str]] = None, 
        neg_examples: Optional[List[str]] = None
    ):
        """Calibrate threshold using example prompts"""
        
        pos_scores, neg_scores = [], []
        
        print("🔧 Calibrating threshold...")
        
        if pos_examples:
            print(f"   Processing {len(pos_examples)} positive examples...")
            for prompt in pos_examples:
                try:
                    _, _, attention_maps, _, token_ranges, _ = self.model.inference(
                        self.instruction, prompt, max_output_tokens=1
                    )
                    score = self._compute_attention_score(attention_maps, token_ranges)
                    pos_scores.append(score)
                except Exception as e:
                    print(f"⚠️ Error processing positive example: {e}")
        
        if neg_examples:
            print(f"   Processing {len(neg_examples)} negative examples...")
            for prompt in neg_examples:
                try:
                    _, _, attention_maps, _, token_ranges, _ = self.model.inference(
                        self.instruction, prompt, max_output_tokens=1
                    )
                    score = self._compute_attention_score(attention_maps, token_ranges)
                    neg_scores.append(score)
                except Exception as e:
                    print(f"⚠️ Error processing negative example: {e}")
        
        # Calibrate threshold
        new_threshold = self.thresholder.calibrate_from_examples(pos_scores, neg_scores)
        print(f"✅ Threshold calibrated: {new_threshold:.4f}")
        
        if pos_scores and neg_scores:
            self._print_calibration_stats(pos_scores, neg_scores)
    
    def _print_calibration_stats(self, pos_scores: List[float], neg_scores: List[float]):
        """Print calibration statistics"""
        pos_mean, pos_std = np.mean(pos_scores), np.std(pos_scores)
        neg_mean, neg_std = np.mean(neg_scores), np.std(neg_scores)
        
        print(f"   Positive examples: μ={pos_mean:.4f}, σ={pos_std:.4f}")
        print(f"   Negative examples: μ={neg_mean:.4f}, σ={neg_std:.4f}")
        print(f"   Separation: {abs(pos_mean - neg_mean):.4f}")
    
    def _compute_attention_score(
        self, 
        attention_maps: List[List[torch.Tensor]], 
        token_ranges: Tuple[Tuple[int, int], Tuple[int, int]]
    ) -> float:
        """Compute attention score with multiple token strategies"""
        
        if not attention_maps:
            return 0.0
        
        scores = []
        
        if self.use_token == "first":
            # Use only first generated token
            if len(attention_maps) > 0:
                heatmap = AttentionProcessor.process_attention_map(
                    attention_maps[0], token_ranges, self.attention_method
                )
                score = AttentionProcessor.calculate_attention_score(heatmap, self.important_heads)
                scores.append(score)
                
        elif self.use_token == "all":
            # Use all generated tokens
            for attention_map in attention_maps:
                heatmap = AttentionProcessor.process_attention_map(
                    attention_map, token_ranges, self.attention_method
                )
                score = AttentionProcessor.calculate_attention_score(heatmap, self.important_heads)
                scores.append(score)
                
        elif isinstance(self.use_token, int):
            # Use first N tokens
            for attention_map in attention_maps[:self.use_token]:
                heatmap = AttentionProcessor.process_attention_map(
                    attention_map, token_ranges, self.attention_method
                )
                score = AttentionProcessor.calculate_attention_score(heatmap, self.important_heads)
                scores.append(score)
        
        return float(np.mean(scores)) if scores else 0.0
    
    def detect(self, data_prompt: str) -> Tuple[bool, Dict[str, Any]]:
        """Detect prompt injection with comprehensive analysis"""
        
        start_time = time.time()
        
        try:
            with MemoryOptimizer.memory_cleanup():
                # Run inference
                _, _, attention_maps, _, token_ranges, _ = self.model.inference(
                    self.instruction, data_prompt, max_output_tokens=1
                )
                
                # Compute attention score
                focus_score = self._compute_attention_score(attention_maps, token_ranges)
                
                # Make detection decision
                is_injection = focus_score <= self.thresholder.threshold
                
                # Calculate detection confidence
                confidence = abs(focus_score - self.thresholder.threshold) / max(self.thresholder.threshold, 1e-6)
                
                # Update statistics
                detection_time = time.time() - start_time
                self._update_detection_stats(is_injection, focus_score, detection_time)
                
                # Prepare detailed results
                details = {
                    'focus_score': focus_score,
                    'threshold': self.thresholder.threshold,
                    'confidence': confidence,
                    'detection_time': detection_time,
                    'attention_method': self.attention_method,
                    'important_heads_count': len(self.important_heads)
                }
                
                return is_injection, details
                
        except Exception as e:
            print(f"❌ Error in detection: {e}")
            return False, {'focus_score': 0.0, 'error': str(e)}
    
    def detect_batch(self, prompts: List[str]) -> List[Tuple[bool, Dict[str, Any]]]:
        """Batch detection for multiple prompts"""
        
        print(f"🔍 Processing batch of {len(prompts)} prompts...")
        results = []
        
        for i, prompt in enumerate(prompts):
            try:
                result = self.detect(prompt)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"   Processed {i + 1}/{len(prompts)} prompts")
                    
            except Exception as e:
                print(f"❌ Error processing prompt {i}: {e}")
                results.append((False, {'error': str(e)}))
        
        return results
    
    def _update_detection_stats(self, is_injection: bool, score: float, detection_time: float):
        """Update detection statistics"""
        
        self.detection_stats['total_queries'] += 1
        if is_injection:
            self.detection_stats['total_detections'] += 1
            self.detection_stats['score_distribution']['positive'].append(score)
        else:
            self.detection_stats['score_distribution']['negative'].append(score)
        
        self.detection_stats['detection_times'].append(detection_time)
    
    def evaluate_on_dataset(self, dataset: List[Dict]) -> Dict[str, float]:
        """Comprehensive evaluation on dataset"""
        
        print(f"📊 Evaluating on dataset with {len(dataset)} samples...")
        
        true_labels = []
        predictions = []
        scores = []
        
        start_time = time.time()
        
        for i, item in enumerate(dataset):
            try:
                text = item['text']
                label = item['label']
                
                is_injection, details = self.detect(text)
                
                true_labels.append(label)
                predictions.append(is_injection)
                scores.append(1 - details['focus_score'])  # Invert for AUC calculation
                
                if (i + 1) % 50 == 0:
                    print(f"   Evaluated {i + 1}/{len(dataset)} samples")
                    
            except Exception as e:
                print(f"❌ Error evaluating sample {i}: {e}")
                continue
        
        # Calculate metrics
        metrics = self._calculate_metrics(true_labels, predictions, scores)
        metrics['evaluation_time'] = time.time() - start_time
        metrics['samples_evaluated'] = len(true_labels)
        
        # Update adaptive threshold if enabled
        if self.adaptive_threshold:
            self.thresholder.update_from_performance(metrics)
        
        return metrics
    
    def _calculate_metrics(
        self, 
        true_labels: List[int], 
        predictions: List[bool], 
        scores: List[float]
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        
        try:
            # Convert predictions to int
            pred_int = [int(p) for p in predictions]
            
            # Basic metrics
            tn, fp, fn, tp = confusion_matrix(true_labels, pred_int).ravel()
            
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # AUC metrics
            auc = roc_auc_score(true_labels, scores) if len(set(true_labels)) > 1 else 0
            auprc = average_precision_score(true_labels, scores) if len(set(true_labels)) > 1 else 0
            
            return {
                'accuracy': round(accuracy, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'fnr': round(fnr, 4),
                'fpr': round(fpr, 4),
                'auc': round(auc, 4),
                'auprc': round(auprc, 4),
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            }
            
        except Exception as e:
            print(f"❌ Error calculating metrics: {e}")
            return {}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        stats = self.detection_stats
        
        summary = {
            'total_queries': stats['total_queries'],
            'total_detections': stats['total_detections'],
            'detection_rate': stats['total_detections'] / max(stats['total_queries'], 1),
            'threshold': self.thresholder.threshold
        }
        
        if stats['detection_times']:
            summary['avg_detection_time'] = np.mean(stats['detection_times'])
            summary['max_detection_time'] = max(stats['detection_times'])
        
        if stats['score_distribution']['positive']:
            summary['positive_score_stats'] = {
                'mean': np.mean(stats['score_distribution']['positive']),
                'std': np.std(stats['score_distribution']['positive']),
                'count': len(stats['score_distribution']['positive'])
            }
        
        if stats['score_distribution']['negative']:
            summary['negative_score_stats'] = {
                'mean': np.mean(stats['score_distribution']['negative']),
                'std': np.std(stats['score_distribution']['negative']),
                'count': len(stats['score_distribution']['negative'])
            }
        
        return summary
    
    def visualize_score_distribution(self, save_path: Optional[str] = None):
        """Visualize score distribution"""
        
        pos_scores = self.detection_stats['score_distribution']['positive']
        neg_scores = self.detection_stats['score_distribution']['negative']
        
        if not pos_scores and not neg_scores:
            print("📊 No data to visualize")
            return
        
        plt.figure(figsize=(10, 6))
        
        if pos_scores:
            plt.hist(pos_scores, bins=30, alpha=0.7, label=f'Positive ({len(pos_scores)})', color='red')
        
        if neg_scores:
            plt.hist(neg_scores, bins=30, alpha=0.7, label=f'Negative ({len(neg_scores)})', color='blue')
        
        plt.axvline(self.thresholder.threshold, color='black', linestyle='--', 
                   label=f'Threshold ({self.thresholder.threshold:.3f})')
        
        plt.xlabel('Focus Score')
        plt.ylabel('Frequency')
        plt.title('Attention Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to: {save_path}")
        
        plt.show()
    
    def export_results(self, filepath: str):
        """Export detection results and statistics"""
        
        export_data = {
            'detector_config': {
                'name': self.name,
                'model_name': self.model.name,
                'instruction': self.instruction,
                'use_token': self.use_token,
                'attention_method': self.attention_method,
                'threshold': self.thresholder.threshold,
                'important_heads_count': len(self.important_heads)
            },
            'performance_summary': self.get_performance_summary(),
            'calibration_history': self.thresholder.calibration_history,
            'model_performance': self.model.get_performance_stats()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Results exported to: {filepath}")
    
    def cleanup(self):
        """Clean up detector resources"""
        if hasattr(self, 'model'):
            self.model.cleanup()
        print("🧹 Detector cleanup completed") 