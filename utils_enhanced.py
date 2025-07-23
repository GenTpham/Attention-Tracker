"""
Enhanced Utilities for Attention Tracker
Improved model loading, configuration, and evaluation utilities
"""

import os
import json
import random
import torch
import numpy as np
import warnings
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import hashlib
import pickle
from datetime import datetime
from enhanced_attention_model import EnhancedAttentionModel
from enhanced_detector import EnhancedAttentionDetector

warnings.filterwarnings('ignore')

class ConfigManager:
    """Enhanced configuration management"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration with enhanced validation and error handling"""
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Validate configuration
            if not ConfigManager.validate_config(config):
                raise ValueError("Invalid configuration format")
            
            print(f"✅ Configuration loaded: {config_path.name}")
            return config
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading config: {e}")
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Validate configuration structure"""
        required_keys = ["model_info", "params"]
        model_info_keys = ["provider", "name", "model_id"]
        params_keys = ["temperature", "max_output_tokens", "important_heads"]
        
        # Check top-level keys
        if not all(key in config for key in required_keys):
            print(f"❌ Missing required config keys: {required_keys}")
            return False
        
        # Check model_info keys
        if not all(key in config["model_info"] for key in model_info_keys):
            print(f"❌ Missing model_info keys: {model_info_keys}")
            return False
        
        # Check params keys
        if not all(key in config["params"] for key in params_keys):
            print(f"❌ Missing params keys: {params_keys}")
            return False
        
        return True
    
    @staticmethod
    def get_available_configs() -> List[str]:
        """Get list of available model configurations"""
        config_dir = Path("configs/model_configs")
        if not config_dir.exists():
            print(f"❌ Config directory not found: {config_dir}")
            return []
        
        configs = []
        for file_path in config_dir.glob("*-attn_config.json"):
            model_name = file_path.stem.replace("-attn_config", "")
            configs.append(model_name)
        
        return sorted(configs)
    
    @staticmethod
    def create_config_template(model_name: str, model_id: str) -> Dict[str, Any]:
        """Create configuration template for new models"""
        return {
            "model_info": {
                "provider": "attn-hf",
                "name": f"{model_name}-attn",
                "model_id": model_id
            },
            "params": {
                "temperature": 0.1,
                "max_output_tokens": 32,
                "important_heads": "all",
                "top_k": 50,
                "top_p": None
            }
        }

class ModelFactory:
    """Enhanced model factory with caching and optimization"""
    
    _model_cache = {}
    _config_cache = {}
    
    @classmethod
    def create_model(cls, config_path: str, use_cache: bool = True) -> EnhancedAttentionModel:
        """Create model with caching support"""
        
        # Create cache key
        cache_key = cls._get_cache_key(config_path)
        
        if use_cache and cache_key in cls._model_cache:
            print(f"📦 Using cached model for {config_path}")
            return cls._model_cache[cache_key]
        
        # Load configuration
        config = ConfigManager.load_config(config_path)
        
        # Create model
        print(f"🏗️ Creating model: {config['model_info']['name']}")
        model = EnhancedAttentionModel(config)
        
        # Cache model if requested
        if use_cache:
            cls._model_cache[cache_key] = model
            print(f"💾 Model cached with key: {cache_key[:8]}...")
        
        return model
    
    @classmethod
    def create_detector(
        cls, 
        config_path: str, 
        detector_config: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> EnhancedAttentionDetector:
        """Create detector with enhanced configuration"""
        
        # Create model
        model = cls.create_model(config_path, use_cache)
        
        # Default detector configuration
        default_config = {
            'instruction': "Say xxxxxx",
            'threshold': 0.5,
            'use_token': 'first',
            'attention_method': 'normalize_sum',
            'adaptive_threshold': True
        }
        
        # Update with user config
        if detector_config:
            default_config.update(detector_config)
        
        # Create detector
        detector = EnhancedAttentionDetector(model, **default_config)
        
        return detector
    
    @staticmethod
    def _get_cache_key(config_path: str) -> str:
        """Generate cache key for configuration"""
        with open(config_path, 'rb') as f:
            config_hash = hashlib.md5(f.read()).hexdigest()
        return f"{Path(config_path).stem}_{config_hash}"
    
    @classmethod
    def clear_cache(cls):
        """Clear model cache"""
        for model in cls._model_cache.values():
            model.cleanup()
        cls._model_cache.clear()
        cls._config_cache.clear()
        print("🧹 Model cache cleared")

class SeedManager:
    """Enhanced seed management for reproducibility"""
    
    @staticmethod
    def set_seed(seed: int = 42, deterministic: bool = True) -> None:
        """Set comprehensive random seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
        
        # Set environment variables for additional reproducibility
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        print(f"🎲 Seed set to: {seed} (deterministic: {deterministic})")

class DatasetLoader:
    """Enhanced dataset loading and processing"""
    
    @staticmethod
    def load_prompt_injection_dataset(dataset_name: str = "deepset/prompt-injections") -> Dict[str, Any]:
        """Load and validate prompt injection dataset"""
        try:
            from datasets import load_dataset
            
            print(f"📊 Loading dataset: {dataset_name}")
            dataset = load_dataset(dataset_name)
            
            # Validate dataset structure
            if 'test' not in dataset:
                raise ValueError("Dataset must contain 'test' split")
            
            test_data = dataset['test']
            
            # Check required fields
            required_fields = ['text', 'label']
            sample = test_data[0]
            
            if not all(field in sample for field in required_fields):
                raise ValueError(f"Dataset must contain fields: {required_fields}")
            
            print(f"✅ Dataset loaded successfully")
            print(f"   Test samples: {len(test_data)}")
            print(f"   Positive samples: {sum(1 for item in test_data if item['label'] == 1)}")
            print(f"   Negative samples: {sum(1 for item in test_data if item['label'] == 0)}")
            
            return dataset
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise

class ResultsManager:
    """Enhanced results management and export"""
    
    @staticmethod
    def create_results_directory(base_dir: str = "results") -> Path:
        """Create timestamped results directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(base_dir) / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Results directory created: {results_dir}")
        return results_dir
    
    @staticmethod
    def save_results(
        results: Dict[str, Any], 
        filepath: Union[str, Path], 
        format: str = "json"
    ) -> None:
        """Save results in various formats"""
        filepath = Path(filepath)
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        elif format.lower() == "pickle":
            with open(filepath, 'wb') as f:
                pickle.dump(results, f)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"💾 Results saved: {filepath}")
    
    @staticmethod
    def load_results(filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load results from file"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")
        
        if filepath.suffix == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        elif filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

class PerformanceMonitor:
    """Performance monitoring and benchmarking utilities"""
    
    def __init__(self):
        self.metrics = {
            'memory_usage': [],
            'inference_times': [],
            'gpu_utilization': []
        }
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.initial_memory = torch.cuda.memory_allocated()
    
    def record_step(self, step_name: str):
        """Record performance metrics for a step"""
        current_time = time.time()
        
        step_metrics = {
            'step': step_name,
            'timestamp': current_time,
            'elapsed_time': current_time - self.start_time if hasattr(self, 'start_time') else 0
        }
        
        if torch.cuda.is_available():
            step_metrics.update({
                'memory_allocated': torch.cuda.memory_allocated() / 1e9,
                'memory_cached': torch.cuda.memory_reserved() / 1e9,
                'max_memory_allocated': torch.cuda.max_memory_allocated() / 1e9
            })
        
        self.metrics['inference_times'].append(step_metrics)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.metrics['inference_times']:
            return {}
        
        times = [m['elapsed_time'] for m in self.metrics['inference_times']]
        
        summary = {
            'total_time': max(times) if times else 0,
            'avg_step_time': np.mean(times) if times else 0,
            'total_steps': len(times)
        }
        
        if torch.cuda.is_available() and self.metrics['inference_times']:
            memories = [m.get('memory_allocated', 0) for m in self.metrics['inference_times']]
            summary.update({
                'peak_memory_gb': max(memories) if memories else 0,
                'avg_memory_gb': np.mean(memories) if memories else 0
            })
        
        return summary

# Convenience functions for easy use
def create_model_from_config(model_name: str, use_cache: bool = True) -> EnhancedAttentionModel:
    """Create model from model name (convenience function)"""
    config_path = f"configs/model_configs/{model_name}-attn_config.json"
    return ModelFactory.create_model(config_path, use_cache)

def create_detector_from_config(
    model_name: str, 
    detector_config: Optional[Dict[str, Any]] = None,
    use_cache: bool = True
) -> EnhancedAttentionDetector:
    """Create detector from model name (convenience function)"""
    config_path = f"configs/model_configs/{model_name}-attn_config.json"
    return ModelFactory.create_detector(config_path, detector_config, use_cache)

def set_optimal_environment():
    """Set optimal environment for attention tracking"""
    # Set optimal memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Set memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.9)
    
    # Set optimal number of threads
    if hasattr(torch, 'set_num_threads'):
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    
    print("🔧 Environment optimized for attention tracking")

def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information"""
    info = {
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cpu_count': os.cpu_count()
    }
    
    if torch.cuda.is_available():
        info.update({
            'gpu_name': torch.cuda.get_device_name(),
            'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
            'cuda_version': torch.version.cuda
        })
    
    return info

# Import required modules
import sys
import time 