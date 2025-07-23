"""
Enhanced Attention Model for Prompt Injection Detection
Optimized for Qwen2 and Granite3 models with improved memory usage and error handling
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import time
import warnings
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from contextlib import contextmanager

warnings.filterwarnings('ignore')

class MemoryOptimizer:
    """Memory optimization utilities for efficient model inference"""
    
    @staticmethod
    @contextmanager
    def memory_cleanup():
        """Context manager for automatic memory cleanup"""
        try:
            yield
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    @staticmethod
    def get_memory_info():
        """Get current memory usage information"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            return f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"
        return "CPU Memory - Monitoring not available"

class TokenRangeCalculator:
    """Optimized token range calculation for different model architectures"""
    
    MODEL_CONFIGS = {
        'qwen': {
            'instruction_offset': 3,
            'data_offset': -5,
            'special_tokens': ['<|im_start|>', '<|im_end|>']
        },
        'granite': {
            'instruction_offset': 3,
            'data_offset': -5,
            'special_tokens': ['<|start_of_role|>', '<|end_of_role|>']
        },
        'phi3': {
            'instruction_offset': 1,
            'data_offset': -2,
            'special_tokens': ['<|user|>', '<|assistant|>']
        },
        'llama': {
            'instruction_offset': 2,
            'data_offset': -3,
            'special_tokens': ['<|begin_of_text|>', '<|end_of_text|>']
        }
    }
    
    @classmethod
    def get_token_ranges(cls, model_name: str, tokenizer, instruction: str, data: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Calculate token ranges based on model architecture"""
        try:
            instruction_len = len(tokenizer.encode(instruction, add_special_tokens=False))
            data_len = len(tokenizer.encode(data, add_special_tokens=False))
            
            # Detect model type
            model_type = cls._detect_model_type(model_name.lower())
            config = cls.MODEL_CONFIGS.get(model_type, cls.MODEL_CONFIGS['qwen'])
            
            inst_range = (config['instruction_offset'], config['instruction_offset'] + instruction_len)
            data_range = (config['data_offset'] - data_len, config['data_offset'])
            
            return inst_range, data_range
            
        except Exception as e:
            print(f"⚠️ Error calculating token ranges: {e}")
            # Safe fallback
            return ((1, 10), (-10, -1))
    
    @staticmethod
    def _detect_model_type(model_name: str) -> str:
        """Detect model type from name"""
        if 'qwen' in model_name:
            return 'qwen'
        elif 'granite' in model_name:
            return 'granite'
        elif 'phi' in model_name:
            return 'phi3'
        elif 'llama' in model_name:
            return 'llama'
        else:
            return 'qwen'  # Default

class EnhancedAttentionModel:
    """Enhanced Attention Model with optimizations for Qwen2 and Granite3"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_info = config["model_info"]
        self.params = config["params"]
        
        # Basic info
        self.provider = self.model_info["provider"]
        self.name = self.model_info["name"]
        self.model_id = self.model_info["model_id"]
        self.temperature = float(self.params.get("temperature", 1.0))
        self.max_output_tokens = int(self.params["max_output_tokens"])
        
        # Generation parameters
        self.top_k = self.params.get("top_k", 50)
        self.top_p = self.params.get("top_p", None)
        
        # Device setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Using device: {self.device}")
        
        # Initialize model and tokenizer
        self._initialize_tokenizer()
        self._initialize_model()
        
        # Initialize attention heads
        self._initialize_attention_heads()
        
        # Performance monitoring
        self.inference_stats = {
            'total_inferences': 0,
            'total_time': 0,
            'avg_time_per_inference': 0,
            'memory_peaks': []
        }
        
        print(f"✅ Enhanced model initialized: {self.name}")
        self.print_model_info()
    
    def _initialize_tokenizer(self):
        """Initialize tokenizer with error handling"""
        try:
            print(f"📝 Loading tokenizer: {self.model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                use_fast=True  # Use fast tokenizer for better performance
            )
            
            # Add padding token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("🔧 Added padding token")
                
        except Exception as e:
            print(f"❌ Error loading tokenizer: {e}")
            raise
    
    def _initialize_model(self):
        """Initialize model with memory optimization"""
        try:
            print(f"🤖 Loading model: {self.model_id}")
            
            # Determine optimal dtype and device_map
            dtype = torch.bfloat16 if self.device == 'cuda' else torch.float32
            device_map = 'auto' if self.device == 'cuda' else 'cpu'
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
                use_cache=False,  # Disable cache for memory efficiency
            ).eval()
            
            print(f"✅ Model loaded successfully")
            print(MemoryOptimizer.get_memory_info())
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _initialize_attention_heads(self):
        """Initialize important attention heads"""
        try:
            if self.params["important_heads"] == "all":
                print("📏 Calculating attention dimensions...")
                attn_size = self._get_attention_dimensions()
                self.important_heads = [
                    [i, j] for i in range(attn_size[0])
                    for j in range(attn_size[1])
                ]
                print(f"🎯 Using all attention heads: {attn_size[0]}x{attn_size[1]} = {len(self.important_heads)} heads")
            else:
                self.important_heads = self.params["important_heads"]
                print(f"🎯 Using pre-selected heads: {len(self.important_heads)} heads")
                
        except Exception as e:
            print(f"⚠️ Error initializing attention heads: {e}")
            self.important_heads = [[0, 0]]  # Safe fallback
    
    def _get_attention_dimensions(self) -> Tuple[int, int]:
        """Get attention map dimensions efficiently"""
        try:
            with MemoryOptimizer.memory_cleanup():
                _, _, attention_maps, _, _, _ = self.inference("test", "", max_output_tokens=1)
                if attention_maps and len(attention_maps) > 0:
                    attention_map = attention_maps[0]
                    dims = (len(attention_map), attention_map[0].shape[1])
                    print(f"✅ Attention dimensions: {dims}")
                    return dims
                else:
                    print("⚠️ No attention maps found, using default dimensions")
                    return (32, 32)
        except Exception as e:
            print(f"❌ Error getting attention dimensions: {e}")
            return (32, 32)  # Safe default
    
    def sample_token_optimized(self, logits: torch.Tensor) -> torch.Tensor:
        """Optimized token sampling with multiple strategies - returns scalar tensor"""
        try:
            # Apply temperature
            if self.temperature != 1.0:
                logits = logits / self.temperature
            
            # Top-k sampling
            if self.top_k is not None and self.top_k > 0:
                top_k = min(self.top_k, logits.size(-1))
                values, indices = torch.topk(logits, top_k, dim=-1)
                probs = F.softmax(values, dim=-1)
                sampled_idx = torch.multinomial(probs, 1)
                next_token_id = indices[sampled_idx]
                return next_token_id.squeeze()  # Return scalar
            
            # Top-p (nucleus) sampling
            if self.top_p is not None and 0 < self.top_p < 1:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > self.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Standard sampling
            return torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze()  # Return scalar
            
        except Exception as e:
            print(f"❌ Error in token sampling: {e}")
            # Fallback to greedy
            return torch.argmax(logits, dim=-1)  # Already scalar
    
    def inference(self, instruction: str, data: str, max_output_tokens: Optional[int] = None) -> Tuple:
        """Enhanced inference with comprehensive error handling and optimization"""
        start_time = time.time()
        
        try:
            with MemoryOptimizer.memory_cleanup():
                # Prepare messages
                messages = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": f"Data: {data}"}
                ]
                
                # Create input text with model-specific template
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Tokenize with optimization
                model_inputs = self.tokenizer(
                    [text],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                    return_attention_mask=True
                ).to(self.device)
                
                input_tokens = self.tokenizer.convert_ids_to_tokens(model_inputs['input_ids'][0])
                
                # Calculate token ranges
                inst_range, data_range = TokenRangeCalculator.get_token_ranges(
                    self.name, self.tokenizer, instruction, data
                )
                token_ranges = (inst_range, data_range)
                
                # Initialize generation variables
                generated_tokens = []
                generated_probs = []
                attention_maps = []
                
                input_ids = model_inputs.input_ids
                attention_mask = model_inputs.attention_mask
                
                n_tokens = max_output_tokens if max_output_tokens else self.max_output_tokens
                
                # Generation loop with optimizations
                with torch.no_grad():
                    for i in range(n_tokens):
                        try:
                            # Forward pass
                            outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                output_attentions=True,
                                use_cache=False  # Disable cache for memory efficiency
                            )
                            
                            # Process logits
                            logits = outputs.logits[:, -1, :]
                            probs = F.softmax(logits, dim=-1)
                            
                            # Sample next token
                            next_token_id = self.sample_token_optimized(logits[0])
                            
                            # Store results
                            generated_probs.append(probs[0, next_token_id.item()].item())
                            generated_tokens.append(next_token_id.item())
                            
                            # Check for EOS
                            if next_token_id.item() == self.tokenizer.eos_token_id:
                                break
                            
                            # Update sequences
                            input_ids = torch.cat(
                                (input_ids, next_token_id.unsqueeze(0).unsqueeze(0)), dim=-1
                            )
                            attention_mask = torch.cat(
                                (attention_mask, torch.tensor([[1]], device=self.device)), dim=-1
                            )
                            
                            # Process attention maps efficiently
                            attention_map = self._process_attention_maps(outputs.attentions)
                            attention_maps.append(attention_map)
                            
                            # Clean up intermediate results
                            del outputs
                            
                            # Periodic memory cleanup
                            if i % 5 == 0:
                                torch.cuda.empty_cache() if self.device == 'cuda' else None
                                
                        except Exception as e:
                            print(f"❌ Error during generation step {i}: {e}")
                            break
                
                # Decode generated tokens
                output_tokens = [
                    self.tokenizer.decode(token, skip_special_tokens=True)
                    for token in generated_tokens
                ]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Update performance stats
                inference_time = time.time() - start_time
                self._update_performance_stats(inference_time)
                
                return generated_text, output_tokens, attention_maps, input_tokens, token_ranges, generated_probs
                
        except Exception as e:
            print(f"❌ Error in inference: {e}")
            return "", [], [], [], ((0, 1), (-1, 0)), []
    
    def _process_attention_maps(self, attentions: Tuple[torch.Tensor]) -> List[torch.Tensor]:
        """Process attention maps efficiently"""
        try:
            attention_map = []
            for attention in attentions:
                # Convert to half precision and move to CPU immediately
                processed = attention.detach().cpu().half()
                # Handle NaN values
                processed = torch.nan_to_num(processed, nan=0.0)
                # Extract last attention
                processed = processed[:, :, -1, :].unsqueeze(2)
                attention_map.append(processed)
            
            return attention_map
            
        except Exception as e:
            print(f"❌ Error processing attention maps: {e}")
            return []
    
    def _update_performance_stats(self, inference_time: float):
        """Update performance statistics"""
        self.inference_stats['total_inferences'] += 1
        self.inference_stats['total_time'] += inference_time
        self.inference_stats['avg_time_per_inference'] = (
            self.inference_stats['total_time'] / self.inference_stats['total_inferences']
        )
        
        if torch.cuda.is_available():
            self.inference_stats['memory_peaks'].append(torch.cuda.max_memory_allocated() / 1e9)
    
    def print_model_info(self):
        """Print comprehensive model information"""
        info_lines = [
            f"🤖 Enhanced Model Information",
            f"   Provider: {self.provider}",
            f"   Name: {self.name}",
            f"   Model ID: {self.model_id}",
            f"   Device: {self.device}",
            f"   Temperature: {self.temperature}",
            f"   Max Output Tokens: {self.max_output_tokens}",
            f"   Important Heads: {len(self.important_heads)}",
            f"   Memory: {MemoryOptimizer.get_memory_info()}"
        ]
        
        max_len = max(len(line) for line in info_lines)
        print("=" * max_len)
        for line in info_lines:
            print(line)
        print("=" * max_len)
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        stats = self.inference_stats.copy()
        if stats['memory_peaks']:
            stats['avg_memory_peak'] = np.mean(stats['memory_peaks'])
            stats['max_memory_peak'] = max(stats['memory_peaks'])
        return stats
    
    def cleanup(self):
        """Clean up model resources"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("🧹 Model cleanup completed") 