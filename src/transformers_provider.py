#!/usr/bin/env python3
"""
Transformers-based LLM Provider Implementation
Supports Phi-3.5, Llama 3.2, and Qwen 2.5 models
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Optional, Dict, Any
import logging
import gc
from llm_providers import LLMProvider

logger = logging.getLogger(__name__)

class TransformersLLM(LLMProvider):
    """HuggingFace Transformers-based LLM provider"""
    
    def __init__(self, model_name: str, device: str = "auto", torch_dtype: str = "float16", **kwargs):
        super().__init__(model_name, **kwargs)
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.pipeline: Optional[pipeline] = None
        
        # Model-specific configurations
        self.model_configs = {
            "microsoft/Phi-3.5-mini-instruct": {
                "max_length": 4096,
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
                "device_map": "auto"
            },
            "meta-llama/Llama-3.2-3B-Instruct": {
                "max_length": 8192,
                "trust_remote_code": False,
                "torch_dtype": torch.float16,
                "device_map": "auto"
            },
            "Qwen/Qwen2.5-7B-Instruct": {
                "max_length": 8192,
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
                "device_map": "auto"
            }
        }
    
    def _initialize(self):
        """Initialize the model and tokenizer"""
        try:
            logger.info(f"Initializing {self.model_name}...")
            
            # Get model-specific config
            config = self.model_configs.get(self.model_name, {
                "max_length": 4096,
                "trust_remote_code": False,
                "torch_dtype": self.torch_dtype,
                "device_map": "auto"
            })
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=config["trust_remote_code"],
                padding_side="left"
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=config["torch_dtype"],
                device_map=config["device_map"],
                trust_remote_code=config["trust_remote_code"],
                low_cpu_mem_usage=True,
                **self.kwargs
            )
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=config["torch_dtype"],
                device_map=config["device_map"]
            )
            
            self._initialized = True
            logger.info(f"Successfully initialized {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.model_name}: {str(e)}")
            self._initialized = False
            raise
    
    def _generate_raw(self, prompt: str, **kwargs) -> str:
        """Generate raw text response"""
        if not self._initialized:
            raise RuntimeError("Model not initialized")
        
        try:
            # Format prompt for chat models
            formatted_prompt = self._format_prompt(prompt)
            
            # Generation parameters
            generation_kwargs = {
                "max_new_tokens": kwargs.get("max_tokens", 1024),
                "temperature": kwargs.get("temperature", 0.3),
                "do_sample": True,
                "top_p": kwargs.get("top_p", 0.9),
                "top_k": kwargs.get("top_k", 50),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
                "pad_token_id": self.tokenizer.eos_token_id,
                "return_full_text": False
            }
            
            # Generate response
            outputs = self.pipeline(
                formatted_prompt,
                **generation_kwargs
            )
            
            # Extract generated text
            if outputs and len(outputs) > 0:
                generated_text = outputs[0]["generated_text"]
                return generated_text.strip()
            else:
                raise RuntimeError("Empty response from model")
                
        except Exception as e:
            logger.error(f"Generation failed for {self.model_name}: {str(e)}")
            raise
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt based on model type"""
        if "Phi-3" in self.model_name:
            return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        elif "Llama" in self.model_name:
            return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif "Qwen" in self.model_name:
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            # Generic format
            return f"### Human: {prompt}\n\n### Assistant:"
    
    def is_available(self) -> bool:
        """Check if the model is available and working"""
        if not self._initialized:
            try:
                self._initialize()
            except Exception:
                return False
        
        try:
            # Quick test generation
            test_response = self._generate_raw("Hello", max_tokens=10)
            return len(test_response) > 0
        except Exception:
            return False
    
    def cleanup(self):
        """Clean up model resources"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        self._initialized = False
        logger.info(f"Cleaned up {self.model_name}")

class ModelFactory:
    """Factory for creating LLM providers"""
    
    SUPPORTED_MODELS = {
        "phi35": "microsoft/Phi-3.5-mini-instruct",
        "llama32": "meta-llama/Llama-3.2-3B-Instruct", 
        "qwen25": "Qwen/Qwen2.5-7B-Instruct"
    }
    
    @classmethod
    def create_model(cls, model_key: str, **kwargs) -> TransformersLLM:
        """Create a model instance by key"""
        if model_key not in cls.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model key: {model_key}. Supported: {list(cls.SUPPORTED_MODELS.keys())}")
        
        model_name = cls.SUPPORTED_MODELS[model_key]
        return TransformersLLM(model_name, **kwargs)
    
    @classmethod
    def get_all_models(cls) -> Dict[str, TransformersLLM]:
        """Create all supported models"""
        models = {}
        for key in cls.SUPPORTED_MODELS:
            try:
                models[key] = cls.create_model(key)
                logger.info(f"Created model: {key}")
            except Exception as e:
                logger.warning(f"Failed to create model {key}: {str(e)}")
        
        return models
