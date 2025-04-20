"""
Model architecture definitions for the generative AI project.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig


def load_model_from_config(config):
    """
    Load a model based on configuration.
    
    Args:
        config (dict): Model configuration parameters.
        
    Returns:
        model: Initialized model.
    """
    model_type = config.get("model_type", "transformer")
    model_name = config.get("model_name", "gpt2")
    
    if model_type == "transformer":
        model_config = AutoConfig.from_pretrained(
            model_name,
            hidden_size=config.get("hidden_size", 768),
            num_hidden_layers=config.get("num_layers", 12),
            num_attention_heads=config.get("num_heads", 12),
            dropout=config.get("dropout", 0.1),
            vocab_size=config.get("vocab_size", 50257),
        )
        model = AutoModelForCausalLM.from_config(model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    return model


class GenerativeModel(nn.Module):
    """
    Wrapper class for generative models.
    """
    
    def __init__(self, config):
        """
        Initialize the model.
        
        Args:
            config (dict): Model configuration parameters.
        """
        super(GenerativeModel, self).__init__()
        self.config = config
        self.model = load_model_from_config(config)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass.
        
        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Target labels.
            
        Returns:
            dict: Model outputs.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        
        return outputs
    
    def generate(self, input_ids, attention_mask=None, max_length=100, **kwargs):
        """
        Generate text.
        
        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            max_length (int): Maximum length of generated sequence.
            **kwargs: Additional arguments for generation.
            
        Returns:
            torch.Tensor: Generated token IDs.
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            **kwargs,
        )
