"""
Unit tests for the model.
"""

import os
import sys
import pytest
import torch

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model import GenerativeModel


def test_model_initialization():
    """Test that the model can be initialized."""
    config = {
        "model_type": "transformer",
        "model_name": "gpt2",
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "dropout": 0.1,
        "vocab_size": 50257,
    }
    
    model = GenerativeModel(config)
    assert model is not None
    assert hasattr(model, "model")


def test_model_forward():
    """Test the model's forward pass."""
    config = {
        "model_type": "transformer",
        "model_name": "gpt2",
        "hidden_size": 768,
        "num_layers": 2,  # Smaller model for testing
        "num_heads": 2,   # Smaller model for testing
        "dropout": 0.1,
        "vocab_size": 50257,
    }
    
    model = GenerativeModel(config)
    
    # Create dummy inputs
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
    
    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Check outputs
    assert "logits" in outputs
    assert outputs.logits.shape == (batch_size, seq_length, config["vocab_size"])


def test_model_generate():
    """Test the model's text generation."""
    config = {
        "model_type": "transformer",
        "model_name": "gpt2",
        "hidden_size": 768,
        "num_layers": 2,  # Smaller model for testing
        "num_heads": 2,   # Smaller model for testing
        "dropout": 0.1,
        "vocab_size": 50257,
    }
    
    model = GenerativeModel(config)
    
    # Create dummy inputs
    batch_size = 1
    seq_length = 5
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
    
    # Generate text
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=10,
    )
    
    # Check outputs
    assert generated_ids.shape[1] >= input_ids.shape[1]  # should generate at least as many tokens as input
