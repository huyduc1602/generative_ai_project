"""
Inference script for the generative AI model.
"""

import os
import argparse
import yaml
import torch
from transformers import AutoTokenizer
import logging

from src.models.model import GenerativeModel

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_path):
    """Load a trained model from checkpoint."""
    logger.info(f"Loading model from {model_path}")
    
    # Load model configuration
    model_config_path = "configs/model_config.yaml"  # Adjust as needed
    model_config = load_config(model_config_path)
    
    # Initialize model
    model = GenerativeModel(model_config)
    
    # Load saved weights
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def generate_text(model, tokenizer, input_text, max_length=100, **kwargs):
    """
    Generate text from the given input.
    
    Args:
        model: The generative model.
        tokenizer: The tokenizer.
        input_text (str): Input text to continue from.
        max_length (int): Maximum length of generated text.
        **kwargs: Additional generation parameters.
        
    Returns:
        str: Generated text.
    """
    # Encode input text
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Set default generation parameters if not provided
    generation_kwargs = {
        "max_length": max_length,
        "num_return_sequences": 1,
        "no_repeat_ngram_size": 2,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.8,
        **kwargs,
    }
    
    # Generate text
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        **generation_kwargs,
    )
    
    # Decode the generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(description='Generate text with a pretrained model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--input', type=str, required=True,
                        help='Input text for generation')
    parser.add_argument('--max_length', type=int, default=100,
                        help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (higher=more random)')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p (nucleus) sampling parameter')
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path).to(device)
    model.eval()
    
    # Load tokenizer
    model_config_path = "configs/model_config.yaml"  # Adjust as needed
    model_config = load_config(model_config_path)
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"])
    
    # Generate text
    generated_text = generate_text(
        model,
        tokenizer,
        args.input,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    
    # Print the generated text
    print(f"\nInput: {args.input}")
    print(f"\nGenerated text: {generated_text}")


if __name__ == "__main__":
    main()
