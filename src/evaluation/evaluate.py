"""
Evaluation script for the generative AI model.
"""

import os
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import logging
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import corpus_bleu

from src.models.model import GenerativeModel
from src.data.dataset import TextDataset, GenerativeDataset

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


def compute_metrics(references, hypotheses):
    """
    Compute evaluation metrics.
    
    Args:
        references (list): List of reference texts.
        hypotheses (list): List of generated texts.
        
    Returns:
        dict: Dictionary of evaluation metrics.
    """
    # Prepare for BLEU calculation
    tokenized_refs = [[ref.split()] for ref in references]
    tokenized_hyps = [hyp.split() for hyp in hypotheses]
    
    # Calculate BLEU score
    try:
        bleu_score = corpus_bleu(tokenized_refs, tokenized_hyps)
    except Exception as e:
        logger.error(f"Error computing BLEU score: {e}")
        bleu_score = 0.0
    
    # Calculate ROUGE scores
    rouge_metrics = ["rouge1", "rouge2", "rougeL"]
    scorer = rouge_scorer.RougeScorer(rouge_metrics, use_stemmer=True)
    
    rouge_scores = {metric: 0.0 for metric in rouge_metrics}
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        for metric in rouge_metrics:
            rouge_scores[metric] += scores[metric].fmeasure
    
    # Average ROUGE scores
    for metric in rouge_metrics:
        rouge_scores[metric] /= len(references)
    
    # Combine all metrics
    metrics = {
        "bleu": bleu_score,
        **rouge_scores
    }
    
    return metrics


def evaluate(model_path, test_data_path, output_dir=None):
    """
    Evaluate the model on test data.
    
    Args:
        model_path (str): Path to the trained model.
        test_data_path (str): Path to the test data.
        output_dir (str, optional): Directory to save evaluation results.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model(model_path).to(device)
    model.eval()
    
    # Load tokenizer
    model_config_path = "configs/model_config.yaml"  # Adjust as needed
    model_config = load_config(model_config_path)
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"])
    
    # Load test data
    # In a real project, this would load your actual test data
    # Example:
    # test_data = pd.read_csv(test_data_path)
    # input_texts = test_data["input"].tolist()
    # reference_texts = test_data["output"].tolist()
    
    # For demonstration, use placeholder data
    input_texts = ["Sample input text for testing"] * 20  # Placeholder
    reference_texts = ["Sample reference text for testing"] * 20  # Placeholder
    
    # Generate text
    logger.info("Generating text...")
    generated_texts = []
    
    with torch.no_grad():
        for input_text in tqdm(input_texts, desc="Generating"):
            # Tokenize input
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            
            # Generate
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=100,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                top_k=50,
                top_p=0.95,
            )
            
            # Decode output
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            generated_texts.append(generated_text)
    
    # Compute metrics
    logger.info("Computing evaluation metrics...")
    metrics = compute_metrics(reference_texts, generated_texts)
    
    # Print metrics
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    
    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        with open(os.path.join(output_dir, "metrics.yaml"), "w") as f:
            yaml.dump(metrics, f)
        
        # Save generated texts
        with open(os.path.join(output_dir, "generations.txt"), "w") as f:
            for i, (input_text, reference_text, generated_text) in enumerate(zip(input_texts, reference_texts, generated_texts)):
                f.write(f"Example {i+1}:\n")
                f.write(f"Input: {input_text}\n")
                f.write(f"Reference: {reference_text}\n")
                f.write(f"Generated: {generated_text}\n")
                f.write("\n---\n\n")
        
        logger.info(f"Evaluation results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate a generative AI model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to the test data')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    args = parser.parse_args()
    
    # Evaluate the model
    evaluate(args.model_path, args.test_data, args.output_dir)


if __name__ == "__main__":
    main()
