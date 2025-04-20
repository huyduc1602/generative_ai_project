"""
Training script for the generative AI model.
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import logging

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


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_datasets(config):
    """Load training and validation datasets."""
    data_dir = config["data_dir"]
    
    # Load tokenizer
    model_config_path = os.path.join(os.path.dirname(config["config_path"]), "../configs/model_config.yaml")
    model_config = load_config(model_config_path)
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"])
    
    # Load data
    # This is a placeholder - in a real project, you would load your processed data
    # Example:
    # train_texts = pd.read_csv(os.path.join(data_dir, "train.csv"))["text"].tolist()
    # val_texts = pd.read_csv(os.path.join(data_dir, "val.csv"))["text"].tolist()
    
    train_texts = ["Sample text for training"] * 100  # Placeholder
    val_texts = ["Sample text for validation"] * 20  # Placeholder
    
    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer)
    val_dataset = TextDataset(val_texts, tokenizer)
    
    return train_dataset, val_dataset, tokenizer


def train(config):
    """Train the model using the provided configuration."""
    # Set seed for reproducibility
    set_seed(config["seed"])
    
    # Set device
    device = torch.device(config["device"] if torch.cuda.is_available() and config["device"] == "cuda" else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load datasets
    train_dataset, val_dataset, tokenizer = load_datasets(config)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )
    
    # Load model configuration
    model_config_path = os.path.join(os.path.dirname(config["config_path"]), "../configs/model_config.yaml")
    model_config = load_config(model_config_path)
    
    # Initialize model
    model = GenerativeModel(model_config).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    
    # Initialize scheduler
    total_steps = len(train_loader) * config["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=total_steps,
    )
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(config["num_epochs"]):
        logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
        
        # Training
        model.train()
        train_loss = 0
        train_steps = 0
        
        for step, batch in enumerate(tqdm(train_loader, desc="Training")):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["input_ids"],  # For language modeling, targets are the same as inputs
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            
            # Optimization step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            train_steps += 1
            
            if step % 100 == 0:
                logger.info(f"Step: {step}, Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / train_steps
        logger.info(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["input_ids"],
                )
                
                loss = outputs.loss
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        logger.info(f"Average validation loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        if not os.path.exists(config["output_dir"]):
            os.makedirs(config["output_dir"])
        
        checkpoint_path = os.path.join(config["output_dir"], f"model_epoch_{epoch + 1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(config["output_dir"], "model_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
    }, final_model_path)
    
    logger.info(f"Training completed. Final model saved to {final_model_path}")


def main():
    parser = argparse.ArgumentParser(description='Train a generative AI model')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                        help='Path to the training configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config["config_path"] = args.config
    
    # Train the model
    train(config)


if __name__ == "__main__":
    main()
