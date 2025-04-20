"""
Dataset classes for the generative AI project.
"""

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Dataset for text sequences."""
    
    def __init__(self, texts, tokenizer, max_length=512):
        """
        Initialize the dataset.
        
        Args:
            texts (list): List of text sequences.
            tokenizer: Tokenizer to use for encoding texts.
            max_length (int): Maximum sequence length.
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }


class GenerativeDataset(Dataset):
    """Dataset for generative models with input-output pairs."""
    
    def __init__(self, input_texts, target_texts, tokenizer, max_length=512):
        """
        Initialize the dataset.
        
        Args:
            input_texts (list): List of input text sequences.
            target_texts (list): List of target text sequences.
            tokenizer: Tokenizer to use for encoding texts.
            max_length (int): Maximum sequence length.
        """
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.input_texts)
    
    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        target_text = self.target_texts[idx]
        
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "input_ids": input_encoding["input_ids"].flatten(),
            "attention_mask": input_encoding["attention_mask"].flatten(),
            "labels": target_encoding["input_ids"].flatten(),
        }
