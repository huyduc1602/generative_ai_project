"""
API server for the generative AI model.
"""

import os
import argparse
import yaml
import torch
from transformers import AutoTokenizer
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import logging

from src.models.model import GenerativeModel
from src.inference.predict import load_model, generate_text

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Generative AI API", description="API for text generation")

# Global variables
model = None
tokenizer = None


class GenerationRequest(BaseModel):
    """Request model for text generation."""
    text: str
    max_length: int = 100
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95


class GenerationResponse(BaseModel):
    """Response model for text generation."""
    generated_text: str


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def initialize_model(model_path):
    """Initialize the model and tokenizer."""
    global model, tokenizer
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path).to(device)
    model.eval()
    
    # Load tokenizer
    model_config_path = "configs/model_config.yaml"  # Adjust as needed
    model_config = load_config(model_config_path)
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"])
    
    logger.info("Model and tokenizer initialized successfully")


@app.get("/")
def read_root():
    """Root endpoint."""
    return {"status": "ok", "message": "Generative AI API is running"}


@app.get("/health")
def health_check():
    """Health check endpoint."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {"status": "ok", "model_loaded": True}


@app.post("/generate", response_model=GenerationResponse)
def generate(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Generate text based on the input."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        generated_text = generate_text(
            model,
            tokenizer,
            request.text,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )
        
        return {"generated_text": generated_text}
    except Exception as e:
        logger.error(f"Error during text generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    parser = argparse.ArgumentParser(description='Serve the generative AI model via API')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--host', type=str, default="0.0.0.0",
                        help='Host to serve the API on')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to serve the API on')
    args = parser.parse_args()
    
    # Initialize model
    initialize_model(args.model_path)
    
    # Start the API server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
