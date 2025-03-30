#!/usr/bin/env python
"""
Training script for multiple SVF expert vectors.

This script trains multiple domain-specific experts using Singular Value Fine-tuning (SVF)
on various tasks sequentially.
"""

import os
import sys
import argparse
import subprocess
import logging
import json
from pathlib import Path

# Add the repository root to the Python path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train_multiple_experts.log")
    ]
)
logger = logging.getLogger("train_multiple_experts")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train multiple SVF expert vectors")
    parser.add_argument(
        "--config-dir", 
        type=str, 
        default="config/experiments", 
        help="Directory containing configuration files"
    )
    parser.add_argument(
        "--configs", 
        type=str, 
        nargs="+", 
        help="List of configuration file names to run (without path)"
    )
    return parser.parse_args()

def train_expert(config_path):
    """Train a single expert using the specified configuration."""
    logger.info(f"Training expert with configuration: {config_path}")
    
    # Read configuration to get expert name for logging
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            expert_name = config.get("expert", {}).get("name", "unknown")
    except Exception as e:
        logger.error(f"Failed to read configuration file: {e}")
        return False
    
    # Create log file for this expert
    log_file = f"logs/{expert_name}_training.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Run training command - use the full path to the train.py script
    train_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "train_eval",
        "train.py"
    )
    cmd = [
        sys.executable, 
        train_script, 
        "--config", 
        config_path,
        "--log-file", 
        log_file
    ]
    
    try:
        logger.info(f"Starting training process for expert: {expert_name}")
        process = subprocess.run(
            cmd,
            check=True,
            stderr=subprocess.STDOUT
        )
        logger.info(f"Successfully trained expert: {expert_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed for expert {expert_name}: {e}")
        return False

def main():
    """Main function for training multiple experts."""
    args = parse_args()
    
    # If no configs specified, use all JSON files in the config directory
    if not args.configs:
        config_dir = Path(args.config_dir)
        configs = list(config_dir.glob("*.json"))
        config_paths = [str(config) for config in configs]
    else:
        # Use specified config files
        config_paths = [os.path.join(args.config_dir, config) for config in args.configs]
    
    # Check if config files exist
    for config_path in config_paths:
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            return
    
    # Train experts
    logger.info(f"Training {len(config_paths)} experts")
    
    for i, config_path in enumerate(config_paths):
        logger.info(f"Training expert {i+1}/{len(config_paths)}")
        success = train_expert(config_path)
        if not success:
            logger.error(f"Failed to train expert with configuration: {config_path}")
    
    logger.info("All expert training complete!")

if __name__ == "__main__":
    main()
