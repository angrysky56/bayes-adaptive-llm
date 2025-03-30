#!/usr/bin/env python
"""
Dataset download and preprocessing script for Bayesian Self-Adaptive LLM.

This script downloads and preprocesses datasets for training different expert vectors:
- Math: Simplified GSM8K samples
- Code: MBPP (Mostly Basic Python Programming) samples
- Reasoning: ARC (AI2 Reasoning Challenge) samples
"""

import os
import json
import random
import argparse
import logging
import requests
import zipfile
import io
from pathlib import Path
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("download_datasets")

# Base directories
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Ensure directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Dataset URLs
DATASET_URLS = {
    "gsm8k": "https://raw.githubusercontent.com/openai/grade-school-math/refs/heads/master/grade_school_math/data/train_socratic.jsonl",
    "mbpp": "https://github.com/google-research/google-research/raw/master/mbpp/mbpp.jsonl",
    "arc_easy": "https://s3-us-west-2.amazonaws.com/ai2-website/data/ARC-V1-Feb2018.zip",
    "arc_challenge": "https://s3-us-west-2.amazonaws.com/ai2-website/data/ARC-V1-Feb2018.zip"
}

def download_file(url, save_path):
    """Download a file from URL to save_path."""
    logger.info(f"Downloading {url} to {save_path}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    
    with open(save_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=save_path.name) as pbar:
            for data in response.iter_content(block_size):
                pbar.update(len(data))
                f.write(data)
    
    return save_path

def extract_zip(zip_path, extract_dir):
    """Extract a ZIP file to extract_dir."""
    logger.info(f"Extracting {zip_path} to {extract_dir}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

def download_from_url(url, save_dir):
    """Download a file from URL to save_dir."""
    filename = url.split('/')[-1]
    save_path = save_dir / filename
    
    if save_path.exists():
        logger.info(f"File already exists: {save_path}")
        return save_path
    
    return download_file(url, save_path)

def process_gsm8k(input_path, train_output_path, eval_output_path, max_samples=1000, eval_split=0.2):
    """Process GSM8K dataset."""
    logger.info(f"Processing GSM8K dataset: {input_path}")
    
    samples = []
    with open(input_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    # Take a subset for faster training
    random.shuffle(samples)
    samples = samples[:max_samples]
    
    # Split into train/eval
    split_idx = int(len(samples) * (1 - eval_split))
    train_samples = samples[:split_idx]
    eval_samples = samples[split_idx:]
    
    # Convert to the expected format
    processed_train = []
    processed_eval = []
    
    for sample in train_samples:
        question = sample['question']
        answer = sample['answer'].split('####')[-1].strip()
        
        try:
            target = int(float(answer))
            processed_train.append({
                "input_ids": [101] + [random.randint(1000, 30000) for _ in range(len(question.split()))] + [102],
                "attention_mask": [1] * (len(question.split()) + 2),
                "target": target
            })
        except:
            pass  # Skip samples with non-numeric answers
    
    for sample in eval_samples:
        question = sample['question']
        answer = sample['answer'].split('####')[-1].strip()
        
        try:
            target = int(float(answer))
            processed_eval.append({
                "input_ids": [101] + [random.randint(1000, 30000) for _ in range(len(question.split()))] + [102],
                "attention_mask": [1] * (len(question.split()) + 2),
                "target": target
            })
        except:
            pass  # Skip samples with non-numeric answers
    
    # Write to output files
    with open(train_output_path, 'w') as f:
        for sample in processed_train:
            f.write(json.dumps(sample) + '\n')
    
    with open(eval_output_path, 'w') as f:
        for sample in processed_eval:
            f.write(json.dumps(sample) + '\n')
    
    logger.info(f"Processed {len(processed_train)} training samples, {len(processed_eval)} evaluation samples")

def process_mbpp(input_path, train_output_path, eval_output_path, max_samples=1000, eval_split=0.2):
    """Process MBPP dataset."""
    logger.info(f"Processing MBPP dataset: {input_path}")
    
    # Parse JSONL file (each line is a separate JSON object)
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line.strip()))
    
    # Take a subset for faster training
    random.shuffle(data)
    samples = data[:max_samples]
    
    # Split into train/eval
    split_idx = int(len(samples) * (1 - eval_split))
    train_samples = samples[:split_idx]
    eval_samples = samples[split_idx:]
    
    # Convert to the expected format
    processed_train = []
    processed_eval = []
    
    for i, sample in enumerate(train_samples):
        task = sample['text']
        code = sample['code']
        
        processed_train.append({
            "input_ids": [101] + [random.randint(1000, 30000) for _ in range(len(task.split()))] + [102],
            "attention_mask": [1] * (len(task.split()) + 2),
            "target": i % 100  # Use a dummy target
        })
    
    for i, sample in enumerate(eval_samples):
        task = sample['text']
        code = sample['code']
        
        processed_eval.append({
            "input_ids": [101] + [random.randint(1000, 30000) for _ in range(len(task.split()))] + [102],
            "attention_mask": [1] * (len(task.split()) + 2),
            "target": i % 100  # Use a dummy target
        })
    
    # Write to output files
    with open(train_output_path, 'w') as f:
        for sample in processed_train:
            f.write(json.dumps(sample) + '\n')
    
    with open(eval_output_path, 'w') as f:
        for sample in processed_eval:
            f.write(json.dumps(sample) + '\n')
    
    logger.info(f"Processed {len(processed_train)} training samples, {len(processed_eval)} evaluation samples")

def process_arc(input_dir, train_output_path, eval_output_path, max_samples=1000, eval_split=0.2):
    """Process ARC dataset."""
    logger.info(f"Processing ARC dataset: {input_dir}")
    
    train_dir = input_dir / "train"
    samples = []
    
    for json_file in train_dir.glob("*.jsonl"):
        with open(json_file, 'r') as f:
            for line in f:
                samples.append(json.loads(line))
    
    # Take a subset for faster training
    random.shuffle(samples)
    samples = samples[:max_samples]
    
    # Split into train/eval
    split_idx = int(len(samples) * (1 - eval_split))
    train_samples = samples[:split_idx]
    eval_samples = samples[split_idx:]
    
    # Convert to the expected format
    processed_train = []
    processed_eval = []
    
    for sample in train_samples:
        question = sample['question']['stem']
        options = sample['question']['choices']
        correct_idx = next(i for i, choice in enumerate(options) if choice['label'] == sample['answerKey'])
        
        processed_train.append({
            "input_ids": [101] + [random.randint(1000, 30000) for _ in range(len(question.split()))] + [102],
            "attention_mask": [1] * (len(question.split()) + 2),
            "target": correct_idx
        })
    
    for sample in eval_samples:
        question = sample['question']['stem']
        options = sample['question']['choices']
        correct_idx = next(i for i, choice in enumerate(options) if choice['label'] == sample['answerKey'])
        
        processed_eval.append({
            "input_ids": [101] + [random.randint(1000, 30000) for _ in range(len(question.split()))] + [102],
            "attention_mask": [1] * (len(question.split()) + 2),
            "target": correct_idx
        })
    
    # Write to output files
    with open(train_output_path, 'w') as f:
        for sample in processed_train:
            f.write(json.dumps(sample) + '\n')
    
    with open(eval_output_path, 'w') as f:
        for sample in processed_eval:
            f.write(json.dumps(sample) + '\n')
    
    logger.info(f"Processed {len(processed_train)} training samples, {len(processed_eval)} evaluation samples")

def create_dummy_data(num_samples=100):
    """Create dummy data for testing."""
    logger.info("Creating dummy math dataset")
    
    # Create dummy math data
    with open(PROCESSED_DIR / "math_train.jsonl", 'w') as f:
        for i in range(num_samples):
            sample = {
                "input_ids": [101] + [random.randint(1000, 30000) for _ in range(10)] + [102],
                "attention_mask": [1] * 12,
                "target": random.randint(0, 100)
            }
            f.write(json.dumps(sample) + '\n')
    
    with open(PROCESSED_DIR / "math_eval.jsonl", 'w') as f:
        for i in range(num_samples // 5):
            sample = {
                "input_ids": [101] + [random.randint(1000, 30000) for _ in range(10)] + [102],
                "attention_mask": [1] * 12,
                "target": random.randint(0, 100)
            }
            f.write(json.dumps(sample) + '\n')
    
    logger.info("Creating dummy code dataset")
    
    # Create dummy code data
    with open(PROCESSED_DIR / "code_train.jsonl", 'w') as f:
        for i in range(num_samples):
            sample = {
                "input_ids": [101] + [random.randint(1000, 30000) for _ in range(15)] + [102],
                "attention_mask": [1] * 17,
                "target": random.randint(0, 100)
            }
            f.write(json.dumps(sample) + '\n')
    
    with open(PROCESSED_DIR / "code_eval.jsonl", 'w') as f:
        for i in range(num_samples // 5):
            sample = {
                "input_ids": [101] + [random.randint(1000, 30000) for _ in range(15)] + [102],
                "attention_mask": [1] * 17,
                "target": random.randint(0, 100)
            }
            f.write(json.dumps(sample) + '\n')
    
    logger.info("Creating dummy reasoning dataset")
    
    # Create dummy reasoning data
    with open(PROCESSED_DIR / "reasoning_train.jsonl", 'w') as f:
        for i in range(num_samples):
            sample = {
                "input_ids": [101] + [random.randint(1000, 30000) for _ in range(12)] + [102],
                "attention_mask": [1] * 14,
                "target": random.randint(0, 5)
            }
            f.write(json.dumps(sample) + '\n')
    
    with open(PROCESSED_DIR / "reasoning_eval.jsonl", 'w') as f:
        for i in range(num_samples // 5):
            sample = {
                "input_ids": [101] + [random.randint(1000, 30000) for _ in range(12)] + [102],
                "attention_mask": [1] * 14,
                "target": random.randint(0, 5)
            }
            f.write(json.dumps(sample) + '\n')

def download_and_process_datasets(use_dummy=False):
    """Download and process all datasets."""
    if use_dummy:
        logger.info("Creating dummy datasets for testing")
        create_dummy_data()
        return
    
    # GSM8K (Math)
    gsm8k_path = download_from_url(DATASET_URLS["gsm8k"], RAW_DIR)
    process_gsm8k(
        gsm8k_path,
        PROCESSED_DIR / "math_train.jsonl",
        PROCESSED_DIR / "math_eval.jsonl"
    )
    
    # MBPP (Code)
    mbpp_path = download_from_url(DATASET_URLS["mbpp"], RAW_DIR)
    process_mbpp(
        mbpp_path,
        PROCESSED_DIR / "code_train.jsonl",
        PROCESSED_DIR / "code_eval.jsonl"
    )
    
    # ARC-Easy (Reasoning)
    arc_easy_zip = download_from_url(DATASET_URLS["arc_easy"], RAW_DIR)
    arc_easy_dir = RAW_DIR / "arc_easy"
    arc_easy_dir.mkdir(exist_ok=True)
    extract_zip(arc_easy_zip, arc_easy_dir)
    process_arc(
        arc_easy_dir,
        PROCESSED_DIR / "reasoning_train.jsonl",
        PROCESSED_DIR / "reasoning_eval.jsonl"
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download and process datasets")
    parser.add_argument(
        "--dummy", 
        action="store_true", 
        help="Create dummy datasets for testing"
    )
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    download_and_process_datasets(use_dummy=args.dummy)
    
    logger.info("All datasets processed!")
    logger.info(f"Datasets saved to {PROCESSED_DIR}")

if __name__ == "__main__":
    main()
