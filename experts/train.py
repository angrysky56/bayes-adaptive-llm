"""
Expert Training

This module provides functions for training SVF experts
using reinforcement learning or supervised fine-tuning.
"""

import os
import time
import json
from typing import Dict, List, Optional, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ..svf.svf import SVFFinetuner
from .manage import ExpertManager


class ExpertTrainer:
    """
    Trainer for SVF experts
    
    This class provides functionality for training experts
    using reinforcement learning or supervised fine-tuning.
    """
    
    def __init__(
        self, 
        model: nn.Module,
        finetuner: SVFFinetuner,
        expert_manager: ExpertManager,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize an expert trainer
        
        Args:
            model: Model to fine-tune
            finetuner: SVF finetuner managing the experts
            expert_manager: Expert manager
            device: Device to use for training
        """
        self.model = model
        self.finetuner = finetuner
        self.expert_manager = expert_manager
        self.device = device
        
        # Move model to device
        self.model.to(self.device)
    
    def train_supervised(
        self,
        expert_name: str,
        dataset: Dataset,
        loss_fn: nn.Module,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        num_epochs: int = 10,
        val_dataset: Optional[Dataset] = None,
        val_interval: int = 1,
        patience: int = 3,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Train an expert using supervised fine-tuning
        
        Args:
            expert_name: Name of the expert
            dataset: Training dataset
            loss_fn: Loss function
            batch_size: Batch size
            learning_rate: Learning rate
            num_epochs: Number of epochs
            val_dataset: Optional validation dataset
            val_interval: Validation interval in epochs
            patience: Early stopping patience
            metadata: Optional metadata to store with the expert
            
        Returns:
            Training history
        """
        # Create expert if it doesn't exist
        if expert_name not in self.finetuner.adapters:
            self.finetuner.create_expert(expert_name)
        
        # Get trainable parameters
        params = self.finetuner.get_trainable_parameters(expert_name)
        
        # Create optimizer
        optimizer = optim.Adam(params, lr=learning_rate)
        
        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )
        
        # Initialize training history
        history = {
            "train_loss": [],
            "val_loss": [] if val_dataset else None,
            "best_val_loss": float("inf") if val_dataset else None,
            "best_epoch": 0,
        }
        
        # Initialize early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = {
            param.detach().cpu().clone() for param in params
        }
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0.0
            
            with tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                for batch in pbar:
                    # Move batch to device
                    if isinstance(batch, torch.Tensor):
                        batch = batch.to(self.device)
                    elif isinstance(batch, (tuple, list)):
                        batch = [
                            item.to(self.device) if isinstance(item, torch.Tensor) else item
                            for item in batch
                        ]
                    elif isinstance(batch, dict):
                        batch = {
                            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                            for key, value in batch.items()
                        }
                    
                    # Forward pass
                    outputs = self.model(batch)
                    
                    # Compute loss
                    loss = loss_fn(outputs, batch)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Update progress bar
                    epoch_loss += loss.item()
                    pbar.set_postfix(loss=epoch_loss / (pbar.n + 1))
            
            # Record training loss
            train_loss = epoch_loss / len(data_loader)
            history["train_loss"].append(train_loss)
            
            # Validation phase
            if val_dataset and (epoch + 1) % val_interval == 0:
                val_loss = self._validate(val_loader, loss_fn)
                history["val_loss"].append(val_loss)
                
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_loss:.4f} - "
                      f"Val Loss: {val_loss:.4f}")
                
                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    history["best_val_loss"] = val_loss
                    history["best_epoch"] = epoch + 1
                    
                    # Reset patience counter
                    patience_counter = 0
                    
                    # Save best state
                    best_state = {
                        param.detach().cpu().clone() for param in params
                    }
                else:
                    # Increment patience counter
                    patience_counter += 1
                    
                    # Check for early stopping
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
        
        # Restore best state if validation was used
        if val_dataset:
            for i, param in enumerate(params):
                param.data = best_state[i].to(self.device)
        
        # Save expert
        metadata = metadata or {}
        metadata.update({
            "training_method": "supervised",
            "train_history": history,
            "timestamp": time.time(),
        })
        
        self.expert_manager.save_expert(expert_name, metadata)
        
        return history
    
    def train_rl(
        self,
        expert_name: str,
        dataset: Dataset,
        reward_fn: Callable,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        num_epochs: int = 10,
        val_dataset: Optional[Dataset] = None,
        val_interval: int = 1,
        patience: int = 3,
        kl_coef: float = 0.1,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Train an expert using reinforcement learning
        
        Args:
            expert_name: Name of the expert
            dataset: Training dataset
            reward_fn: Function that computes rewards
            batch_size: Batch size
            learning_rate: Learning rate
            num_epochs: Number of epochs
            val_dataset: Optional validation dataset
            val_interval: Validation interval in epochs
            patience: Early stopping patience
            kl_coef: KL divergence coefficient for regularization
            metadata: Optional metadata to store with the expert
            
        Returns:
            Training history
        """
        # Create expert if it doesn't exist
        if expert_name not in self.finetuner.adapters:
            self.finetuner.create_expert(expert_name)
        
        # Get trainable parameters
        params = self.finetuner.get_trainable_parameters(expert_name)
        
        # Create optimizer
        optimizer = optim.Adam(params, lr=learning_rate)
        
        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )
        
        # Initialize training history
        history = {
            "train_reward": [],
            "val_reward": [] if val_dataset else None,
            "best_val_reward": float("-inf") if val_dataset else None,
            "best_epoch": 0,
        }
        
        # Initialize early stopping
        best_val_reward = float("-inf")
        patience_counter = 0
        best_state = {
            param.detach().cpu().clone() for param in params
        }
        
        # Clone base model for KL regularization
        base_model = type(self.model)(**self.model.config.__dict__)
        base_model.load_state_dict(self.model.state_dict())
        base_model.to(self.device)
        base_model.eval()
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            epoch_reward = 0.0
            
            with tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                for batch in pbar:
                    # Move batch to device
                    if isinstance(batch, torch.Tensor):
                        batch = batch.to(self.device)
                    elif isinstance(batch, (tuple, list)):
                        batch = [
                            item.to(self.device) if isinstance(item, torch.Tensor) else item
                            for item in batch
                        ]
                    elif isinstance(batch, dict):
                        batch = {
                            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                            for key, value in batch.items()
                        }
                    
                    # Forward pass with adapted model
                    outputs = self.model(batch)
                    
                    # Compute reward
                    reward = reward_fn(outputs, batch)
                    
                    # Compute KL divergence for regularization
                    with torch.no_grad():
                        base_outputs = base_model(batch)
                    
                    kl_div = F.kl_div(
                        F.log_softmax(outputs, dim=-1),
                        F.softmax(base_outputs, dim=-1),
                        reduction="batchmean"
                    )
                    
                    # Compute loss
                    loss = -reward + kl_coef * kl_div
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Update progress bar
                    epoch_reward += reward.item()
                    pbar.set_postfix(reward=epoch_reward / (pbar.n + 1))
            
            # Record training reward
            train_reward = epoch_reward / len(data_loader)
            history["train_reward"].append(train_reward)
            
            # Validation phase
            if val_dataset and (epoch + 1) % val_interval == 0:
                val_reward = self._validate_rl(val_loader, reward_fn)
                history["val_reward"].append(val_reward)
                
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Reward: {train_reward:.4f} - "
                      f"Val Reward: {val_reward:.4f}")
                
                # Check for improvement
                if val_reward > best_val_reward:
                    best_val_reward = val_reward
                    history["best_val_reward"] = val_reward
                    history["best_epoch"] = epoch + 1
                    
                    # Reset patience counter
                    patience_counter = 0
                    
                    # Save best state
                    best_state = {
                        param.detach().cpu().clone() for param in params
                    }
                else:
                    # Increment patience counter
                    patience_counter += 1
                    
                    # Check for early stopping
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Reward: {train_reward:.4f}")
        
        # Restore best state if validation was used
        if val_dataset:
            for i, param in enumerate(params):
                param.data = best_state[i].to(self.device)
        
        # Save expert
        metadata = metadata or {}
        metadata.update({
            "training_method": "reinforcement_learning",
            "train_history": history,
            "timestamp": time.time(),
        })
        
        self.expert_manager.save_expert(expert_name, metadata)
        
        return history
    
    def _validate(self, val_loader: DataLoader, loss_fn: nn.Module) -> float:
        """
        Validate the model
        
        Args:
            val_loader: Validation data loader
            loss_fn: Loss function
            
        Returns:
            Validation loss
        """
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                if isinstance(batch, torch.Tensor):
                    batch = batch.to(self.device)
                elif isinstance(batch, (tuple, list)):
                    batch = [
                        item.to(self.device) if isinstance(item, torch.Tensor) else item
                        for item in batch
                    ]
                elif isinstance(batch, dict):
                    batch = {
                        key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                        for key, value in batch.items()
                    }
                
                # Forward pass
                outputs = self.model(batch)
                
                # Compute loss
                loss = loss_fn(outputs, batch)
                
                # Accumulate loss
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def _validate_rl(self, val_loader: DataLoader, reward_fn: Callable) -> float:
        """
        Validate the model with a reward function
        
        Args:
            val_loader: Validation data loader
            reward_fn: Reward function
            
        Returns:
            Validation reward
        """
        self.model.eval()
        val_reward = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                if isinstance(batch, torch.Tensor):
                    batch = batch.to(self.device)
                elif isinstance(batch, (tuple, list)):
                    batch = [
                        item.to(self.device) if isinstance(item, torch.Tensor) else item
                        for item in batch
                    ]
                elif isinstance(batch, dict):
                    batch = {
                        key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                        for key, value in batch.items()
                    }
                
                # Forward pass
                outputs = self.model(batch)
                
                # Compute reward
                reward = reward_fn(outputs, batch)
                
                # Accumulate reward
                val_reward += reward.item()
        
        return val_reward / len(val_loader)
