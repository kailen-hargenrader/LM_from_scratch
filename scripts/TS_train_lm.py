"""
Training script for the Transformer Language Model.

Features:
- Configurable model and optimizer hyperparameters via CLI/config
- Memory-efficient data loading using np.memmap for large datasets
- Checkpoint saving and loading
- Periodic logging to console and Weights & Biases
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from basicformer.model.transformer import TransformerLM
from basicformer.loss.cross_entropy import CrossEntropy
from data.TS_dataloader import get_batch, load_tiny_stories_train_ids, load_tiny_stories_valid_ids, load_tiny_stories_minibatch_ids


GET_TRAIN_IDS = load_tiny_stories_train_ids
GET_VALID_IDS = load_tiny_stories_valid_ids

class TrainingConfig:
    """Configuration container for training hyperparameters."""
    
    def __init__(self, **kwargs):
        """Initialize config from keyword arguments or defaults."""
        # Model architecture
        self.vocab_size = kwargs.get('vocab_size', 256)
        self.context_length = kwargs.get('context_length', 128)
        self.d_model = kwargs.get('d_model', 256)
        self.num_layers = kwargs.get('num_layers', 4)
        self.num_heads = kwargs.get('num_heads', 8)
        self.d_ff = kwargs.get('d_ff', 1024)
        self.norm = kwargs.get('norm', True)
        
        # Optimization
        self.learning_rate = kwargs.get('learning_rate', 5e-4)
        self.batch_size = kwargs.get('batch_size', 64)
        self.num_epochs = kwargs.get('num_epochs', 10)
        self.weight_decay = kwargs.get('weight_decay', 0.01)
        
        # Checkpointing and logging
        self.checkpoint_dir = kwargs.get('checkpoint_dir', 'checkpoints')
        self.log_interval = kwargs.get('log_interval', 100)
        self.val_interval = kwargs.get('val_interval', 500)
        self.checkpoint_interval = kwargs.get('checkpoint_interval', 1000)
        
        # Device and precision
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = getattr(torch, kwargs.get('dtype', 'float32'))
        
        # Logging
        self.use_wandb = kwargs.get('use_wandb', False)
        self.wandb_project = kwargs.get('wandb_project', 'transformer-lm')
        self.wandb_entity = kwargs.get('wandb_entity', None)
        self.run_name = kwargs.get('run_name', None)
    
    @classmethod
    def from_json(cls, path: str):
        """Load config from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, path: str):
        """Save config to JSON file."""
        config_dict = self.__dict__
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return self.__dict__.copy()


class Trainer:
    """Main training loop manager."""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer with configuration.
        
        Args:
            config: TrainingConfig instance
        """
        self.config = config
        self.device = torch.device(config.device)
        self.step = 0
        self.epoch = 0
        
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        self._init_model()
        self._init_optimizer()
        self._init_loss()
        self._init_datasets()
        self._init_logging()
    
    def _init_model(self):
        """Initialize the Transformer LM."""
        self.model = TransformerLM(
            vocab_size=self.config.vocab_size,
            context_length=self.config.context_length,
            d_model=self.config.d_model,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            d_ff=self.config.d_ff,
            device=self.device,
            dtype=self.config.dtype,
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model initialized: {total_params:,} total params, {trainable_params:,} trainable")
    
    def _init_optimizer(self):
        """Initialize optimizer."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
    
    def _init_loss(self):
        """Initialize loss function."""
        self.loss_fn = CrossEntropy()
    
    def _init_datasets(self):
        """Initialize train and validation datasets."""
        try:
            self.train_dataset = GET_TRAIN_IDS()
            print(f"Training dataset loaded: {len(self.train_dataset):,} tokens")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            self.train_dataset = None
        
        try:
            self.val_dataset = GET_VALID_IDS()
            print(f"Validation dataset loaded: {len(self.val_dataset):,} tokens")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            self.val_dataset = None
        
        self._validate_vocab_size()
    
    def _validate_vocab_size(self):
        """
        Validate that the maximum token ID in the dataset matches the configured vocab size.
        Raises an error if there's a mismatch.
        """
        if self.train_dataset is None:
            return
        
        max_token_id = int(self.train_dataset.max())
        expected_vocab_size = self.config.vocab_size
        
        if max_token_id >= expected_vocab_size:
            raise ValueError(
                f"ERROR: Data contains token IDs up to {max_token_id}, but vocab_size is configured as {expected_vocab_size}. "
                f"This will cause index out of bounds errors during training. "
                f"Please update vocab_size in your config to at least {max_token_id + 1}."
            )
    
    def _init_logging(self):
        """Initialize logging (console and optional Weights & Biases)."""
        self.wandb = None
        if self.config.use_wandb:
            try:
                import wandb
                self.wandb = wandb
                
                # Set API key from environment variable if available
                wandb_api_key = os.environ.get('WANDB_API_KEY')
                if not wandb_api_key:
                    print("Wandb API key not found in environment variables, set with `export WANDB_API_KEY=<your_api_key>`.")
                wandb.login(key=wandb_api_key, verify=True)
                
                run_name = self.config.run_name or f"transformer-lm-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                self.wandb.init(
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    name=run_name,
                    config=self.config.to_dict(),
                )
                print(f"Weights & Biases logging initialized: {run_name}")
            except ImportError:
                print("Warning: wandb not installed, logging to console only")
                self.config.use_wandb = False
    
    def train(self):
        """Run the full training loop."""
        if self.train_dataset is None:
            raise ValueError("Training dataset not available")
        
        print(f"\nStarting training for {self.config.num_epochs} epochs")
        print(f"Device: {self.device}, Dtype: {self.config.dtype}")
        print("-" * 80)
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            self.train_epoch()
        
        print("\nTraining complete!")
        self._log_metrics({"status": "training_complete"})
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        num_steps_per_epoch = max(1, len(self.train_dataset) // (
            self.config.batch_size * self.config.context_length
        ))
        
        for step in range(num_steps_per_epoch):
            loss = self.train_step()
            epoch_loss += loss.item()
            num_batches += 1
            self.step += 1
            
            if self.step % self.config.log_interval == 0:
                avg_loss = epoch_loss / max(1, num_batches)
                print(f"Epoch {self.epoch + 1}/{self.config.num_epochs} | "
                      f"Step {step}/{num_steps_per_epoch} | "
                      f"Total Step: {self.step}/{num_steps_per_epoch * self.config.num_epochs} | "
                      f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f})")
                
                self._log_metrics({
                    "train/loss": loss.item(),
                    "train/loss_avg": avg_loss,
                    "epoch": self.epoch + 1,
                    "step": self.step,
                })
            
            if self.step % self.config.checkpoint_interval == 0:
                self.save_checkpoint()
            
            if self.step % self.config.val_interval == 0 and self.val_dataset is not None:
                val_loss = self.validate()
                self._log_metrics({
                    "val/loss": val_loss,
                    "step": self.step,
                })
    
    def train_step(self) -> torch.Tensor:
        """Execute a single training step."""
        x, y = get_batch(
            self.train_dataset,
            self.config.batch_size,
            self.config.context_length,
            str(self.device),
        )
        
        logits = self.model(x, norm=self.config.norm)
        
        loss = self.loss_fn(logits, y)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss
    
    @torch.no_grad()
    def validate(self, num_batches: int = 10) -> float:
        """
        Evaluate model on validation set.
        
        Args:
            num_batches: Number of validation batches to evaluate
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        val_loss = 0.0
        
        for _ in range(num_batches):
            x, y = get_batch(
                self.val_dataset,
                self.config.batch_size,
                self.config.context_length,
                str(self.device),
            )
            
            logits = self.model(x, norm=self.config.norm)
            loss = self.loss_fn(logits, y)
            val_loss += loss.item()
        
        avg_val_loss = val_loss / num_batches
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        self.model.train()
        return avg_val_loss
    
    def save_checkpoint(self, name: Optional[str] = None):
        """
        Save model checkpoint.
        
        Args:
            name: Optional checkpoint name. Defaults to step-based naming.
        """
        if name is None:
            name = f"checkpoint-step-{self.step}.pt"
        
        checkpoint_path = os.path.join(self.config.checkpoint_dir, name)
        
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        self._log_metrics({"checkpoint": checkpoint_path})
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from step {self.step}, epoch {self.epoch}")
    
    def _log_metrics(self, metrics: dict):
        """
        Log metrics to console and Weights & Biases.
        
        Args:
            metrics: Dictionary of metrics to log
        """
        if self.wandb is not None and self.config.use_wandb:
            self.wandb.log(metrics, step=self.step)


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description='Train Transformer Language Model')
    
    # Model architecture
    parser.add_argument('--vocab-size', type=int, default=256, 
                        help='Vocabulary size')
    parser.add_argument('--context-length', type=int, default=128,
                        help='Context/sequence length')
    parser.add_argument('--d-model', type=int, default=256,
                        help='Model embedding dimension')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Number of Transformer blocks')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--d-ff', type=int, default=1024,
                        help='Feedforward hidden dimension')
    
    # Optimization
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay / L2 regularization')
    
    # Checkpointing and logging
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging interval (steps)')
    parser.add_argument('--val-interval', type=int, default=500,
                        help='Validation interval (steps)')
    parser.add_argument('--checkpoint-interval', type=int, default=1000,
                        help='Checkpoint saving interval (steps)')
    
    # Device and precision
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda/cpu)')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='Data type (float32/float16)')
    
    # W&B logging
    parser.add_argument('--use-wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='transformer-lm',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='Weights & Biases entity name')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Run name for W&B logging')
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON config file (overrides CLI args)')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    if args.config:
        config = TrainingConfig.from_json(args.config)
    else:
        config = TrainingConfig(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            weight_decay=args.weight_decay,
            checkpoint_dir=args.checkpoint_dir,
            log_interval=args.log_interval,
            val_interval=args.val_interval,
            checkpoint_interval=args.checkpoint_interval,
            device=args.device,
            dtype=args.dtype,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            run_name=args.run_name,
        )
    
    trainer = Trainer(config)
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    trainer.train()


if __name__ == '__main__':
    main()
