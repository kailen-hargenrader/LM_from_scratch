"""
Evaluation script for the Transformer Language Model.

Features:
- Load model from checkpoint
- Evaluate on validation/test dataset using perplexity metric
- Compute loss and perplexity statistics
- Support for batch evaluation
- Generate text from model checkpoint
"""

import os
import argparse
import json
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from basicformer.model.transformer import TransformerLM
from basicformer.loss.cross_entropy import CrossEntropy
from basicformer.loss.Perplexity import Perplexity
from data.TS_dataloader import get_batch, load_tiny_stories_valid_ids


class EvalConfig:
    """Configuration container for evaluation hyperparameters.
    
    Can be initialized from training config JSON files - the training-specific
    fields (learning_rate, num_epochs, weight_decay, etc.) are automatically ignored.
    """
    
    def __init__(self, **kwargs):
        """Initialize config from keyword arguments or defaults.
        
        Supports both evaluation-specific and training config files.
        Training-only fields are safely ignored.
        """
        # Model architecture (usually loaded from checkpoint)
        self.vocab_size = kwargs.get('vocab_size', 256)
        self.context_length = kwargs.get('context_length', 128)
        self.d_model = kwargs.get('d_model', 256)
        self.num_layers = kwargs.get('num_layers', 4)
        self.num_heads = kwargs.get('num_heads', 8)
        self.d_ff = kwargs.get('d_ff', 1024)
        
        # Evaluation settings
        self.batch_size = kwargs.get('batch_size', 64)
        self.num_batches = kwargs.get('num_batches', None)  # None = all batches
        
        # Device and precision
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = getattr(torch, kwargs.get('dtype', 'float32'))
    
    @classmethod
    def from_json(cls, path: str):
        """Load config from JSON file.
        
        Supports both evaluation-specific configs and training configs.
        Extra training fields are safely ignored.
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return self.__dict__.copy()


class Evaluator:
    """Model evaluation manager."""
    
    def __init__(self, config: EvalConfig, checkpoint_path: str, tokenizer_path: Optional[str] = None):
        """
        Initialize evaluator with configuration and checkpoint.
        
        Args:
            config: EvalConfig instance
            checkpoint_path: Path to checkpoint file
            tokenizer_path: Optional path to tokenizer pickle file for decoding generations
        """
        self.config = config
        self.device = torch.device(config.device)
        self.tokenizer = None
        
        self._init_model()
        self._load_checkpoint(checkpoint_path)
        self._init_loss()
        self._init_dataset()
        
        if tokenizer_path:
            self._load_tokenizer(tokenizer_path)
    
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
    
    def _load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Update config from checkpoint if available
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            self.config.vocab_size = saved_config.get('vocab_size', self.config.vocab_size)
            self.config.context_length = saved_config.get('context_length', self.config.context_length)
            self.config.d_model = saved_config.get('d_model', self.config.d_model)
            self.config.num_layers = saved_config.get('num_layers', self.config.num_layers)
            self.config.num_heads = saved_config.get('num_heads', self.config.num_heads)
            self.config.d_ff = saved_config.get('d_ff', self.config.d_ff)
            print(f"Config loaded from checkpoint")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from checkpoint: {checkpoint_path}")
        
        # Print training info if available
        if 'step' in checkpoint:
            print(f"Checkpoint from step {checkpoint['step']}, epoch {checkpoint.get('epoch', 'N/A')}")
    
    def _init_loss(self):
        """Initialize loss and perplexity functions."""
        self.loss_fn = CrossEntropy()
        self.perplexity_fn = Perplexity()
    
    def _init_dataset(self):
        """Initialize validation dataset."""
        try:
            self.val_dataset = load_tiny_stories_valid_ids()
            print(f"Validation dataset loaded: {len(self.val_dataset):,} tokens")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            self.val_dataset = None
    
    def _load_tokenizer(self, tokenizer_path: str):
        """
        Load tokenizer from pickle file.
        
        Args:
            tokenizer_path: Path to tokenizer pickle file
        """
        if not os.path.exists(tokenizer_path):
            print(f"Warning: Tokenizer not found at {tokenizer_path}, skipping tokenizer loading")
            return
        
        try:
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            print(f"Tokenizer loaded from: {tokenizer_path}")
        except Exception as e:
            print(f"Warning: Failed to load tokenizer: {e}")
            self.tokenizer = None
    
    def _decode_tokens(self, token_ids: list) -> str:
        """
        Decode a sequence of token IDs to text.
        
        Args:
            token_ids: List of token IDs
        
        Returns:
            Decoded text string
        """
        if self.tokenizer is None:
            return " ".join(str(tid) for tid in token_ids)
        
        try:
            # Assuming tokenizer has an 'itos' (id-to-string) mapping
            if hasattr(self.tokenizer, 'itos'):
                decoded_tokens = [self.tokenizer.itos.get(tid, f"<unk:{tid}>") for tid in token_ids]
                return "".join(decoded_tokens)
            elif hasattr(self.tokenizer, 'decode'):
                return self.tokenizer.decode(token_ids)
            else:
                return " ".join(str(tid) for tid in token_ids)
        except Exception as e:
            print(f"Warning: Failed to decode tokens: {e}")
            return " ".join(str(tid) for tid in token_ids)
    
    
    @torch.no_grad()
    def evaluate(self) -> dict:
        """
        Evaluate model on validation set.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.val_dataset is None:
            raise ValueError("Validation dataset not available")
        
        self.model.eval()
        
        total_loss = 0.0
        total_perplexity = 0.0
        num_batches = 0
        all_losses = []
        
        # Calculate number of batches
        if self.config.num_batches is None:
            num_batches_to_eval = max(1, len(self.val_dataset) // (
                self.config.batch_size * self.config.context_length
            ))
        else:
            num_batches_to_eval = self.config.num_batches
        
        print(f"\nEvaluating on {num_batches_to_eval} batches...")
        print("-" * 80)
        
        for batch_idx in range(num_batches_to_eval):
            # Get batch
            x, y = get_batch(
                self.val_dataset,
                self.config.batch_size,
                self.config.context_length,
                str(self.device),
            )
            
            # Forward pass
            logits = self.model(x)  # (batch_size, seq_len, vocab_size)
            
            # Compute loss
            loss = self.loss_fn(logits, y)
            
            # Compute perplexity
            perplexity = self.perplexity_fn(logits, y)
            
            total_loss += loss.item()
            total_perplexity += perplexity.item()
            all_losses.append(loss.item())
            num_batches += 1
            
            # Print progress
            if (batch_idx + 1) % max(1, num_batches_to_eval // 10) == 0 or batch_idx == 0:
                avg_loss = total_loss / num_batches
                avg_perplexity = total_perplexity / num_batches
                print(f"Batch {batch_idx + 1}/{num_batches_to_eval} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Perplexity: {perplexity.item():.4f} | "
                      f"Avg Loss: {avg_loss:.4f} | "
                      f"Avg Perplexity: {avg_perplexity:.4f}")
        
        # Compute final metrics
        avg_loss = total_loss / num_batches
        avg_perplexity = total_perplexity / num_batches
        
        # Compute additional statistics
        loss_std = np.std(all_losses)
        loss_min = np.min(all_losses)
        loss_max = np.max(all_losses)
        
        metrics = {
            'avg_loss': avg_loss,
            'avg_perplexity': avg_perplexity,
            'loss_std': loss_std,
            'loss_min': loss_min,
            'loss_max': loss_max,
            'num_batches': num_batches,
        }
        
        return metrics
    
    def print_metrics(self, metrics: dict):
        """
        Print evaluation metrics in a formatted way.
        
        Args:
            metrics: Dictionary of metrics from evaluate()
        """
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"Model Config:")
        print(f"  Vocab Size: {self.config.vocab_size}")
        print(f"  Context Length: {self.config.context_length}")
        print(f"  D Model: {self.config.d_model}")
        print(f"  Num Layers: {self.config.num_layers}")
        print(f"  Num Heads: {self.config.num_heads}")
        print(f"  D FF: {self.config.d_ff}")
        print(f"\nMetrics:")
        print(f"  Average Loss: {metrics['avg_loss']:.4f}")
        print(f"  Average Perplexity: {metrics['avg_perplexity']:.4f}")
        print(f"  Loss Std Dev: {metrics['loss_std']:.4f}")
        print(f"  Loss Min: {metrics['loss_min']:.4f}")
        print(f"  Loss Max: {metrics['loss_max']:.4f}")
        print(f"  Num Batches: {metrics['num_batches']}")
        print("=" * 80)


def main():
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate Transformer Language Model')
    
    # Checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    
    # Model architecture (usually from checkpoint)
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
    
    # Evaluation settings
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--num-batches', type=int, default=None,
                        help='Number of batches to evaluate (None = all)')
    
    # Device and precision
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda/cpu)')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='Data type (float32/float16)')
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON config file (can be training config; overrides CLI args)')
    
    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save evaluation results as JSON')
    
    args = parser.parse_args()
    
    if args.config:
        config = EvalConfig.from_json(args.config)
    else:
        config = EvalConfig(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            device=args.device,
            dtype=args.dtype,
        )
    
    evaluator = Evaluator(config, args.checkpoint)
    
    # Run evaluation
    metrics = evaluator.evaluate()
    evaluator.print_metrics(metrics)
    
    # Save results if requested
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_metrics = {k: float(v) if isinstance(v, (np.floating, torch.Tensor)) else v 
                           for k, v in metrics.items()}
            json.dump(json_metrics, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
