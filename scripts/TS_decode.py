"""
Decoding script for the Transformer Language Model.

Features:
- Generate completions for a user-provided prompt
- Control maximum number of generated tokens
- Temperature scaling for controlling output diversity
- Top-p (nucleus) sampling for constrained generation
- Checkpoint loading and inference
"""

import os
import argparse
import json
import pickle
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch

from basicformer.model.transformer import TransformerLM
from basicformer.utils.Softmax import Softmax


class TextDecoder:
    """
    Handles text generation from a trained TransformerLM model.
    Supports temperature scaling and top-p sampling.
    """
    
    def __init__(
        self,
        model: TransformerLM,
        tokenizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the TextDecoder.
        
        Args:
            model: Trained TransformerLM instance
            tokenizer: BPE tokenizer with encode/decode methods
            device: Device to run inference on
            dtype: Data type for computations
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.dtype = dtype or torch.float32
        self.softmax = Softmax()
        self.model.eval()
    
    @staticmethod
    def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Apply temperature scaling to logits to control output diversity.
        
        Args:
            logits: Raw model logits [vocab_size]
            temperature: Scaling factor (>1 = more diverse, <1 = more deterministic)
                        Values typically in [0.1, 2.0]
        
        Returns:
            Scaled logits
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        return logits / temperature
    
    @staticmethod
    def top_p_sampling(
        logits: torch.Tensor,
        top_p: float = 0.9,
        min_tokens_to_keep: int = 1,
    ) -> torch.Tensor:
        """
        Apply top-p (nucleus) sampling to logits.
        
        Keeps the smallest set of tokens whose cumulative probability exceeds top_p,
        and sets all other logits to -inf (zero probability after softmax).
        
        Args:
            logits: Raw model logits [vocab_size]
            top_p: Cumulative probability threshold (typically 0.9-0.99)
            min_tokens_to_keep: Minimum number of tokens to keep
        
        Returns:
            Filtered logits with low-probability tokens set to -inf
        """
        if top_p <= 0 or top_p > 1:
            raise ValueError("top_p must be in (0, 1]")
        
        # Sort logits and get sorted indices
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        
        # Compute cumulative probabilities using manual softmax
        max_logits = sorted_logits.max()
        exp_logits = torch.exp(sorted_logits - max_logits)
        sorted_probs = exp_logits / exp_logits.sum()
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Determine which tokens to keep: those whose cumulative probability is within top_p
        sorted_indices_to_keep = cumsum_probs <= top_p
        
        # Ensure we keep at least min_tokens_to_keep
        sorted_indices_to_keep[: min_tokens_to_keep] = True
        
        # Create a mask for the original logit positions
        indices_to_keep = sorted_indices[sorted_indices_to_keep]
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[indices_to_keep] = True
        
        # Set non-kept logits to -inf
        filtered_logits = logits.clone()
        filtered_logits[~mask] = -float('inf')
        
        return filtered_logits
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        endoftext_token: str = "<|endoftext|>",
        verbose: bool = False,
    ) -> str:
        """
        Generate text completion for a given prompt.
        
        Args:
            prompt: Starting text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature scaling factor (1.0 = no scaling)
            top_p: Nucleus sampling threshold. If None, uses standard sampling.
            endoftext_token: End-of-text token as a string (e.g., "<|endoftext|>")
            verbose: If True, print generation progress
        
        Returns:
            Generated text (prompt + completion, excluding end-of-text token)
        
        Raises:
            ValueError: If temperature, top_p values are invalid, or if endoftext_token not found
            RuntimeError: If tokenizer encoding/decoding fails
        """
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_p is not None and (top_p <= 0 or top_p > 1):
            raise ValueError("top_p must be in (0, 1]")
        
        # Look up the token ID for the end-of-text token
        try:
            endoftext_token_bytes = endoftext_token.encode('utf-8')
            endoftext_token_id = self.tokenizer.vocab.get_key(endoftext_token_bytes)
            if endoftext_token_id is None:
                raise ValueError(
                    f"End-of-text token '{endoftext_token}' not found in tokenizer vocab. "
                    f"Available special tokens: {self.tokenizer.special_tokens}"
                )
        except Exception as e:
            raise ValueError(
                f"Error looking up end-of-text token '{endoftext_token}': {e}. "
                f"Available special tokens: {self.tokenizer.special_tokens}"
            )
        
        # Encode prompt to token IDs
        try:
            prompt_token_ids = self.tokenizer.encode(prompt)
        except Exception as e:
            raise RuntimeError(f"Failed to encode prompt: {e}")
        
        # Convert to tensor
        tokens = torch.tensor(prompt_token_ids, dtype=torch.long, device=self.device)
        
        # Ensure prompt doesn't exceed context length
        context_length = self.model.context_length
        if len(tokens) > context_length:
            tokens = tokens[-context_length:]
            if verbose:
                print(f"Warning: Prompt truncated to {context_length} tokens (model context length)")
        
        # Generation loop
        for step in range(max_new_tokens):
            # Prepare input: take last context_length tokens
            input_tokens = tokens[-context_length:].unsqueeze(0)  # [1, seq_len]
            
            # Forward pass
            logits = self.model(input_tokens)  # [1, seq_len, vocab_size]
            
            # Get logits for the last position
            next_token_logits = logits[0, -1, :]  # [vocab_size]
            
            # Apply temperature scaling
            next_token_logits = self.apply_temperature(next_token_logits, temperature)
            
            # Apply top-p sampling if specified
            if top_p is not None:
                next_token_logits = self.top_p_sampling(next_token_logits, top_p)
            
            # Sample from the distribution using Softmax
            probs = self.softmax(next_token_logits.unsqueeze(0), dim=-1).squeeze(0)
            next_token = torch.multinomial(probs, num_samples=1, replacement=True)
            
            # Append to sequence
            tokens = torch.cat([tokens, next_token])
            
            if verbose:
                print(f"Step {step + 1}/{max_new_tokens}: sampled token {next_token.item()}")
            
            # Check for end-of-text token and stop immediately
            if next_token.item() == endoftext_token_id:
                if verbose:
                    print(f"End-of-text token reached at step {step + 1}")
                # Remove the end-of-text token before returning
                tokens = tokens[:-1]
                break
        
        # Decode all tokens back to text
        try:
            generated_text = self.tokenizer.decode(tokens.cpu().numpy().tolist())
        except Exception as e:
            raise RuntimeError(f"Failed to decode generated tokens: {e}")
        
        return generated_text
    
    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        endoftext_token: str = "<|endoftext|>",
    ) -> List[str]:
        """
        Generate text completions for multiple prompts.
        
        Args:
            prompts: List of starting text prompts
            max_new_tokens: Maximum number of tokens to generate per prompt
            temperature: Temperature scaling factor
            top_p: Nucleus sampling threshold
            endoftext_token: End-of-text token as a string
        
        Returns:
            List of generated texts (one per prompt)
        """
        return [
            self.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                endoftext_token=endoftext_token,
            )
            for prompt in prompts
        ]


def load_checkpoint(checkpoint_path: str, device: str) -> Tuple[TransformerLM, dict]:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
    
    Returns:
        Tuple of (model, config_dict)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct model from config
    config = checkpoint['config']
    
    # Handle dtype: it may be stored as a string or as a torch.dtype object
    dtype = config['dtype']
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    
    model = TransformerLM(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        device=torch.device(device),
        dtype=dtype,
    )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, config


def load_tokenizer(tokenizer_path: str):
    """
    Load tokenizer from pickle file.
    
    Args:
        tokenizer_path: Path to tokenizer pickle file
    
    Returns:
        Loaded tokenizer instance
    """
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    return tokenizer


def main():
    """Parse arguments and run text generation."""
    parser = argparse.ArgumentParser(
        description='Generate text completions with a trained Transformer Language Model'
    )
    
    # Model and tokenizer paths
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint file'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        required=True,
        help='Path to tokenizer pickle file'
    )
    
    # Generation parameters
    parser.add_argument(
        '--prompt',
        type=str,
        default='Once upon a time',
        help='Text prompt to generate completion for'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=100,
        help='Maximum number of tokens to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Temperature for softmax scaling (>1 = more diverse, <1 = more deterministic)'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=None,
        help='Nucleus sampling threshold (e.g., 0.9). If not set, uses standard sampling.'
    )
    parser.add_argument(
        '--endoftext-token',
        type=str,
        default='<|endoftext|>',
        help='End-of-text token as a string (e.g., "<|endoftext|>")'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run inference on (cuda/cpu)'
    )
    
    # Verbose
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print generation progress'
    )
    
    args = parser.parse_args()
    
    # Validate parameters
    if args.temperature <= 0:
        raise ValueError("Temperature must be positive")
    if args.top_p is not None and (args.top_p <= 0 or args.top_p > 1):
        raise ValueError("top_p must be in (0, 1]")
    if args.max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    
    print("=" * 80)
    print("Transformer Language Model - Text Generation")
    print("=" * 80)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    model, config = load_checkpoint(args.checkpoint, args.device)
    print(f"✓ Model loaded: {config['vocab_size']} vocab, {config['context_length']} context")
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = load_tokenizer(args.tokenizer)
    print(f"✓ Tokenizer loaded")
    
    # Initialize decoder
    decoder = TextDecoder(model, tokenizer, device=args.device)
    
    # Generate
    print(f"\n{'=' * 80}")
    print("Generation Parameters:")
    print(f"{'=' * 80}")
    print(f"Prompt:           {args.prompt}")
    print(f"Max tokens:       {args.max_tokens}")
    print(f"Temperature:      {args.temperature}")
    print(f"Top-p:            {args.top_p if args.top_p else 'disabled (standard sampling)'}")
    print(f"Device:           {args.device}")
    print(f"{'=' * 80}\n")
    
    try:
        generated_text = decoder.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            endoftext_token=args.endoftext_token,
            verbose=args.verbose,
        )
        
        print(f"{'=' * 80}")
        print("Generated Text:")
        print(f"{'=' * 80}")
        print(generated_text)
        print(f"{'=' * 80}")
        
    except Exception as e:
        print(f"Error during generation: {e}")
        raise


if __name__ == '__main__':
    main()
