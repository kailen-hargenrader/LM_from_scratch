# Training Guide

This guide explains how to use the `train_lm.py` script to train the Transformer Language Model.

## Features

The training script includes:

- **Configurable Hyperparameters**: Control model architecture and optimization settings via CLI arguments or JSON config files
- **Memory-Efficient Data Loading**: Uses `np.memmap` for loading large datasets without loading entire arrays into RAM
- **Checkpoint Management**: Automatic checkpoint saving at regular intervals, with support for resuming training
- **Comprehensive Logging**: Console and optional Weights & Biases integration for tracking experiments
- **Gradient Clipping**: Prevents training instability with max norm clipping

## Quick Start

### Basic Training

Train with default settings:

```bash
uv run scripts/train_lm.py
```

### With Custom Hyperparameters

```bash
uv run scripts/train_lm.py \
  --num-epochs 20 \
  --batch-size 128 \
  --lr 1e-4 \
  --d-model 512 \
  --num-layers 6 \
  --num-heads 8
```

## Configuration

### Via Command-Line Arguments

All hyperparameters can be specified via CLI flags:

```bash
uv run scripts/train_lm.py \
  --vocab-size 256 \
  --context-length 128 \
  --d-model 256 \
  --num-layers 4 \
  --num-heads 8 \
  --d-ff 1024 \
  --lr 5e-4 \
  --batch-size 64 \
  --num-epochs 10 \
  --weight-decay 0.01 \
  --train-data data/TinyStoriesV2-GPT4-train-ids.npy \
  --val-data data/TinyStoriesV2-GPT4-valid-ids.npy \
  --checkpoint-dir checkpoints \
  --log-interval 100 \
  --val-interval 500 \
  --checkpoint-interval 1000
```

### Via JSON Config File

Create a JSON config file (e.g., `config.json`):

```json
{
  "vocab_size": 256,
  "context_length": 128,
  "d_model": 256,
  "num_layers": 4,
  "num_heads": 8,
  "d_ff": 1024,
  "learning_rate": 5e-4,
  "batch_size": 64,
  "num_epochs": 10,
  "weight_decay": 0.01,
  "train_data_path": "data/TinyStoriesV2-GPT4-train-ids.npy",
  "val_data_path": "data/TinyStoriesV2-GPT4-valid-ids.npy",
  "checkpoint_dir": "checkpoints",
  "log_interval": 100,
  "val_interval": 500,
  "checkpoint_interval": 1000,
  "device": "cuda",
  "dtype": "float32",
  "use_wandb": false
}
```

Then run:

```bash
uv run scripts/train_lm.py --config config.json
```

## Hyperparameter Reference

### Model Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--vocab-size` | 256 | Size of token vocabulary |
| `--context-length` | 128 | Maximum sequence length (context window) |
| `--d-model` | 256 | Embedding and hidden dimension |
| `--num-layers` | 4 | Number of Transformer blocks |
| `--num-heads` | 8 | Number of attention heads |
| `--d-ff` | 1024 | Feedforward hidden dimension |

### Optimization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lr` | 5e-4 | Learning rate (Adam optimizer) |
| `--batch-size` | 64 | Training batch size |
| `--num-epochs` | 10 | Number of training epochs |
| `--weight-decay` | 0.01 | L2 regularization coefficient |

### Data

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--train-data` | `data/TinyStoriesV2-GPT4-train-ids.npy` | Path to training data (numpy array) |
| `--val-data` | `data/TinyStoriesV2-GPT4-valid-ids.npy` | Path to validation data (numpy array) |

### Checkpointing & Logging

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint-dir` | `checkpoints` | Directory to save model checkpoints |
| `--log-interval` | 100 | Print loss every N steps |
| `--val-interval` | 500 | Run validation every N steps |
| `--checkpoint-interval` | 1000 | Save checkpoint every N steps |

### Device & Precision

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--device` | `cuda` (if available) | `cuda` or `cpu` |
| `--dtype` | `float32` | `float32` or `float16` (mixed precision) |

### Weights & Biases Logging

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use-wandb` | False | Enable W&B experiment tracking |
| `--wandb-project` | `transformer-lm` | W&B project name |
| `--wandb-entity` | None | W&B entity/team name |
| `--run-name` | Auto-generated | Custom run name for W&B |

## Checkpointing

### Automatic Checkpointing

Checkpoints are automatically saved every `--checkpoint-interval` steps:

```bash
uv run scripts/train_lm.py --checkpoint-interval 500
```

Checkpoints include:
- Model weights (`state_dict`)
- Optimizer state
- Training step and epoch counters
- Configuration used for training

Checkpoints are saved as `checkpoint-step-{N}.pt` in the checkpoint directory.

### Manual Checkpoint Saving

The trainer automatically saves "best" checkpoints. You can manually save via:
```bash
trainer.save_checkpoint(name="model-final.pt")
```

### Resuming Training

Resume training from a checkpoint:

```bash
uv run scripts/train_lm.py --resume checkpoints/checkpoint-step-5000.pt
```

This will:
- Load the model weights
- Restore optimizer state
- Continue from the same step/epoch
- Use the same configuration

## Logging

### Console Logging

Training progress is logged to console every `--log-interval` steps:

```
Epoch 1/10 | Step 0/1563 | Loss: 8.5234 (avg: 8.5234)
Epoch 1/10 | Step 100/1563 | Loss: 7.2145 (avg: 7.3189)
Epoch 1/10 | Step 200/1563 | Loss: 6.8901 (avg: 7.1667)
...
Checkpoint saved: checkpoints/checkpoint-step-1000.pt
Validation Loss: 6.5432
```

### Weights & Biases Integration

Enable W&B logging for experiment tracking, visualization, and comparison:

```bash
uv pip install wandb
uv run scripts/train_lm.py --use-wandb --wandb-project transformer-lm
```

To save a W&B API key (to reduce logins). Store the key in an environment variable

```bash
export WANDB_API_KEY='<API-Key>'
```

Logged metrics:
- `train/loss`: Loss at each step
- `train/loss_avg`: Moving average loss
- `val/loss`: Validation loss
- Hyperparameters and config
- System metrics (GPU usage, etc.)

## Data Format

Training and validation data should be numpy arrays (`.npy` format) containing token IDs:

- Shape: `(seq_len,)` - 1D array of integers
- Values: Token IDs in range `[0, vocab_size)`
- Format: Saved via `np.save(path, token_ids)`

Example of preparing data:

```python
import numpy as np

# Your tokenized text
token_ids = np.array([1, 5, 23, 45, 12, ..., 89, 102])

# Save for training
np.save('data/train-ids.npy', token_ids)
```

## Example Configurations

### Small Model (Fast Experimentation)

```bash
uv run scripts/train_lm.py \
  --d-model 128 \
  --num-layers 2 \
  --num-heads 4 \
  --d-ff 512 \
  --batch-size 32 \
  --lr 1e-3
```

### Medium Model (Balanced)

```bash
uv run scripts/train_lm.py \
  --d-model 512 \
  --num-layers 6 \
  --num-heads 8 \
  --d-ff 2048 \
  --batch-size 128 \
  --lr 5e-4
```

### Large Model (GPU-Intensive)

```bash
uv run scripts/train_lm.py \
  --d-model 1024 \
  --num-layers 12 \
  --num-heads 16 \
  --d-ff 4096 \
  --batch-size 256 \
  --lr 2e-4 \
  --dtype float16
```

### Research Setup (With W&B)

```bash
uv run scripts/train_lm.py \
  --num-epochs 50 \
  --d-model 512 \
  --num-layers 8 \
  --batch-size 256 \
  --checkpoint-interval 5000 \
  --log-interval 50 \
  --val-interval 1000 \
  --use-wandb \
  --wandb-project my-experiments \
  --run-name baseline-v1
```

## Troubleshooting

### Out of Memory (OOM)

- Reduce `--batch-size`
- Reduce `--d-model` or `--num-layers`
- Use `--dtype float16` for reduced precision
- Ensure datasets are using memmap (should be automatic)

### Slow Training

- Verify GPU is being used: `--device cuda`
- Reduce `--log-interval` to avoid excess logging overhead
- Check that data loading isn't the bottleneck

### NaN Loss

- Reduce learning rate: `--lr 1e-5`
- Enable gradient clipping (enabled by default)
- Check for numerical issues in data (NaN or very large values)

### Checkpoint Issues

- Ensure `--checkpoint-dir` exists and is writable
- Verify checkpoint path exists when resuming: `--resume path/to/checkpoint.pt`
- Check disk space for large models

## API Usage (Programmatic)

```python
from scripts.train_lm import TrainingConfig, Trainer

# Create config
config = TrainingConfig(
    vocab_size=256,
    d_model=512,
    num_layers=6,
    batch_size=128,
    num_epochs=20,
    learning_rate=5e-4,
)

# Create trainer
trainer = Trainer(config)

# Train
trainer.train()

# Or manually step through training
for epoch in range(config.num_epochs):
    trainer.train_epoch()
    if trainer.val_dataset:
        val_loss = trainer.validate()
    trainer.save_checkpoint(name=f"epoch-{epoch}.pt")
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
- [PyTorch Training Guide](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
