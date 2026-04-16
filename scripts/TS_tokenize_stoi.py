import os
import pickle
import numpy as np
from tqdm import tqdm
from basicformer.tokenizer.bpe import BPETokenizer

VOCAB_SIZE = 10000

# Paths
DATA_DIR = 'data'
INPUT_FILE = os.path.join(DATA_DIR, 'TinyStoriesV2-GPT4-valid.txt')
ARTIFACTS_DIR = os.path.join('artifacts', 'tokenizers')
TOKENIZER_PICKLE_PATH = os.path.join(ARTIFACTS_DIR, f'TS_bpe_tokenizer_{VOCAB_SIZE}.pkl')
OUTPUT_PATH = os.path.join(DATA_DIR, 'TinyStoriesV2-GPT4-valid-ids.npy')


def load_tokenizer(pickle_path=TOKENIZER_PICKLE_PATH):
    """Load the pre-trained BPE tokenizer from pickle file."""
    with open(pickle_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer


def tokenize_file(input_file, tokenizer, chunk_size=5_000_000):
    """
    Tokenize a file in chunks to handle large files efficiently.
    Yields batches of token IDs for efficient processing.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            token_ids = tokenizer.encode(chunk)
            yield token_ids


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} does not exist.")
        return
    
    if not os.path.exists(TOKENIZER_PICKLE_PATH):
        print(f"Tokenizer file {TOKENIZER_PICKLE_PATH} does not exist.")
        return
    
    print(f"Loading tokenizer from {TOKENIZER_PICKLE_PATH}...")
    tokenizer = load_tokenizer()
    print(f"✓ Tokenizer loaded with vocab size: {len(tokenizer.vocab)}")
    
    print(f"\nTokenizing {INPUT_FILE}...")
    
    # Tokenize file and collect all token ID batches
    all_token_batches = []
    for token_batch in tqdm(tokenize_file(INPUT_FILE, tokenizer), desc="Tokenizing chunks"):
        all_token_batches.append(token_batch)
    
    # Concatenate all batches into a single array
    token_array = np.concatenate([np.array(batch, dtype=np.uint16) for batch in all_token_batches])
    print(f"✓ Tokenization complete: {len(token_array):,} tokens")
    
    # Save to disk
    print(f"\nSaving token IDs to {OUTPUT_PATH}...")
    np.save(OUTPUT_PATH, token_array)
    print(f"✓ Saved successfully!")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"{'='*60}")
    print(f"Input file:              {INPUT_FILE}")
    print(f"Tokenizer:               {TOKENIZER_PICKLE_PATH}")
    print(f"Output file:             {OUTPUT_PATH}")
    print(f"Total tokens:            {len(token_array):,}")
    print(f"Array dtype:             {token_array.dtype}")
    print(f"Array shape:             {token_array.shape}")
    print(f"Memory usage:            {token_array.nbytes / (1024**3):.2f} GB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
