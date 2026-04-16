import os
import numpy as np

DATA_DIR = 'data'
INPUT_FILE = os.path.join(DATA_DIR, 'TinyStoriesV2-GPT4-valid.txt')
OUTPUT_PATH = os.path.join(DATA_DIR, 'TinyStoriesV2-GPT4-valid-ids.npy')


def main():
    if not os.path.exists(OUTPUT_PATH):
        print(f"Token indices file {OUTPUT_PATH} does not exist.")
        return
    
    # Load the token indices
    print(f"Loading token indices from {OUTPUT_PATH}...")
    token_array = np.load(OUTPUT_PATH)
    print(f"✓ Loaded successfully!")
    
    # Count indices
    num_tokens = len(token_array)
    print(f"\nNumber of tokens: {num_tokens:,}")
    
    # Get original file size
    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} does not exist.")
        return
    
    file_size_bytes = os.path.getsize(INPUT_FILE)
    print(f"Original file size: {file_size_bytes:,} bytes")
    
    # Calculate compression ratio
    compression_ratio = num_tokens / file_size_bytes
    print(f"\n{'='*60}")
    print(f"Comparison:")
    print(f"{'='*60}")
    print(f"Tokens: {num_tokens:,}")
    print(f"Bytes:  {file_size_bytes:,}")
    print(f"Ratio (tokens/bytes): {compression_ratio:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
