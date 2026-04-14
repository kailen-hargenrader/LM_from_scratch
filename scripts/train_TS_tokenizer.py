import os
import pickle
from tqdm import tqdm
from basicformer.tokenizer.bpe import BPETokenizer, make_standard_vocab
from basicformer.utils.BiDict import BiDict

VOCAB_SIZE = 10000

# Paths
DATA_DIR = 'data'
DATA_FILE = os.path.join(DATA_DIR, 'TinyStoriesV2-GPT4-train.txt')
ARTIFACTS_DIR = os.path.join('artifacts', 'tokenizers')
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
TOKENIZER_PICKLE_PATH = os.path.join(ARTIFACTS_DIR, f'TS_bpe_tokenizer_{VOCAB_SIZE}.pkl')


def train_bpe_tokenizer(input_file, vocab_size, special_tokens=["<|endoftext|>"]):
    # Initialize with standard byte vocabulary (0-255)
    vocab = make_standard_vocab()
    
    # Add special tokens to vocab
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode('utf-8')
    
    # Initialize empty merges and create tokenizer
    merges = {}
    tokenizer = BPETokenizer(vocab, merges, special_tokens)
    
    # Read training file
    with open(input_file, 'r', encoding='utf-8') as f:
        corpus = f.read()
    
    print(f"Training BPE on corpus of {len(corpus)} characters...")
    
    # Create progress bar to track vocab growth
    pbar = tqdm(initial=len(tokenizer.vocab), total=vocab_size, desc="Vocab size", unit=" tokens")
    
    # Call train with progress bar
    tokenizer.train(corpus, vocab_size, special_tokens, pbar=pbar)
    pbar.close()
    
    return tokenizer

def main():
    if not os.path.exists(DATA_FILE):
        print(f"Training file {DATA_FILE} does not exist.")
        return

    print(f"Training Byte-Pair Encoding tokenizer on {DATA_FILE} ...")
    tokenizer = train_bpe_tokenizer(DATA_FILE, VOCAB_SIZE)
    
    print(f"Saving tokenizer as pickle to {TOKENIZER_PICKLE_PATH} ...")
    with open(TOKENIZER_PICKLE_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    print(f"Done! Tokenizer vocab size: {len(tokenizer.vocab)}")
    print(f"Number of merges learned: {len(tokenizer.merges)}")

if __name__ == "__main__":
    main()