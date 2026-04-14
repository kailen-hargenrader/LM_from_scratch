import os
import pickle
from basicformer.tokenizer.bpe import BPETokenizer

VOCAB_SIZE = 10000
ARTIFACTS_DIR = os.path.join('artifacts', 'tokenizers')
TOKENIZER_PICKLE_PATH = os.path.join(ARTIFACTS_DIR, f'TS_bpe_tokenizer_{VOCAB_SIZE}.pkl')

def load_tokenizer(pickle_path=TOKENIZER_PICKLE_PATH):
    with open(pickle_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

if __name__ == "__main__":
    tokenizer = load_tokenizer()
    print("Tokenizer vocab:", tokenizer.vocab)
    print("Tokenizer merges:", tokenizer.merges)
    print("Loaded BPE Tokenizer with vocab size:", len(tokenizer.vocab))