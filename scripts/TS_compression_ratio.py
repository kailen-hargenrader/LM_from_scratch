import os
import pickle
import random
from basicformer.tokenizer.bpe import BPETokenizer

VOCAB_SIZE = 10000
ARTIFACTS_DIR = os.path.join('artifacts', 'tokenizers')
TOKENIZER_PICKLE_PATH = os.path.join(ARTIFACTS_DIR, f'TS_bpe_tokenizer_{VOCAB_SIZE}.pkl')
DATA_DIR = os.path.join('data')
TRAIN_FILE = os.path.join(DATA_DIR, 'TinyStoriesV2-GPT4-train.txt')


def load_tokenizer(pickle_path=TOKENIZER_PICKLE_PATH):
    """Load the pre-trained BPE tokenizer from pickle file."""
    with open(pickle_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer


def load_and_sample_documents(file_path, num_samples=10):
    """
    Load documents from TinyStories file and sample num_samples documents.
    Documents are separated by '<|endoftext|>' token.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by endoftext token to get individual documents
    documents = content.split('<|endoftext|>')
    documents = [doc.strip() for doc in documents if doc.strip()]
    
    # Sample random documents
    sampled_docs = random.sample(documents, min(num_samples, len(documents)))
    return sampled_docs


def calculate_compression_ratio(tokenizer, documents):
    """
    Calculate compression ratio: total_bytes / total_tokens
    """
    total_bytes = 0
    total_tokens = 0
    
    for doc in documents:
        # Count original bytes in UTF-8 encoding
        doc_bytes = len(doc.encode('utf-8'))
        total_bytes += doc_bytes
        
        # Encode document to token IDs and count tokens
        token_ids = tokenizer.encode(doc)
        total_tokens += len(token_ids)
    
    compression_ratio = total_bytes / total_tokens if total_tokens > 0 else 0
    return compression_ratio, total_bytes, total_tokens


def find_longest_token(tokenizer):
    """
    Find the longest token in the vocabulary.
    Returns the token bytes and its length.
    """
    longest_token = None
    longest_length = 0
    
    for token_id, token_bytes in tokenizer.vocab.items():
        token_length = len(token_bytes)
        if token_length > longest_length:
            longest_length = token_length
            longest_token = token_bytes
    
    return longest_token, longest_length


if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    
    print("Loading tokenizer...")
    tokenizer = load_tokenizer()
    print(f"✓ Tokenizer loaded with vocab size: {len(tokenizer.vocab)}")
    
    # Find and display the longest token
    longest_token, longest_length = find_longest_token(tokenizer)
    try:
        longest_token_str = longest_token.decode('utf-8', errors='replace')
    except:
        longest_token_str = repr(longest_token)
    print(f"✓ Longest token: {longest_token_str!r} ({longest_length} bytes)")
    
    print(f"\nLoading and sampling 10 documents from {TRAIN_FILE}...")
    documents = load_and_sample_documents(TRAIN_FILE, num_samples=10)
    print(f"✓ Sampled {len(documents)} documents")
    
    print(f"\nEncoding documents...")
    compression_ratio, total_bytes, total_tokens = calculate_compression_ratio(tokenizer, documents)
    
    print(f"\n{'='*60}")
    print(f"Compression Ratio Results:")
    print(f"{'='*60}")
    print(f"Total bytes (UTF-8):     {total_bytes:,}")
    print(f"Total tokens:            {total_tokens:,}")
    print(f"Compression ratio:       {compression_ratio:.4f} bytes/token")
    print(f"{'='*60}")
