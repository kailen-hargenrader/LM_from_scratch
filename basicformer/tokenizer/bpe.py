# Core Tokenizer class
from .base import BaseTokenizer, Vocab
from basicformer.utils.DoublyLinkedList import DoublyLinkedList, DoublyLinkedListNode, DoublyLinkedListNodePair
from typing import TypeAlias
from heapq import heappush, heappop
from collections import Counter, defaultdict
import regex

Merges: TypeAlias = dict[(bytes, bytes), int]
PAT = r"""’(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def make_standard_vocab() -> Vocab:
    vocab = Vocab()
    for i in range(256):
        vocab[i] = bytes([i])
    return vocab

class BPETokenizer(BaseTokenizer):
    def __init__(self, initial_vocab: Vocab, initial_merges: Merges, special_tokens: list[str], unknown_token: bytes = b"<unk>"):

        self.vocab: Vocab = initial_vocab
        self.merges: Merges = initial_merges
        self.special_tokens: list[str] = list[str](set[str](special_tokens))
        self.unknown_token: bytes = unknown_token
    
    def _merge(self, bytes_list: DoublyLinkedList[bytes]) -> list[bytes]:
        # make a min heap of the mergable pairs of bytes in the linked list
        min_heap: list[DoublyLinkedListNodePair[bytes]] = []

        # Start from the beginning of the list
        curr_node = bytes_list.first
        while curr_node and curr_node.next:
            next_node = curr_node.next
            priority = self.merges.get((curr_node.data, next_node.data))
            if priority is not None:
                heappush(min_heap, DoublyLinkedListNodePair(curr_node, next_node, priority))
            curr_node = next_node

        while len(min_heap) > 0:
            node_pair: DoublyLinkedListNodePair[bytes] = heappop(min_heap)

            # ignore stale pairs
            if node_pair.node1.next is not node_pair.node2 or node_pair.node2.prev is not node_pair.node1:
                continue

            # merge the nodes
            new_bytes = node_pair.node1.data + node_pair.node2.data
            new_node = DoublyLinkedListNode(new_bytes)
            if node_pair.node1.prev is not None:
                node_pair.node1.prev.next = new_node
            else:
                # Update first node reference in bytes_list if new_node is now first
                bytes_list.first = new_node
            if node_pair.node2.next is not None:
                node_pair.node2.next.prev = new_node
            new_node.prev = node_pair.node1.prev
            new_node.next = node_pair.node2.next

            # add merge with previous node to the min heap
            if new_node.prev is not None:
                priority = self.merges.get((new_node.prev.data, new_node.data))
                if priority is not None:
                    heappush(min_heap, DoublyLinkedListNodePair(new_node.prev, new_node, priority))

            # add merge with next node to the min heap
            if new_node.next is not None:
                priority = self.merges.get((new_node.data, new_node.next.data))
                if priority is not None:
                    heappush(min_heap, DoublyLinkedListNodePair(new_node, new_node.next, priority))
        return bytes_list.to_list()

    def _separate_special_tokens(self, text: str) -> list[str]:
        # turn str into list[str] split by special tokens
        if len(self.special_tokens) == 0:
            return [text]
        pattern = fr"({'|'.join(map(regex.escape, self.special_tokens))})"
        text_list = regex.split(pattern, text) 
        #Note if text starts with a special token, the first element of text_list will be an empty string
        #Thus, the special tokens occur on odd indices of text_list
        return text_list
    
    def _pretokenize(self, text: str) -> list[str]:
        # turn str into list[str] split by PAT
        text_list = regex.finditer(PAT, text)
        return [text.group() for text in text_list]

    def encode(self, text: str) -> list[int]:
        text_list = self._separate_special_tokens(text)

        # for each text:
        #   1. turn str into list[bytes]
        #   2. Merge list[bytes] with self.merges
        #   3. convert list[bytes] into list[int]
        full_bytes_list: list[bytes] = []
        for i in range(len(text_list)):
            
            if i % 2 == 0:
                #not a special token, pretokenize text and merge
                text_list2 = self._pretokenize(text_list[i])
                for text2 in text_list2:
                    bytes_list = [bytes([b]) for b in text2.encode("utf-8")]
                    bytes_linked_list = DoublyLinkedList(bytes_list)
                    full_bytes_list += self._merge(bytes_linked_list)
            else:
                #special token, add to full_bytes_list
                assert text_list[i] in self.special_tokens, f"Special token {text_list[i]} not found in special tokens list"
                full_bytes_list += [text_list[i].encode("utf-8")]
        full_idx_list: list[int | None] = [self.vocab.get_key(bytes_obj, -1) for bytes_obj in full_bytes_list]
        #full_idx_list is a list of positive ints for recognized bytes, and -1 for unrecognized bytes
        return full_idx_list


    def decode(self, ids: list[int]) -> str:
        #turn a list of ints back into bytes
        bytes_list: list[bytes] = [self.vocab.get(id, self.unknown_token) for id in ids if id is not None]
        #return the bytes as a string
        return b"".join(bytes_list).decode("utf-8")
    
    def get_vocab(self) -> Vocab:
        return self.vocab
    
    def train(self, corpus: str, max_vocab_size: int, special_tokens: list[str], pbar=None) -> None:
        #make sure we have space left in the vocab
        if len(self.vocab) >= max_vocab_size:
            print("Vocab size is already at the maximum size.")
            return
        
        #separate special tokens from the corpus
        self.special_tokens = list[str](set[str](self.special_tokens + special_tokens))
        text_list = self._separate_special_tokens(corpus)

        
        word_freqs: dict[tuple[bytes, ...], int] = defaultdict(int)
        
        for i, text in enumerate(text_list):
            if i % 2 == 0:  # Non-special token
                text_list2 = self._pretokenize(text)
                for text2 in text_list2:
                    bytes_tuple = tuple(bytes([b]) for b in text2.encode("utf-8"))
                    word_freqs[bytes_tuple] += 1
        
        # Global Pair Counts: frequency of each adjacent pair
        pair_counts: dict[tuple[bytes, bytes], int] = Counter()
        
        # Inverted Index: pair -> {word_indices}
        inverted_index: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
        
        # Word list for tracking which words contain which pairs
        words_list: list[list[bytes]] = []
        word_freq_list: list[int] = []
        
        # Initialize pairs in the inverted index
        for word_tuple, freq in word_freqs.items():
            word_idx = len(words_list)
            words_list.append(list(word_tuple))
            word_freq_list.append(freq)
            for j in range(len(word_tuple) - 1):
                pair = (word_tuple[j], word_tuple[j + 1])
                pair_counts[pair] += freq
                inverted_index[pair].add(word_idx)
        
        # Priority Queue (Max-Heap with negative frequencies)
        heap: list[tuple[int, tuple[bytes, bytes]]] = []
        for pair, count in pair_counts.items():
            heappush(heap, (-count, pair))
        
        # === TRAINING LOOP ===
        num_merges = min(max_vocab_size - len(self.vocab), len(pair_counts))
        new_token_id = len(self.vocab)
        
        for merge_step in range(num_merges):
            if not heap:
                break
            
            # Pop Best Pair (with validation for stale entries)
            best_pair = None
            while heap:
                neg_freq, pair = heappop(heap)
                current_freq = pair_counts.get(pair, 0)
                
                # Validation: compare with current count
                if abs(neg_freq) == current_freq and current_freq > 0:
                    best_pair = pair
                    break
            
            if best_pair is None:
                break
            
            # Record Merge
            self.merges[best_pair] = new_token_id
            self.vocab[new_token_id] = best_pair[0] + best_pair[1]
            new_token_id += 1
            
            # Update progress bar if provided
            if pbar is not None:
                pbar.update(1)
            
            # Update Words (Local Update)
            affected_word_indices = inverted_index.get(best_pair, set()).copy()
            
            for word_idx in affected_word_indices:
                word = words_list[word_idx]
                word_freq = word_freq_list[word_idx]
                
                # Find all positions of the pair in this word
                j = 0
                while j < len(word) - 1:
                    if word[j] == best_pair[0] and word[j + 1] == best_pair[1]:
                        # Identify neighboring pairs that will be broken
                        if j > 0:
                            old_pair_left = (word[j - 1], word[j])
                            pair_counts[old_pair_left] -= word_freq
                            inverted_index[old_pair_left].discard(word_idx)
                        
                        if j < len(word) - 2:
                            old_pair_right = (word[j + 1], word[j + 2])
                            pair_counts[old_pair_right] -= word_freq
                            inverted_index[old_pair_right].discard(word_idx)
                        
                        # Perform the Merge
                        merged_token = best_pair[0] + best_pair[1]
                        word[j:j+2] = [merged_token]
                        
                        # Identify new pairs created
                        if j > 0:
                            new_pair_left = (word[j - 1], word[j])
                            pair_counts[new_pair_left] += word_freq
                            inverted_index[new_pair_left].add(word_idx)
                            heappush(heap, (-pair_counts[new_pair_left], new_pair_left))
                        
                        if j < len(word) - 1:
                            new_pair_right = (word[j], word[j + 1])
                            pair_counts[new_pair_right] += word_freq
                            inverted_index[new_pair_right].add(word_idx)
                            heappush(heap, (-pair_counts[new_pair_right], new_pair_right))
                        
                        j += 1
                    else:
                        j += 1
            
            # Refresh Inverted Index
            if best_pair in inverted_index:
                del inverted_index[best_pair]
            
            if best_pair in pair_counts and pair_counts[best_pair] >= 0:
                del pair_counts[best_pair]

# if __name__ == "__main__":
#     print("=" * 70)
#     print("BPE TOKENIZER COMPREHENSIVE EXAMPLE")
#     print("=" * 70)
    
#     from basicformer.utils.BiDict import BiDict
    
#     # Initialize with full 256-byte vocabulary (standard byte-level tokenization)
#     vocab: Vocab = BiDict({i: bytes([i]) for i in range(256)})
#     vocab[256] = b"<|endoftext|>"
    
#     merges: Merges = {}
#     special_tokens = ["<|endoftext|>"]
    
#     # Create tokenizer
#     tokenizer = BPETokenizer(vocab, merges, special_tokens)
#     print(f"\nInitialized with {len(vocab)} byte tokens (0-255) + 1 special token")
#     print(f"Initial merges: {len(merges)}")
    
#     # Example 1: Encoding raw text without merges
#     print("\n" + "-" * 70)
#     print("Example 1: Encoding without merges (byte-level)")
#     print("-" * 70)
#     test_text = "hello world"
#     ids = tokenizer.encode(test_text)
#     tokens = [tokenizer.vocab.get(id, b'?').decode('utf-8', errors='replace') for id in ids]
#     print(f"Input: '{test_text}'")
#     print(f"Tokens: {tokens}")
#     print(f"Token IDs: {ids}")
#     print(f"Tokens needed: {len(ids)} (one per byte)")
    
#     # Example 2: Training BPE on a larger corpus
#     print("\n" + "-" * 70)
#     print("Example 2: Training BPE on a 3-paragraph corpus")
#     print("-" * 70)
    
#     corpus = """
#     The quick brown fox jumps over the lazy dog. Machine learning has revolutionized
#     artificial intelligence. Deep learning models can now process natural language,
#     generate images, and solve complex problems. Transformers have become the dominant
#     architecture for many state-of-the-art systems. Natural language processing uses
#     tokenization to convert text into discrete units. Byte pair encoding is an effective
#     subword tokenization algorithm that learns merges from data.
    
#     Tokenization is a critical preprocessing step in natural language processing. It breaks
#     down text into meaningful units such as words, subwords, or characters. Different tokenization
#     strategies have different trade-offs. Character-level tokenization preserves all information
#     but creates very long sequences. Word-level tokenization is more compact but struggles with
#     out-of-vocabulary words and morphologically rich languages. Subword tokenization like BPE
#     combines the benefits of both approaches by learning an optimal vocabulary from data.
    
#     The BPE algorithm starts with a character-level vocabulary and iteratively merges the most
#     frequent adjacent pairs. Each merge operation creates a new token representing the merged
#     pair. After K merge operations, the vocabulary size grows from 256 bytes to 256 + K tokens.
#     This hierarchical structure allows the tokenizer to learn representations at different levels
#     of granularity. Modern language models like GPT use BPE tokenization with 50,000 tokens or more.
#     The choice of vocabulary size directly impacts model capacity and computational efficiency.
#     """
    
#     max_vocab_size = 400
#     print(f"Training corpus: {len(corpus)} characters")
#     print(f"Target vocab size: {max_vocab_size}")
#     print(f"\nVocab size before training: {len(tokenizer.vocab)}")
#     print(f"Merges before training: {len(tokenizer.merges)}")
    
#     tokenizer.train(corpus, max_vocab_size, [])
    
#     print(f"\nVocab size after training: {len(tokenizer.vocab)}")
#     print(f"Merges learned: {len(tokenizer.merges)}")
#     print(f"Vocabulary growth: +{len(tokenizer.vocab) - 257} new tokens")
    
#     # Example 3: Show some learned merges
#     print("\n--- Sample of learned merges (first 20) ---")
#     for i, (pair, token_id) in enumerate(list(tokenizer.merges.items())[:20]):
#         print(f"  {pair} -> {token_id}: {tokenizer.vocab.get(token_id, b'?')}")
    
#     # Example 4: Encode the entire corpus with trained merges
#     print("\n" + "-" * 70)
#     print("Example 4: Encoding entire corpus with trained merges")
#     print("-" * 70)
    
#     ids = tokenizer.encode(corpus)
#     tokens = [tokenizer.vocab.get(id, b'?').decode('utf-8', errors='replace') for id in ids]
#     decoded = tokenizer.decode(ids)
    
#     print(f"\nCorpus length: {len(corpus)} characters")
#     print(f"Token count: {len(ids)}")
#     print(f"Compression ratio: {len(corpus) / len(ids):.2f}x")
#     print(f"\nToken list (first 100 tokens):")
#     print(tokens[:100])
#     print(f"\n... ({len(tokens) - 100} more tokens) ...")
#     print(f"\nLast 50 tokens:")
#     print(tokens[-50:])
    
#     print(f"\nVerifying decode matches original:")
#     if decoded.strip() == corpus.strip():
#         print("✓ Decode successful - corpus recovered perfectly!")
#     else:
#         print("✗ Decode mismatch")
    
#     # Example 5: Compare compression without merges
#     print("\n" + "-" * 70)
#     print("Example 5: Compression comparison")
#     print("-" * 70)
    
#     tokenizer_no_merges = BPETokenizer(BiDict({i: bytes([i]) for i in range(256)}), {}, [])
    
#     ids_no_merge = tokenizer_no_merges.encode(corpus)
    
#     print(f"\nFull corpus compression analysis:")
#     print(f"Corpus size:              {len(corpus)} characters")
#     print(f"Without BPE merges:       {len(ids_no_merge)} tokens")
#     print(f"With BPE merges:          {len(ids)} tokens")
#     print(f"Tokens saved:             {len(ids_no_merge) - len(ids)}")
#     print(f"Compression ratio:        {len(ids_no_merge) / len(ids):.2f}x")
    
#     print("\n" + "=" * 70)
