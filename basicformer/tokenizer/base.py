from abc import ABC, abstractmethod
from basicformer.utils.BiDict import BiDict

"""
Vocab: a mapping from int (token ID in the vocabulary) to bytes (token bytes)
The 
"""
Vocab = BiDict[int, bytes]  

class BaseTokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        pass

    @abstractmethod
    def get_vocab(self) -> Vocab:
        pass

    @abstractmethod
    def train(self, corpus: str, max_vocab_size: int, special_tokens: list[str]) -> None:
        pass