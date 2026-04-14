from typing import Dict, Generic, Iterator, Tuple, TypeVar, Optional

K = TypeVar('K')
V = TypeVar('V')

class BiDict(Dict[K, V], Generic[K, V]):
    """
    Bidirectional dictionary that allows you to look up value by key and key by value in O(1) time.
    Enforces a 1-to-1 mapping.
    Behaves like a regular dict for all normal key->value access and iteration.
    Use get_key(value) for value->key lookup.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._inverse: Dict[V, K] = dict()
        self.update(*args, **kwargs)

    def __setitem__(self, key: K, value: V) -> None:
        if key in self:
            old_value = super().__getitem__(key)
            if old_value == value:
                return
            del self._inverse[old_value]
        if value in self._inverse and self._inverse[value] != key:
            raise ValueError(f"Value {value} is already associated with key {self._inverse[value]}")
        super().__setitem__(key, value)
        self._inverse[value] = key

    def __delitem__(self, key: K) -> None:
        value = super().__getitem__(key)
        super().__delitem__(key)
        del self._inverse[value]

    def __getitem__(self, key: K) -> V:
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        return super().__contains__(key)

    def get_key(self, value: V, default: Optional[K] = None) -> Optional[K]:
        """
        Get the key for a given value.
        Allows: k = bd.get_key(v)
        """
        return self._inverse.get(value, default)

    def pop(self, key: K, *args) -> V:
        if key in self:
            value = self[key]
            del self._inverse[value]
            return super().pop(key)
        elif args:
            return args[0]
        else:
            raise KeyError(key)

    def popitem(self) -> Tuple[K, V]:
        key, value = super().popitem()
        del self._inverse[value]
        return key, value

    def clear(self) -> None:
        super().clear()
        self._inverse.clear()

    def update(self, *args, **kwargs) -> None:
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def inverse(self) -> 'BiDict[V, K]':
        """Get the inverse BiDict (with keys and values swapped)."""
        inv = BiDict[V, K]()
        for v, k in self._inverse.items():
            inv[v] = k
        return inv

    def __repr__(self) -> str:
        return f"BiDict({super().__repr__()})"

    def __copy__(self) -> 'BiDict[K, V]':
        return BiDict(self)

    def __deepcopy__(self, memodict={}) -> 'BiDict[K, V]':
        import copy
        return BiDict(copy.deepcopy(dict(self), memodict))

    # Iteration and dict-like methods remain unchanged;
    # They behave just like a normal dict.
    def keys(self):
        return super().keys()

    def values(self):
        return super().values()

    def items(self):
        return super().items()

    def __iter__(self) -> Iterator[K]:
        return super().__iter__()

    def __len__(self) -> int:
        return super().__len__()

# In summary:
# Yes, with this version, you can write type-hinted specializations like: Vocab = BiDict[int, bytes]
# And type checkers (like mypy, pyright) will enforce usage for those types.