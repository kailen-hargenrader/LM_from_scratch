from typing import Generic, Optional, TypeVar

T = TypeVar('T')

class DoublyLinkedListNode(Generic[T]):
    def __init__(self, data: T):
        self.data: T = data
        self.prev: Optional['DoublyLinkedListNode[T]'] = None
        self.next: Optional['DoublyLinkedListNode[T]'] = None

    def __repr__(self):
        prev_id = id(self.prev) if self.prev else None
        next_id = id(self.next) if self.next else None
        return f"DoublyLinkedListNode(data={self.data!r}, prev={prev_id}, next={next_id})"

class DoublyLinkedListNodePair(Generic[T]):
    def __init__(self, node1: DoublyLinkedListNode[T], node2: DoublyLinkedListNode[T], priority: int):
        self.node1 = node1
        self.node2 = node2
        self.priority = priority

    def __lt__(self, other):
        if not isinstance(other, DoublyLinkedListNodePair):
            return NotImplemented
        return self.priority < other.priority

    def __eq__(self, other):
        if not isinstance(other, DoublyLinkedListNodePair):
            return NotImplemented
        return self.priority == other.priority

    def __repr__(self):
        return (f"DoublyLinkedListNodePair(node1={repr(self.node1)}, "
                f"node2={repr(self.node2)}, priority={self.priority})")


class DoublyLinkedList(Generic[T]):
    def __init__(self, items: list[T]):
        """Initializes the doubly linked list from a list of items."""
        if not items:
            self.first: Optional['DoublyLinkedListNode[T]'] = None
            self.current: Optional['DoublyLinkedListNode[T]'] = None
            return

        head = DoublyLinkedListNode[T](items[0])
        prev_node = head
        for item in items[1:]:
            node = DoublyLinkedListNode[T](item)
            prev_node.next = node
            node.prev = prev_node
            prev_node = node
        self.first = head
        self.current = head

    def increment(self) -> bool:
        """Move the current node to the next node if possible. Returns True if moved, False otherwise."""
        if self.current and self.current.next:
            self.current = self.current.next
            return True
        return False

    def decrement(self) -> bool:
        """Move the current node to the previous node if possible. Returns True if moved, False otherwise."""
        if self.current and self.current.prev:
            self.current = self.current.prev
            return True
        return False

    def to_list(self) -> list[T]:
        """Converts the doubly linked list to a list of items."""
        result = []
        node = self.first
        while node:
            result.append(node.data)
            node = node.next
        return result
 