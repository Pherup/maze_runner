from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)

    def __init__(self,priority, item):
        self.priority = priority
        self.item = item
