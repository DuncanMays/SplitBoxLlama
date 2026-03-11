import asyncio
from collections import defaultdict
from typing import Any, Callable


class EventEmitter:
    def __init__(self):
        self._listeners: dict[str, list[Callable]] = defaultdict(list)

    def on(self, event: str, listener: Callable) -> None:
        self._listeners[event].append(listener)

    def off(self, event: str, listener: Callable) -> None:
        self._listeners[event].remove(listener)

    def once(self, event: str, listener: Callable) -> None:
        def wrapper(*args, **kwargs):
            self.off(event, wrapper)
            return listener(*args, **kwargs)
        self.on(event, wrapper)

    def emit(self, event: str, *args: Any, **kwargs: Any) -> bool:
        listeners = list(self._listeners.get(event, []))
        if not listeners:
            return False
        for listener in listeners:
            result = listener(*args, **kwargs)
            if asyncio.iscoroutine(result):
                asyncio.create_task(result)
        return True
