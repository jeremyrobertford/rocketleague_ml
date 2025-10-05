from typing import Any


class Logger:
    def __init__(self, active: bool = False):
        self.active = active

    def print(self, *args: Any, **kwargs: Any) -> None:
        if self.active:
            print(*args, **kwargs)
