from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rocketleague_ml.core.actor import Actor
    from rocketleague_ml.core.frame import Frame


class Logger:
    def __init__(self, active: bool = False):
        self.active = active

    def print(self, *args: Any, **kwargs: Any) -> None:
        if self.active:
            print(*args, **kwargs)

    def raise_exception(self, message: str, actor: Actor, frame: Frame) -> None:
        raise ValueError(f"{message}: {frame}, {actor}")
