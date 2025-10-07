from __future__ import annotations
from typing import TYPE_CHECKING, cast
from rocketleague_ml.config import ROUND_LENGTH  # type: ignore
from rocketleague_ml.types.attributes import (
    Actor_Export,
)
from rocketleague_ml.core.actor import Actor
from rocketleague_ml.core.car_component import Car_Component, Boost_Car_Component

if TYPE_CHECKING:
    from rocketleague_ml.core.player import Player


class Car(Actor):
    def __init__(self, car: Actor, player: Player):
        super().__init__(car.raw, car.objects)
        if not self.positioning:
            raise ValueError(f"Car failed to position {car.raw}")
        self._boost: Boost_Car_Component | None = None
        self._jump: Car_Component | None = None
        self._dodge: Car_Component | None = None
        self._flip: Car_Component | None = None
        self._double_jump: Car_Component | None = None
        self.steer: Car_Component | None = None
        self.throttle: Car_Component | None = None
        self.handbrake: Car_Component | None = None
        self.player: Player = player

    @property
    def boost(self) -> Boost_Car_Component:
        cc = self._boost
        if cc is None:
            raise ValueError("Car boost not assigned")
        return cc

    @boost.setter
    def boost(self, new_boost: Boost_Car_Component):
        self._boost = new_boost

    @property
    def dodge(self) -> Car_Component:
        cc = self._dodge
        if cc is None:
            raise ValueError("Car dodge not assigned")
        return cc

    @dodge.setter
    def dodge(self, new_dodge: Car_Component):
        self._dodge = new_dodge

    @property
    def jump(self) -> Car_Component:
        cc = self._jump
        if cc is None:
            raise ValueError("Car jump not assigned")
        return cc

    @jump.setter
    def jump(self, new_jump: Car_Component):
        self._jump = new_jump

    @property
    def double_jump(self) -> Car_Component:
        cc = self._double_jump
        if cc is None:
            raise ValueError("Car double jump not assigned")
        return cc

    @double_jump.setter
    def double_jump(self, new_double_jump: Car_Component):
        self._double_jump = new_double_jump

    @property
    def flip(self) -> Car_Component:
        cc = self._flip
        if cc is None:
            raise ValueError("Car flip not assigned")
        return cc

    @flip.setter
    def flip(self, new_flip: Car_Component):
        self._flip = new_flip

    def update_component_connections(self):
        if self.boost:
            self.boost.active_actor_id = self.actor_id
        if self.jump:
            self.jump.active_actor_id = self.actor_id
        if self.flip:
            self.flip.active_actor_id = self.actor_id
        if self.dodge:
            self.dodge.active_actor_id = self.actor_id
        if self.double_jump:
            self.double_jump.active_actor_id = self.actor_id
        if self.steer:
            self.steer.active_actor_id = self.actor_id
        if self.throttle:
            self.throttle.active_actor_id = self.actor_id
        if self.handbrake:
            self.handbrake.active_actor_id = self.actor_id

    def owns(self, possible_child: Actor):
        if self.actor_id == possible_child.active_actor_id:
            return True
        if self.boost and self.boost.is_self(possible_child):
            return True
        if self.jump and self.jump.is_self(possible_child):
            return True
        if self.dodge and self.dodge.is_self(possible_child):
            return True
        if self.flip and self.flip.is_self(possible_child):
            return True
        if self.double_jump and self.double_jump.is_self(possible_child):
            return True
        if self.steer and self.steer.is_self(possible_child):
            return True
        if self.throttle and self.throttle.is_self(possible_child):
            return True
        if self.handbrake and self.handbrake.is_self(possible_child):
            return True
        return False

    def to_dict(self):
        base_car = super().to_dict()
        components = {
            "boost": self.boost.to_dict() if self.boost else None,
            "jump": self.jump.to_dict() if self.jump else None,
            "dodge": self.dodge.to_dict() if self.dodge else None,
            "flip": self.flip.to_dict() if self.flip else None,
            "double_jump": self.double_jump.to_dict() if self.double_jump else None,
            "steer": self.steer.to_dict() if self.steer else None,
            "throttle": self.throttle.to_dict() if self.throttle else None,
            "handbrake": self.handbrake.to_dict() if self.handbrake else None,
        }
        return cast(Actor_Export, base_car | components)
