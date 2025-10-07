from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, cast
from rocketleague_ml.config import ROUND_LENGTH  # type: ignore
from rocketleague_ml.types.attributes import (
    Actor_Export,
    Time_Labeled_Boost_Pickup,
    Time_Labeled_Activity,
    Time_Labeled_Amount,
)
from rocketleague_ml.utils.helpers import convert_byte_to_float
from rocketleague_ml.core.actor import Actor

if TYPE_CHECKING:
    from rocketleague_ml.core.car import Car


class Car_Component(Actor):
    def __init__(self, car_component: Actor, car: Car):
        super().__init__(car_component.raw, car_component.objects)
        self.car: Car = car
        self.amount = 0.0
        self.active = False
        self.activity_updates_by_round: Dict[int, List[Time_Labeled_Activity]] = {}
        self.amount_updates_by_round: Dict[int, List[Time_Labeled_Amount]] = {}

    def update_amount(self, updated_car_component: Actor):
        attribute = updated_car_component.raw["attribute"]
        if not attribute or ("Float" not in attribute and "Byte" not in attribute):
            raise ValueError(
                f"Car component updated without value {updated_car_component.raw}"
            )

        new_amount = None
        if "Float" in attribute:
            new_amount = attribute["Float"]
        if "Byte" in attribute:
            new_amount = convert_byte_to_float(attribute["Byte"])

        if new_amount is None:
            raise ValueError(
                f"Car component updated with unknown value {updated_car_component.raw}"
            )
        return None

    def update_activity(self, updated_actor: Actor):
        attribute = updated_actor.attribute
        if not attribute:
            raise ValueError(
                f"Attribute not found when updating activity for {self.actor_id}: {updated_actor.raw}"
            )
        active: bool | None = attribute.get("Boolean")
        if active is None:
            active = attribute.get("Byte") == 3 if attribute.get("Byte") else None
            if active is None:
                raise ValueError(
                    f"Boolean not found when updating activity for {self.actor_id}: {updated_actor.raw}"
                )

        self.active = active
        return None

    def to_dict(self):
        base_car_component = super().to_dict()
        return cast(Actor_Export, base_car_component | {"amount": self.amount})


class Boost_Car_Component(Car_Component):
    def __init__(self, car_component: Car_Component):
        super().__init__(car_component, car_component.car)
        self.pickups = 0
        self.pickup_updates_by_round: Dict[int, List[Time_Labeled_Boost_Pickup]] = {}

    def update_amount(self, updated_car_component: Actor):
        if (
            not updated_car_component.attribute
            or "ReplicatedBoost" not in updated_car_component.attribute
        ):
            raise ValueError(
                f"Boost car component updated without value {updated_car_component.raw}"
            )

        replicated_boost = updated_car_component.attribute["ReplicatedBoost"]
        self.pickups = replicated_boost["grant_count"]
        self.amount = replicated_boost["boost_amount"]
        return None
