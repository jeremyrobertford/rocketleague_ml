from __future__ import annotations
from typing import TYPE_CHECKING
from rocketleague_ml.utils.helpers import convert_byte_to_float
from rocketleague_ml.core.actor import Actor
from rocketleague_ml.core.rigid_body import Rigid_Body
from rocketleague_ml.core.positioning import Position

if TYPE_CHECKING:
    from rocketleague_ml.core.car import Car


class Simple_Car_Component(Actor):
    def __init__(self, car_component: Actor, car: Car):
        super().__init__(car_component.raw, car_component.objects)
        self.car: Car = car
        self.amount = 0.0
        self.active = False

        if not car_component.attribute:
            return None

        if "Float" in car_component.attribute:
            self.amount = car_component.attribute["Float"]
            return None
        if "Byte" in car_component.attribute:
            self.amount = convert_byte_to_float(car_component.attribute["Byte"])
            self.active = car_component.attribute["Byte"] == 3
            return None

        if "Boolean" in car_component.attribute:
            self.active = car_component.attribute["Boolean"]
            return None

    def update_amount(self, updated_car_component: Actor):
        attribute = updated_car_component.attribute
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


class Car_Component(Rigid_Body):
    def __init__(self, car_component: Actor, car: Car):
        super().__init__(
            car_component, f"{car.player.name}_{car_component.secondary_category}"
        )
        self.car: Car = car
        self.amount = 0.0
        self.active = False

        if not car_component.attribute:
            return None

        if "Float" in car_component.attribute:
            self.amount = car_component.attribute["Float"]
            return None
        if "Byte" in car_component.attribute:
            self.amount = convert_byte_to_float(car_component.attribute["Byte"])
            self.active = car_component.attribute["Byte"] == 3
            return None

        if "Boolean" in car_component.attribute:
            self.active = car_component.attribute["Boolean"]
            return None

    def update_amount(self, updated_car_component: Actor):
        attribute = updated_car_component.attribute
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


class Boost_Car_Component(Car_Component):
    def __init__(self, car_component: Car_Component):
        super().__init__(car_component, car_component.car)
        self.pickups = 0
        self.amount = 0

        if not car_component.attribute:
            return None
        if "ReplicatedBoost" not in car_component.attribute:
            return None

        replicated_boost = car_component.attribute["ReplicatedBoost"]
        self.pickups = replicated_boost["grant_count"]
        self.amount = replicated_boost["boost_amount"]
        return None

    def update_amount(self, updated_car_component: Actor):
        if (
            not updated_car_component.attribute
            or "ReplicatedBoost" not in updated_car_component.attribute
        ):
            raise ValueError(
                f"Boost car component created without value {updated_car_component.raw}"
            )

        replicated_boost = updated_car_component.attribute["ReplicatedBoost"]
        self.pickups = replicated_boost["grant_count"]
        self.amount = replicated_boost["boost_amount"]
        return None


class Dodge_Car_Component(Car_Component):
    def __init__(self, car_component: Car_Component):
        super().__init__(car_component, car_component.car)
        self.last_dodge: Position | None = None
        return None

    def dodge(self, dodge_actor: Actor) -> bool:
        if not dodge_actor.attribute or "Location" not in dodge_actor.attribute:
            raise ValueError(
                f"Cannot perform dodge without dodge actor {dodge_actor.raw}"
            )
        dodge_torque = Position(dodge_actor.attribute["Location"])
        if dodge_torque == self.last_dodge:
            return False
        self.last_dodge = dodge_torque
        return True
