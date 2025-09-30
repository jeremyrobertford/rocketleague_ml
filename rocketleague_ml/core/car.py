from __future__ import annotations
from typing import Dict, List, cast
from rocketleague_ml.types.attributes import (
    Actor_Export,
    Time_Labeled_Attacker_Demo,
    Time_Labeled_Victim_Demo,
    Time_Labeled_Activity,
    Time_Labeled_Amount,
    Demolition_Attribute,
    Raw_Frame,
)
from rocketleague_ml.utils.helpers import convert_byte_to_float
from rocketleague_ml.core.actor import Actor


class Car_Component(Actor):
    def __init__(self, car_component: Actor):
        super().__init__(car_component.raw, car_component.objects)
        self.amount = 0.0
        self.active = False
        self.activity_updates_by_round: Dict[int, List[Time_Labeled_Activity]] = {}
        self.amount_updates_by_round: Dict[int, List[Time_Labeled_Amount]] = {}

    def update_amount(
        self,
        updated_car_component: Actor,
        active_game: bool,
        frame: Raw_Frame,
        round: int,
    ):
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

        if new_amount is not None:
            if round not in self.amount_updates_by_round:
                self.amount_updates_by_round[round] = []
            self.amount_updates_by_round[round].append(
                {
                    "round": round,
                    "time": frame["time"],
                    "delta": frame["delta"],
                    "amount": new_amount,
                }
            )
            return

        raise ValueError(
            f"Car component updated with unknown value {updated_car_component.raw}"
        )

    def update_activity(
        self, updated_actor: Actor, active_game: bool, frame: Raw_Frame, round: int
    ):
        attribute = updated_actor.raw.get("attribute")
        if not attribute:
            raise ValueError(
                f"Attribute not found when updating activity for {self.actor_id}: {updated_actor.raw}"
            )
        active = attribute.get("Boolean")
        if active is None:
            raise ValueError(
                f"Boolean not found when updating activity for {self.actor_id}: {updated_actor.raw}"
            )

        self.active = active
        if active_game:
            if round not in self.activity_updates_by_round:
                self.activity_updates_by_round[round] = []
            self.activity_updates_by_round[round].append(
                {
                    "round": round,
                    "time": frame["time"],
                    "delta": frame["delta"],
                    "active": active,
                }
            )

    def to_dict(self):
        base_car_component = super().to_dict()
        return cast(Actor_Export, base_car_component | {"amount": self.amount})


class Car(Actor):
    def __init__(self, car: Actor):
        super().__init__(car.raw, car.objects)
        if not self.positioning:
            raise ValueError(f"Car failed to position {car.raw}")
        self.boost: Car_Component | None = None
        self.jump: Car_Component | None = None
        self.dodge: Car_Component | None = None
        self.flip: Car_Component | None = None
        self.double_jump: Car_Component | None = None
        self.steer: Car_Component | None = None
        self.throttle: Car_Component | None = None
        self.handbrake: Car_Component | None = None
        self.demos: List[Time_Labeled_Attacker_Demo] = []
        self.got_demod: List[Time_Labeled_Victim_Demo] = []

    def assign_components(self, car_components: List[Car_Component]):
        for car_component in car_components:
            if not self.owns(car_component):
                continue
            if car_component.secondary_category == "boost":
                if not self.boost:
                    self.boost = car_component
                else:
                    self.boost.actor_id = car_component.actor_id
                continue
            if car_component.secondary_category == "dodge":
                if not self.dodge:
                    self.dodge = car_component
                else:
                    self.dodge.actor_id = car_component.actor_id
                continue
            if car_component.secondary_category == "double_jump":
                if not self.double_jump:
                    self.double_jump = car_component
                else:
                    self.double_jump.actor_id = car_component.actor_id
                continue
            if car_component.secondary_category == "flip":
                if not self.flip:
                    self.flip = car_component
                else:
                    self.flip.actor_id = car_component.actor_id
                continue
            if car_component.secondary_category == "jump":
                if not self.jump:
                    self.jump = car_component
                else:
                    self.jump.actor_id = car_component.actor_id
                continue

    def update_components(
        self,
        car_components: List[Car_Component],
        active_game: bool,
        frame: Raw_Frame,
        round: int,
    ):
        for car_component in car_components:
            if self.owns(car_component):
                continue
            if self.steer is None and car_component.secondary_category == "steer":
                self.steer = car_component
            elif (
                self.throttle is None and car_component.secondary_category == "throttle"
            ):
                self.throttle = car_component
            elif self.handbrake is None and (
                car_component.secondary_category == "handbrake"
                or car_component.secondary_category == "handbrake_active"
            ):
                self.handbrake = car_component
            elif (
                self.handbrake
                and car_component.secondary_category == "handbrake_active"
            ):
                self.handbrake.update_activity(car_component, active_game, frame, round)
            elif self.jump and car_component.secondary_category == "rep_jump":
                self.jump.update_amount(car_component, active_game, frame, round)
            elif self.dodge and car_component.secondary_category == "rep_dodge":
                self.dodge.update_amount(car_component, active_game, frame, round)
            elif self.boost and car_component.secondary_category == "rep_boost":
                self.boost.update_amount(car_component, active_game, frame, round)
            elif self.boost and car_component.secondary_category == "rep_boost_amt":
                self.boost.update_amount(car_component, active_game, frame, round)
            elif self.steer and car_component.secondary_category == "steer":
                self.steer.update_amount(car_component, active_game, frame, round)
            elif self.throttle and car_component.secondary_category == "throttle":
                self.throttle.update_amount(car_component, active_game, frame, round)

    def demo(self, victim_car: Car, demo_actor: Actor, frame: Raw_Frame, round: int):
        attribute = demo_actor.raw.get("attribute")
        if not attribute:
            raise ValueError("Demo occurred without attribute {demo_actor}")
        demolition: Demolition_Attribute | None = attribute.get("DemolishExtended")
        if not demolition:
            raise ValueError("Demo occurred without demolish {demo_actor}")
        if self.actor_id == victim_car.actor_id:
            self.got_demod.append(
                {
                    "round": round,
                    "time": frame["time"],
                    "delta": frame["delta"],
                    "velocity": demolition["victim_velocity"],
                    "attacker_actor_id": demolition["attacker"]["actor"],
                    "attacker_velocity": demolition["attacker_velocity"],
                }
            )
        else:
            self.demos.append(
                {
                    "round": round,
                    "time": frame["time"],
                    "delta": frame["delta"],
                    "velocity": demolition["attacker_velocity"],
                    "victim_actor_id": demolition["victim"]["actor"],
                    "victim_velocity": demolition["victim_velocity"],
                }
            )
            victim_car.demo(victim_car, demo_actor, frame, round)

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
