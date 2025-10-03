from __future__ import annotations
from typing import Dict, List, cast
from rocketleague_ml.config import ROUND_LENGTH  # type: ignore
from rocketleague_ml.types.attributes import (
    Actor_Export,
    Time_Labeled_Boost_Pickup,
    Time_Labeled_Rigid_Body_Positioning,
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
                    "match_time": frame["match_time"],
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
        active: bool | None = attribute.get("Boolean")
        if active is None:
            active = attribute.get("Byte") == 3 if attribute.get("Byte") else None
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
                    "match_time": frame["match_time"],
                    "delta": frame["delta"],
                    "active": active,
                }
            )

    def to_dict(self):
        base_car_component = super().to_dict()
        return cast(Actor_Export, base_car_component | {"amount": self.amount})


class Boost_Car_Component(Car_Component):
    def __init__(self, car_component: Car_Component | Actor):
        super().__init__(car_component)
        self.pickup_updates_by_round: Dict[int, List[Time_Labeled_Boost_Pickup]] = {}

    def update_amount(
        self,
        updated_car_component: Actor,
        active_game: bool,
        frame: Raw_Frame,
        round: int,
    ):
        attribute = updated_car_component.raw["attribute"]
        if not attribute or "ReplicatedBoost" not in attribute:
            raise ValueError(
                f"Boost car component updated without value {updated_car_component.raw}"
            )

        replicated_boost = attribute["ReplicatedBoost"]
        if round not in self.amount_updates_by_round:
            self.amount_updates_by_round[round] = []
        self.amount_updates_by_round[round].append(
            {
                "round": round,
                "time": frame["time"],
                "match_time": (
                    frame["match_time"] if "match_time" in frame else ROUND_LENGTH
                ),
                "delta": frame["delta"],
                "amount": replicated_boost["boost_amount"],
            }
        )
        return

    def pickup(
        self,
        boost_data: tuple[float, float, float, int],
        boost_actor: Actor,
        frame: Raw_Frame,
        round: int,
    ):
        if round not in self.pickup_updates_by_round:
            self.pickup_updates_by_round[round] = []
        self.pickup_updates_by_round[round].append(
            {
                "round": round,
                "time": frame["time"],
                "match_time": frame["match_time"],
                "delta": frame["delta"],
                "location": {
                    "x": boost_data[1],
                    "y": boost_data[1],
                    "z": boost_data[2],
                },
                "amount": boost_data[3],
            }
        )
        return


class Dodge_Car_Component(Car_Component):
    def __init__(self, car_component: Car_Component | Actor):
        super().__init__(car_component)
        self.dodges_by_round: Dict[int, List[Time_Labeled_Rigid_Body_Positioning]] = {}

    def dodge(
        self,
        dodge_actor: Car_Component | Actor,
        active_game: bool,
        frame: Raw_Frame,
        round: int,
    ):
        if round not in self.updates_by_round:
            self.dodges_by_round[round] = []
        time_labeled_positioning = cast(
            Time_Labeled_Rigid_Body_Positioning, dodge_actor.to_dict()
        )
        time_labeled_positioning["round"] = round
        time_labeled_positioning["time"] = frame["time"]
        time_labeled_positioning["match_time"] = frame["match_time"]
        time_labeled_positioning["delta"] = frame["delta"]
        self.dodges_by_round[round].append(time_labeled_positioning)


class Car(Actor):
    def __init__(self, car: Actor):
        super().__init__(car.raw, car.objects)
        if not self.positioning:
            raise ValueError(f"Car failed to position {car.raw}")
        self.boost: Boost_Car_Component | None = None
        self.jump: Car_Component | None = None
        self.dodge: Dodge_Car_Component | None = None
        self.flip: Car_Component | None = None
        self.double_jump: Car_Component | None = None
        self.steer: Car_Component | None = None
        self.throttle: Car_Component | None = None
        self.handbrake: Car_Component | None = None
        self.demos: List[Time_Labeled_Attacker_Demo] = []
        self.got_demod: List[Time_Labeled_Victim_Demo] = []

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

    def assign_components(self, car_components: List[Car_Component]):
        for car_component in car_components:
            if not self.owns(car_component):
                continue
            if car_component.secondary_category == "boost":
                if not self.boost:
                    self.boost = Boost_Car_Component(car_component)
                else:
                    self.boost.actor_id = car_component.actor_id
                continue
            if car_component.secondary_category == "dodge":
                if not self.dodge:
                    self.dodge = Dodge_Car_Component(car_component)
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
            if not self.owns(car_component):
                continue
            attribute = car_component.raw.get("attribute")
            active_car_component = (
                attribute["Byte"] == 3 if attribute and "Byte" in attribute else None
            )
            dodge_component = attribute and "Location" in attribute

            # Assign new components
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

            # Custom component usage
            elif dodge_component and self.dodge:
                self.dodge.dodge(car_component, active_game, frame, round)

            # Update current component activity
            elif (
                active_car_component is not None
                and self.jump
                and self.jump.is_self(car_component)
            ):
                self.jump.update_activity(car_component, active_game, frame, round)
            elif (
                active_car_component is not None
                and self.double_jump
                and self.double_jump.is_self(car_component)
            ):
                self.double_jump.update_activity(
                    car_component, active_game, frame, round
                )
            elif (
                active_car_component is not None
                and self.boost
                and self.boost.is_self(car_component)
            ):
                self.boost.update_activity(car_component, active_game, frame, round)
            elif (
                active_car_component is not None
                and self.flip
                and self.flip.is_self(car_component)
            ):
                self.flip.update_activity(car_component, active_game, frame, round)
            elif (
                active_car_component is not None
                and self.dodge
                and self.dodge.is_self(car_component)
            ):
                self.dodge.update_activity(car_component, active_game, frame, round)

            # Update current component amount
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
            raise ValueError(f"Demo occurred without attribute {demo_actor.raw}")
        demolition: Demolition_Attribute | None = attribute.get("DemolishExtended")
        if not demolition:
            raise ValueError(f"Demo occurred without demolish {demo_actor.raw}")
        if not self.positioning or not self.positioning.linear_velocity:
            raise ValueError(f"Demo occurred without speed {demo_actor.raw}")
        if not victim_car.positioning:
            raise ValueError(
                f"Demo occurred without victim positioning {demo_actor.raw}"
            )

        curr = victim_car.positioning.linear_velocity
        prev = victim_car.positioning.previous_linear_velocity
        demod_at = prev if curr is None else curr
        self.demos.append(
            {
                "round": round,
                "time": frame["time"],
                "match_time": frame["match_time"],
                "delta": frame["delta"],
                "linear_velocity": self.positioning.linear_velocity.to_dict(),
                "collision": demolition["attacker_velocity"],
                "victim_actor_id": demolition["victim"]["actor"],
                "victim_linear_velocity": demod_at.to_dict() if demod_at else None,
                "victim_collision": demolition["victim_velocity"],
            }
        )

    def demod(self, attacker_car: Car, demo_actor: Actor, frame: Raw_Frame, round: int):
        attribute = demo_actor.raw.get("attribute")
        if not attribute:
            raise ValueError(f"Demod occurred without attribute {demo_actor.raw}")
        demolition: Demolition_Attribute | None = attribute.get("DemolishExtended")
        if not demolition:
            raise ValueError(f"Demod occurred without demolish {demo_actor.raw}")
        if not self.positioning:
            raise ValueError(
                f"Demod occurred without victim positioning {demo_actor.raw}"
            )
        if not attacker_car.positioning or not attacker_car.positioning.linear_velocity:
            raise ValueError(f"Demod occurred without attacker speed {demo_actor.raw}")

        curr = self.positioning.linear_velocity
        prev = self.positioning.previous_linear_velocity
        demod_at = prev if curr is None else curr
        self.got_demod.append(
            {
                "round": round,
                "time": frame["time"],
                "match_time": frame["match_time"],
                "delta": frame["delta"],
                "linear_velocity": demod_at.to_dict() if demod_at else None,
                "collision": demolition["victim_velocity"],
                "attacker_actor_id": demolition["attacker"]["actor"],
                "attacker_collision": demolition["attacker_velocity"],
                "attacker_linear_velocity": attacker_car.positioning.linear_velocity.to_dict(),
            }
        )

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
