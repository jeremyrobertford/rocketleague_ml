from typing import List
from rocketleague_ml.utils.helpers import convert_byte_to_float
from rocketleague_ml.core.positioning import Positioning
from rocketleague_ml.core.actor import Actor


class Car_Component(Actor):
    def __init__(self, car_component: Actor):
        super().__init__(car_component.raw, car_component.objects)
        self.amount = 0.0

    def update_amount(self, updated_car_component: Actor):
        attribute = updated_car_component.raw["attribute"]
        if not attribute or ("Float" not in attribute and "Byte" not in attribute):
            raise ValueError("Car component updated without value {car_component.raw}")

        if "Float" in attribute:
            self.amount = attribute["Float"]
            return
        if "Byte" in attribute:
            self.amount = convert_byte_to_float(attribute["Byte"])
            return

        raise ValueError("Car component updated with unknown value {car_component.raw}")


class Car(Actor):
    def __init__(self, car: Actor):
        super().__init__(car.raw, car.objects)
        self.positioning: Positioning | None = None
        self.boost: Car_Component | None = None
        self.jump: Car_Component | None = None
        self.dodge: Car_Component | None = None
        self.flip: Car_Component | None = None
        self.double_jump: Car_Component | None = None
        self.steer: Car_Component | None = None
        self.throttle: Car_Component | None = None
        self.handbrake: Car_Component | None = None

    def assign_components(self, car_components: List[Car_Component]):
        for car_component in car_components:
            if not self.owns(car_component):
                continue
            if car_component.secondary_category == "boost":
                self.boost = car_component
                continue
            if car_component.secondary_category == "dodge":
                self.dodge = car_component
                continue
            if car_component.secondary_category == "double_jump":
                self.double_jump = car_component
                continue
            if car_component.secondary_category == "flip":
                self.flip = car_component
                continue
            if car_component.secondary_category == "jump":
                self.jump = car_component
                continue

    def update_components(self, car_components: List[Car_Component]):
        for car_component in car_components:
            if not self.owns(car_component):
                continue
            if self.steer is None and car_component.secondary_category == "steer":
                self.steer = car_component
            elif (
                self.throttle is None and car_component.secondary_category == "throttle"
            ):
                self.throttle = car_component
            elif (
                self.handbrake is None
                and car_component.secondary_category == "handbrake"
            ):
                self.handbrake = car_component
            elif self.jump and car_component.secondary_category == "rep_jump":
                self.jump.update_amount(car_component)
            elif self.dodge and car_component.secondary_category == "rep_dodge":
                self.dodge.update_amount(car_component)
            elif self.boost and car_component.secondary_category == "rep_boost":
                self.boost.update_amount(car_component)
            elif self.boost and car_component.secondary_category == "rep_boost_amt":
                self.boost.update_amount(car_component)
            elif self.steer and car_component.secondary_category == "steer":
                self.steer.update_amount(car_component)
            elif self.throttle and car_component.secondary_category == "throttle":
                self.throttle.update_amount(car_component)
