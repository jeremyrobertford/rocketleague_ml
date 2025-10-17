from __future__ import annotations
from typing import TYPE_CHECKING
from rocketleague_ml.core.actor import Actor
from rocketleague_ml.core.frame import Frame
from rocketleague_ml.core.car_component import (
    Simple_Car_Component,
    Flip_Car_Component,
)

if TYPE_CHECKING:
    from rocketleague_ml.models.frame_by_frame_processor import Frame_By_Frame_Processor


def process_component_usage_in_air(
    processor: Frame_By_Frame_Processor, updated_actor: Actor, frame: Frame
):
    car_component = frame.game.car_components.get(updated_actor.actor_id)
    if not car_component:
        frame.add_updated_actor_to_disconnected_car_component_updates(updated_actor)
        return None
    attribute = updated_actor.attribute
    if attribute is None or "Int" not in attribute:
        raise ValueError(
            f"Component usage in air must have Int attribute {updated_actor.raw}"
        )
    usage_number = attribute.get("Int")
    field_label = car_component.car.player.name + "_component_usage_in_air"
    if car_component.secondary_category == "boost":
        if processor.include_boost_management:
            frame.processed_fields[field_label + "_boost"] = usage_number
    elif processor.include_movement:
        frame.processed_fields[field_label + "_" + car_component.secondary_category] = (
            usage_number
        )
    return None


def process_jump(
    processor: Frame_By_Frame_Processor, updated_actor: Actor, frame: Frame
):
    car_component = frame.game.car_components.get(updated_actor.actor_id)
    if not car_component:
        frame.add_updated_actor_to_disconnected_car_component_updates(updated_actor)
        return None
    attribute = updated_actor.attribute
    if attribute is None or "Int" not in attribute:
        raise ValueError(f"Jump must have Int attribute {updated_actor.raw}")
    if processor.include_movement:
        jump_number = attribute.get("Int")
        field_label = car_component.car.player.name + "_jump"
        frame.processed_fields[field_label] = jump_number
    return None


def process_double_jump(
    processor: Frame_By_Frame_Processor, updated_actor: Actor, frame: Frame
):
    car_component = frame.game.car_components.get(updated_actor.actor_id)
    if not car_component:
        frame.add_updated_actor_to_disconnected_car_component_updates(updated_actor)
        return None
    attribute = updated_actor.attribute
    if attribute is None or "Location" not in attribute:
        raise ValueError(
            f"Double jump must have Location attribute {updated_actor.raw}"
        )
    if processor.include_movement:
        field_label = car_component.car.player.name + "_double_jump"
        frame.processed_fields[field_label + "_x"] = attribute["Location"]["x"]
        frame.processed_fields[field_label + "_y"] = attribute["Location"]["y"]
        frame.processed_fields[field_label + "_z"] = attribute["Location"]["z"]
    # TODO: mechanics.jumps
    # TODO: mechanics.double_jumps
    return None


def process_activate_component(
    processor: Frame_By_Frame_Processor, updated_actor: Actor, frame: Frame
):
    car_component = frame.game.car_components.get(updated_actor.actor_id)
    if not car_component:
        frame.add_updated_actor_to_disconnected_car_component_updates(updated_actor)
        return None
    car_component.update_activity(updated_actor)
    if not car_component.secondary_category:
        raise ValueError(f"Failed to update car component {updated_actor.raw}")
    if (
        processor.include_boost_management
        and car_component.secondary_category == "boost"
    ):
        if updated_actor.attribute and updated_actor.attribute.get("Byte"):
            frame.processed_fields[
                car_component.car.player.name
                + "_"
                + car_component.secondary_category
                + "_byte"
            ] = updated_actor.attribute.get("Byte")
    if processor.include_movement and car_component.secondary_category != "boost":
        if not updated_actor.attribute or "Byte" not in updated_actor.attribute:
            raise ValueError(
                f"Byte not given for car component activity {updated_actor}"
            )
        new_bytes = updated_actor.attribute["Byte"]
        if (
            frame.resync
            and car_component.previous_activity - 25 <= new_bytes
            and new_bytes <= car_component.previous_activity
        ):
            return None
        if updated_actor.attribute and updated_actor.attribute.get("Byte"):
            frame.processed_fields[
                car_component.car.player.name
                + "_"
                + car_component.secondary_category
                + "_byte"
            ] = updated_actor.attribute.get("Byte")
    return None


def process_dodge(
    processor: Frame_By_Frame_Processor, updated_actor: Actor, frame: Frame
):
    car_component = frame.game.car_components.get(updated_actor.actor_id)
    if not car_component:
        frame.add_updated_actor_to_disconnected_car_component_updates(updated_actor)
        return None
    if not isinstance(car_component, Flip_Car_Component):
        raise ValueError(f"Car component cannot dodge {updated_actor.raw}")
    new_dodge_occurred = car_component.dodge(updated_actor)
    if new_dodge_occurred and car_component.last_flip and processor.include_movement:
        field_label = car_component.car.player.name + "_dodge_torque"
        frame.processed_fields[field_label + "_x"] = car_component.last_flip.x
        frame.processed_fields[field_label + "_y"] = car_component.last_flip.y
        frame.processed_fields[field_label + "_z"] = car_component.last_flip.z
    return None


def process_boost(
    processor: Frame_By_Frame_Processor, updated_actor: Actor, frame: Frame
):
    car_component = frame.game.car_components.get(updated_actor.actor_id)
    if not car_component:
        frame.add_updated_actor_to_disconnected_car_component_updates(updated_actor)
        return None
    if not car_component.car.boost:
        raise ValueError(
            f"Car component car does not have boost to update {updated_actor.raw}"
        )
    car_component.car.boost.update_amount(updated_actor)
    if processor.include_boost_management:
        frame.processed_fields[car_component.car.player.name + "_boost_amount"] = (
            car_component.car.boost.amount
        )
    return None


def process_steer(
    processor: Frame_By_Frame_Processor, updated_actor: Actor, frame: Frame
):
    car = frame.game.cars.get(updated_actor.actor_id)
    if not car:
        frame.add_updated_actor_to_disconnected_car_component_updates(updated_actor)
        return None
    attribute = updated_actor.attribute
    if attribute is None or "Byte" not in attribute:
        raise ValueError(f"Car steering does not have attribute {updated_actor.raw}")
    car_component = Simple_Car_Component(updated_actor, car)
    if not car.steer:
        car.steer = car_component
    car.steer.update_amount(car_component)
    if processor.include_movement:
        frame.processed_fields[car.player.name + "_steering"] = car_component.amount
    # TODO: sharp_turns, full_turns, slight_turns
    return None


def process_activate_handbrake(
    processor: Frame_By_Frame_Processor, updated_actor: Actor, frame: Frame
):
    car = frame.game.cars.get(updated_actor.actor_id)
    if not car:
        frame.add_updated_actor_to_disconnected_car_component_updates(updated_actor)
        return None
    attribute = updated_actor.attribute
    if attribute is None or "Boolean" not in attribute:
        raise ValueError(f"Car handbrake does not have attribute {updated_actor.raw}")
    if not car.handbrake:
        car.handbrake = Simple_Car_Component(updated_actor, car)
    car.handbrake.update_activity(updated_actor)
    if processor.include_movement:
        frame.processed_fields[car.player.name + "_drift_active"] = (
            1 if attribute["Boolean"] else 0
        )
    return None


def process_car_component(
    processor: Frame_By_Frame_Processor, updated_actor: Actor, frame: Frame
):
    car_component = frame.game.car_components.get(updated_actor.actor_id)
    car = frame.game.cars.get(updated_actor.actor_id)
    if updated_actor.secondary_category in [
        "throttle",
        "handbrake",
        "driving",
        "component_usage_in_air",
    ]:
        return None

    if (
        not car_component
        and not car
        and frame.resync
        and frame.game.disconnected_car_components.get(updated_actor.actor_id)
    ):
        return None

    if (
        not car_component
        and not car
        and frame.game.boost_pads.get(updated_actor.actor_id)
    ):
        # I have no idea why this happens
        return None

    if not car_component and not car:
        frame.add_updated_actor_to_disconnected_car_component_updates(updated_actor)
        return None

    raise ValueError(f"Unknown car component not updated {updated_actor.raw}")
