from __future__ import annotations
from typing import TYPE_CHECKING
from rocketleague_ml.types.attributes import Rigid_Body_Attribute
from rocketleague_ml.core.actor import Actor
from rocketleague_ml.core.frame import Frame
from rocketleague_ml.core.car_component import (
    Simple_Car_Component,
    Flip_Car_Component,
)
from rocketleague_ml.core.rigid_body import Rigid_Body

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
    include_in_processing = (
        processor.include_boost_management
        and car_component.secondary_category == "boost"
    ) or (processor.include_movement and car_component.secondary_category != "boost")
    if not updated_actor.attribute or "Byte" not in updated_actor.attribute:
        raise ValueError(f"Byte not given for car component activity {updated_actor}")
    new_bytes = updated_actor.attribute["Byte"]
    if (
        frame.resync
        and car_component.previous_activity - 25 <= new_bytes
        and new_bytes <= car_component.previous_activity
    ):
        return None
    if include_in_processing:
        frame.processed_fields[
            car_component.car.player.name
            + "_"
            + car_component.secondary_category
            + "_activation"
        ] = car_component.activity
        frame.processed_fields[
            car_component.car.player.name
            + "_"
            + car_component.secondary_category
            + "_active"
        ] = (1 if car_component.active else 0)
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


def process_boost_pickup(
    processor: Frame_By_Frame_Processor, updated_actor: Actor, frame: Frame
):
    if not updated_actor.attribute or "PickupNew" not in updated_actor.attribute:
        processor.logger.raise_exception(
            "Boost pickup requires PickupNew attribute", updated_actor, frame
        )
        return None

    instigator = updated_actor.attribute["PickupNew"]["instigator"]
    if instigator in frame.game.cars:
        car = frame.game.cars[instigator]
        if updated_actor.actor_id not in frame.game.boost_pads:
            rigid_body_attribute: Rigid_Body_Attribute = {
                "RigidBody": {
                    "sleeping": True,
                    "location": {
                        "x": car.positioning.location.x,
                        "y": car.positioning.location.y,
                        "z": car.positioning.location.z,
                    },
                    "rotation": {"x": 0, "y": 0, "z": 0, "w": 0},
                    "linear_velocity": None,
                    "angular_velocity": None,
                    "previous_linear_velocity": None,
                }
            }
            updated_actor.attribute = rigid_body_attribute
            frame.game.boost_pads[updated_actor.actor_id] = Rigid_Body(
                updated_actor, "boost_pad"
            )
        boost_pad = frame.game.boost_pads[updated_actor.actor_id]

        if processor.include_boost_management:
            field_label = f"{car.player.name}_boost_pickup"
            frame.processed_fields[field_label + "_x"] = (
                boost_pad.positioning.location.x
            )
            frame.processed_fields[field_label + "_y"] = (
                boost_pad.positioning.location.y
            )
            frame.processed_fields[field_label + "_z"] = (
                boost_pad.positioning.location.z
            )
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
    return None


def process_throttle(
    processor: Frame_By_Frame_Processor, updated_actor: Actor, frame: Frame
):
    car = frame.game.cars.get(updated_actor.actor_id)
    if not car:
        frame.add_updated_actor_to_disconnected_car_component_updates(updated_actor)
        return None
    attribute = updated_actor.attribute
    if attribute is None or "Byte" not in attribute:
        raise ValueError(f"Car throttle does not have attribute {updated_actor.raw}")
    car_component = Simple_Car_Component(updated_actor, car)
    if not car.throttle:
        car.throttle = car_component
    car.throttle.update_amount(car_component)
    if processor.include_movement:
        frame.processed_fields[car.player.name + "_throttle"] = car_component.amount
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
        "driving",
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


def process_demo(
    processor: Frame_By_Frame_Processor, updated_actor: Actor, frame: Frame
):
    attribute = updated_actor.attribute
    if not attribute or "DemolishExtended" not in attribute:
        raise ValueError(f"Demo cannot occur without attribute {updated_actor.raw}")
    demolition = attribute["DemolishExtended"]
    attacker_car_actor_id = demolition["attacker"]["actor"]
    victim_car_actor_id = demolition["victim"]["actor"]
    if (
        attacker_car_actor_id not in frame.game.cars
        or victim_car_actor_id not in frame.game.cars
    ):
        frame.add_updated_actor_to_disconnected_car_component_updates(updated_actor)
        return None
    attacker_car = frame.game.cars[attacker_car_actor_id]
    victim_car = frame.game.cars[victim_car_actor_id]
    if not attacker_car:
        raise ValueError(f"No matching attacker car in demo actor {updated_actor.raw}")
    if not victim_car:
        raise ValueError(f"No matching victim car in demo actor {updated_actor.raw}")
    if processor.include_player_demos:
        field_label = f"{attacker_car.player.name}_demo"
        frame.processed_fields[field_label + "_x"] = demolition["attacker_velocity"][
            "x"
        ]
        frame.processed_fields[field_label + "_y"] = demolition["attacker_velocity"][
            "y"
        ]
        frame.processed_fields[field_label + "_z"] = demolition["attacker_velocity"][
            "z"
        ]
        field_label = f"{victim_car.player.name}_demoed"
        frame.processed_fields[field_label + "_x"] = demolition["victim_velocity"]["x"]
        frame.processed_fields[field_label + "_y"] = demolition["victim_velocity"]["y"]
        frame.processed_fields[field_label + "_z"] = demolition["victim_velocity"]["z"]
        frame.game.stop_tracking_position_for(victim_car)
    return None
