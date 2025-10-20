from __future__ import annotations
from typing import TYPE_CHECKING
from rocketleague_ml.core.actor import Actor
from rocketleague_ml.core.frame import Frame
from rocketleague_ml.utils.helpers import parse_boost_actor_name

if TYPE_CHECKING:
    from rocketleague_ml.models.frame_by_frame_processor import Frame_By_Frame_Processor


def process_boost_pickup(
    processor: Frame_By_Frame_Processor, updated_actor: Actor, frame: Frame
):
    attribute = updated_actor.attribute
    if not attribute or "PickupNew" not in attribute:
        raise ValueError(
            f"Boost pickup cannot occur without attribute {updated_actor.raw}"
        )
    boost_pickup = attribute["PickupNew"]
    boost_actor = frame.game.boost_pads[updated_actor.actor_id]
    if not boost_actor:
        raise ValueError(f"Cannot find boost actor {updated_actor.actor_id}")
    boost_data = parse_boost_actor_name(boost_actor.object)
    if not boost_data:
        # print(
        #     f"Cannot find boost {updated_actor.actor_id}, {boost_actor.object}"
        # )
        # raise ValueError(
        #     f"Boost data not found for boost actor {boost_actor.object}"
        # )
        return None
    if boost_pickup["instigator"] is None or boost_pickup["instigator"] == -1:
        # print(f"Null boost pickup {updated_actor.raw}")
        return None
    if boost_pickup["instigator"] in frame.game.disconnected_cars:
        frame.add_updated_actor_to_disconnected_car_component_updates(updated_actor)
        return None
    car = frame.game.cars[boost_pickup["instigator"]]
    if processor.include_boost_management:
        field_label = f"{car.player.name}_boost_pickup"
        frame.processed_fields[field_label + "_x"] = boost_data[0]
        frame.processed_fields[field_label + "_y"] = boost_data[1]
        frame.processed_fields[field_label + "_z"] = boost_data[2]
        frame.processed_fields[field_label + "_amount"] = boost_data[3]
    return None
