from __future__ import annotations
from typing import TYPE_CHECKING
from rocketleague_ml.core.actor import Actor
from rocketleague_ml.core.frame import Frame
from rocketleague_ml.utils.helpers import parse_boost_actor_name

if TYPE_CHECKING:
    from rocketleague_ml.models.frame_by_frame_processor import Frame_By_Frame_Processor


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
        field_label = f"{victim_car.player.name}_demod"
        frame.processed_fields[field_label + "_x"] = demolition["victim_velocity"]["x"]
        frame.processed_fields[field_label + "_y"] = demolition["victim_velocity"]["y"]
        frame.processed_fields[field_label + "_z"] = demolition["victim_velocity"]["z"]
    return None


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
    car = frame.game.cars[boost_pickup["instigator"]]
    if processor.include_boost_management:
        field_label = f"{car.player.name}_boost_pickup"
        frame.processed_fields[field_label + "_x"] = boost_data[0]
        frame.processed_fields[field_label + "_y"] = boost_data[1]
        frame.processed_fields[field_label + "_z"] = boost_data[2]
        frame.processed_fields[field_label + "_amount"] = boost_data[3]
    return None


def process_team_ball_hit(
    processor: Frame_By_Frame_Processor, updated_actor: Actor, frame: Frame
):
    attribute = updated_actor.attribute
    if not attribute or "Byte" not in attribute:
        raise ValueError(f"Team ball hit cannot occur without team {updated_actor.raw}")
    if processor.include_possession:
        field_label = f"{"orange" if attribute["Byte"] else "blue"}_team_ball_hit"
        frame.processed_fields[field_label] = 1
    return None
