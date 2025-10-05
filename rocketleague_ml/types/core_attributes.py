from typing import List, TypedDict
from rocketleague_ml.core.car import Car, Car_Component
from rocketleague_ml.core.actor import Actor


class Categorized_New_Actors(TypedDict):
    cars: List[Car]
    car_components: List[Car_Component]


class Categorized_Updated_Actors(TypedDict):
    events: List[Actor]
    others: List[Actor]
    cars_for_players: List[Actor]
