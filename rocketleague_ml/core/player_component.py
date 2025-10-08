from __future__ import annotations
from rocketleague_ml.utils.helpers import convert_byte_to_float
from rocketleague_ml.core.actor import Actor


class Player_Component(Actor):
    def __init__(self, player_component: Actor):
        super().__init__(player_component.raw, player_component.objects)
        attribute = player_component.attribute
        if not attribute:
            raise ValueError(
                f"Player component without attribute {player_component.raw}"
            )

        self.amount = 0
        self.active = False

        if "Int" in attribute:
            self.amount = attribute["Int"]
            return None
        if "Float" in attribute:
            self.amount = attribute["Float"]
            return None
        if "Byte" in attribute:
            self.amount = convert_byte_to_float(attribute["Byte"])
            self.active = attribute["Byte"] == 3
            return None

        active: bool | None = attribute.get("Boolean")
        if active is not None:
            self.active = active
            return None

        raise ValueError(f"Player component failed to init {player_component.raw}")
