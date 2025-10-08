from rocketleague_ml.core.actor import Actor
from rocketleague_ml.core.rigid_body import Rigid_Body
from rocketleague_ml.types.attributes import Raw_Frame


class Ball(Rigid_Body):
    def __init__(self, ball: Actor):
        super().__init__(ball, "ball")

    def team_hit(self, team_hit_actor: Actor, frame: Raw_Frame, round: int):
        return
