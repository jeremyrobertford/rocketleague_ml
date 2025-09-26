from rocketleague_ml.core.actor import Actor


class Ball(Actor):
    def __init__(self, ball: Actor):
        super().__init__(ball.raw, ball.objects)
        if not self.positioning:
            raise ValueError(f"Ball failed to position {ball.raw}")
