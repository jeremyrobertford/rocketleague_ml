from rocketleague_ml.types.attributes import Raw_Game_Data
from rocketleague_ml.utils.logging import Logger


class Game_Data_Wrangler:
    def __init__(self, logger: Logger):
        self.logger = logger

    def clean(self, data: Raw_Game_Data):
        return data
