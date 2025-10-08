from typing import Dict, Any
from rocketleague_ml.models.rrrocket_json_preprocessor import RRRocket_JSON_Preprocessor
from rocketleague_ml.models.game_data_wrangler import Game_Data_Wrangler
from rocketleague_ml.models.frame_by_frame_processor import Frame_By_Frame_Processor
from rocketleague_ml.models.feature_extractor import Rocket_League_Feature_Extractor
from rocketleague_ml.models.playstyle_model import Playstyle_Model
from rocketleague_ml.models.playstyle_consistency_model import (
    Playstyle_Consistency_Model,
)
from rocketleague_ml.utils.logging import Logger


class Rocket_League_Pipeline:
    def __init__(self, goal: str = "find_playstyle"):
        self.logger = Logger(active=True)
        self.preprocessor = RRRocket_JSON_Preprocessor(logger=self.logger)
        self.wrangler = Game_Data_Wrangler(logger=self.logger)
        self.processor = Frame_By_Frame_Processor(
            preprocessor=self.preprocessor,
            logger=self.logger,
        )
        self.extractor = Rocket_League_Feature_Extractor(
            processor=self.processor, logger=self.logger
        )
        goals: Dict[str, Any] = {
            "find_playstyle": Playstyle_Model,
            "find_playstyle_consistency": Playstyle_Consistency_Model,
        }
        if goal not in goals:
            raise ValueError(
                f"Unknown goal provided {goal}. Options: {','.join(goals.keys())}"
            )
        self.model: Playstyle_Model | Playstyle_Consistency_Model = goals[goal](
            extractor=self.extractor, logger=self.logger
        )

        return None
