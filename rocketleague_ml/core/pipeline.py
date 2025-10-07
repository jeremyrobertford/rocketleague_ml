from rocketleague_ml.core.rrrocket_json_preprocessor import RRRocket_JSON_Preprocessor
from rocketleague_ml.core.game_data_wrangler import Game_Data_Wrangler
from rocketleague_ml.core.frame_by_frame_processor import Frame_By_Frame_Processor
from rocketleague_ml.core.feature_extractor import Rocket_League_Feature_Extractor
from rocketleague_ml.utils.logging import Logger


class Rocket_League_Pipeline:
    def __init__(self):
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
        return None
