from rocketleague_ml.core.game import Game
from rocketleague_ml.types.attributes import Raw_Game_Data


def process_game(game_data: Raw_Game_Data):
    game = Game(game_data)
    for frame in game.frames:
        game.analyze_frame(frame)
    return game
