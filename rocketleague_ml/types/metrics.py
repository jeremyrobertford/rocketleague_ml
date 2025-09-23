from typing import TypedDict, Dict, Any


class BaseGameStats(TypedDict):
    # Scoreboard
    score: int
    goals: int
    assists: int
    shots: int
    saves: int
    goals_conceded: int

    # Stats
    shooting_percentage: float
    # custom_shooting_percentage: float
    save_percentage: float
    # custom_save_percentage: float

    # Physicality
    bumps_inflicted: int
    demos_inflicted: int
    bumps_taken: int
    demos_taken: int

    # Ball
    percentage_in_possession: float
    percentage_while_pressuring: float
    avg_distance_from_ball: float
    percentage_closest_to_ball: float
    percentage_farthest_from_ball: float
    percentage_in_front_of_ball: float
    percentage_behind_ball: float
    max_ball_hit_speed: int
    average_ball_hit_speed: int
    percentage_slow_ball_hit: int
    percentage_medium_ball_hit: int
    percentage_fast_ball_hit: int

    # Boost
    boost_per_minute: float
    percentage_while_boosting: float
    average_boost: float
    boost_collected: int
    boost_used: int
    boost_stolen: int
    # custom_boost_stolen: int
    big_pads_collected: int
    small_pads_collected: int
    big_pads_stolen: int
    # custom_big_pads_stolen: int
    small_pads_stolen: int
    # custom_small_pads_stolen: int
    average_seconds_on_0_boost: float
    average_seconds_on_full_boost: float
    overfill: int
    avg_overfill: int
    stolen_overfill: int
    # custom_stolen_overfill: int
    average_stolen_overfill: int
    # custom_average_stolen_overfill: int
    percentage_with_no_boost: float
    percentage_with_first_quarter_of_boost: float
    percentage_with_second_quarter_of_boost: float
    percentage_with_third_quarter_of_boost: float
    percentage_with_fourth_quarter_of_boost: float
    percentage_with_full_boost: float
    amount_used_while_supersonic: float
    average_amount_used_while_supersonic: float

    # Positioning
    average_distance_to_teammates: float
    average_distance_to_front_man: float
    average_distance_to_middle_man: float
    average_distance_to_back_man: float
    average_distance_to_first_man: float
    average_distance_to_second_man: float
    average_distance_to_third_man: float
    percentage_as_front_man: float
    percentage_as_middle_man: float
    percentage_as_back_man: float
    percentage_as_first_man: float
    percentage_as_second_man: float
    percentage_as_third_man: float
    percentage_in_offensive_third: float
    percentage_in_neutral_third: float
    percentage_in_defensive_third: float
    percentage_in_offensive_half: float
    percentage_in_defensive_half: float
    rotations: int
    average_seconds_before_rotating: int

    # Speed
    percentage_while_supersonic: float
    percentage_while_boost_speed: float
    percentage_while_slow: float
    average_speed: float
    average_seconds_once_supersonic: float
    average_seconds_once_slow: float
    percentage_supersonic_with_no_boost: float
    percentage_supersonic_with_first_quarter_of_boost: float
    percentage_supersonic_with_second_quarter_of_boost: float
    percentage_supersonic_with_third_quarter_of_boost: float
    percentage_supersonic_with_fourth_quarter_of_boost: float
    percentage_supersonic_with_full_boost: float
    percentage_slow_with_no_boost: float
    percentage_slow_with_first_quarter_of_boost: float
    percentage_slow_with_second_quarter_of_boost: float
    percentage_slow_with_third_quarter_of_boost: float
    percentage_slow_with_fourth_quarter_of_boost: float
    percentage_slow_with_full_boost: float
    percentage_reversing: float
    sharp_turns: int
    full_turns: int

    # Gameplay
    single_jumps: int
    double_jumps: int
    dodges: int
    front_flips: int
    side_flips: int
    back_flips: int
    half_flips: int
    speed_flips: int
    percentage_grounded: float
    percentage_on_wall: float
    percentage_on_left_wall: float
    percentage_on_right_wall: float
    percentage_on_front_wall: float
    percentage_on_back_wall: float
    percentage_on_ceiling: float
    percentage_on_offensive_ceiling: float
    percentage_on_defensive_ceiling: float
    percentage_in_air: float
    average_seconds_in_air_once_airborne: float
    percentage_in_air_under_crossbar: float
    percentage_in_air_over_crossbar: float
    fifty_fifties: int

    # Kickoffs
    kickoffs: int
    average_seconds_to_kickoff_ball: float
    average_speed_of_kickoff_ball: float
    # map size dependent on config.map_simplification_areas
    kickoff_ball_spillout_map: dict[int, int]


class AugmentedGameStats(TypedDict):
    percentage_with_augment: int
    average_seconds_with_augment: int

    # Collect all the stats for each of these modifiers
    as_first_man: BaseGameStats
    as_second_man: BaseGameStats
    as_third_man: BaseGameStats
    as_front_man: BaseGameStats
    as_middle_man: BaseGameStats
    as_back_man: BaseGameStats
    in_offensive_third: BaseGameStats
    in_neutral_third: BaseGameStats
    in_defensive_third: BaseGameStats
    in_offensive_half: BaseGameStats
    in_defensive_half: BaseGameStats


class GameStats(TypedDict):
    player_name: str

    boost_pick_up_map: dict[int, int]
    # map size dependent on config.map_simplification_areas
    possession_pick_up_map: dict[int, int]
    positioning_map: dict[int, int]

    general: AugmentedGameStats
    with_possession: AugmentedGameStats
    without_possession: AugmentedGameStats


class TaggedStats(TypedDict):
    instances: int
    total_seconds: float
    deactivating_frame: Dict[str, Any] | None
    active: bool


class TaggedStatsWithMetrics(TaggedStats):
    metrics: Dict[str, Any]


class TrackerDict(TypedDict):
    metrics: Dict[str, Any]
    tags: Dict[str, TaggedStats]
