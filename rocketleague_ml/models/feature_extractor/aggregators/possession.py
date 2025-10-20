import pandas as pd
from typing import Any, Dict, List


def _get_valid_opponent_stints(
    game: pd.DataFrame, main_player: str, teams: Dict[str, List[str]]
):
    """
    Identifies valid opponent possession stints that end with the player taking or contesting possession.
    Skips stints ending due to rotation.
    """
    all_players = [p for team in teams.values() for p in team]
    team_name = next(t for t, players in teams.items() if main_player in players)
    opponent_players = [p for p in all_players if p not in teams[team_name]]

    active_opp_sets: List[frozenset[str]] = []
    for i in range(len(game)):
        active: set[str] = set()
        for opp in opponent_players:
            if (
                game.loc[i, f"{opp}_in_possession"]
                or game.loc[i, f"{opp}_in_contested_possession"]
            ):
                active.add(opp)
        active_opp_sets.append(frozenset(active))

    stint_ids = [0]
    for i in range(1, len(active_opp_sets)):
        if active_opp_sets[i] != active_opp_sets[i - 1]:
            stint_ids.append(stint_ids[-1] + 1)
        else:
            stint_ids.append(stint_ids[-1])

    game["_opp_stint_id"] = stint_ids
    game["_active_opps"] = active_opp_sets

    stints: List[Dict[str, Any]] = []
    for stint_id, stint_df in game.groupby("_opp_stint_id"):  # type: ignore
        if not stint_df["_active_opps"].iloc[0]:
            continue

        start_idx = stint_df.index[0]
        end_idx = stint_df.index[-1]
        duration = end_idx - start_idx
        active_opps = list(stint_df["_active_opps"].iloc[0])
        end_row = stint_df.iloc[-1]

        took = bool(end_row[f"{main_player}_in_possession"])
        contested = bool(end_row[f"{main_player}_in_contested_possession"])
        rotated = end_row[f"{main_player}_player_rotation_first_man"] == 0 and (
            end_row[f"{main_player}_player_rotation_second_man"] == 1
            or end_row[f"{main_player}_player_rotation_third_man"] == 1
        )

        if rotated:
            outcome = "rotated"
        elif took:
            outcome = "taken"
        elif contested:
            outcome = "contested"
        else:
            outcome = "other"

        stints.append(
            {
                "stint_id": stint_id,
                "start_frame": start_idx,
                "end_frame": end_idx,
                "duration": duration,
                "active_opponents": active_opps,
                "outcome": outcome,
            }
        )

    stint_df = pd.DataFrame(stints)
    valid_stints = stint_df[
        stint_df["outcome"].isin(["taken", "contested"])  # type: ignore
    ].reset_index(drop=True)
    return valid_stints


def aggregate_possession_sequences(
    features: Dict[str, float],
    game: pd.DataFrame,
    main_player: str,
    teams: Dict[str, List[str]],
) -> Dict[str, float]:
    """
    Computes stats for when the opponent starts a possession stint,
    and the main player ends it by taking or contesting possession.
    Excludes rotations and opponent-to-opponent passes.
    """
    valid_stints = _get_valid_opponent_stints(game, main_player, teams)
    if valid_stints.empty:
        return {
            "opp_possession_stint_count": 0,
            "opp_possession_total_time": 0.0,
            "opp_possession_avg_stint": 0.0,
        }

    # --- Aggregations ---
    total_time = valid_stints["duration"].sum()
    avg_stint = valid_stints["duration"].mean()
    stint_count = len(valid_stints)

    # Optional: separate contested vs taken breakdown
    taken_time = valid_stints.loc[valid_stints["outcome"] == "taken", "duration"].sum()
    contested_time = valid_stints.loc[
        valid_stints["outcome"] == "contested", "duration"
    ].sum()

    # Store in features
    features.update(
        {
            "opp_possession_stint_count": stint_count,
            "opp_possession_total_time": total_time,
            "opp_possession_avg_stint": avg_stint,
            "opp_possession_taken_time": taken_time,
            "opp_possession_contested_time": contested_time,
        }
    )

    return features


def aggregate_possession(
    game: pd.DataFrame,
    main_player: str,
    teams: Dict[str, List[str]],
) -> Dict[str, float]:
    features: Dict[str, float] = {}

    # Helper to compute percentage and stints
    def compute_stats(mask: pd.Series):
        time = (mask * game["delta"]).sum()

        # Runs / stints
        run_ids = mask.ne(mask.shift()).cumsum()
        stint_groups = game[mask].groupby(run_ids)  # type: ignore
        stint_durations = stint_groups["delta"].sum()
        avg_stint = stint_durations.mean() if not stint_durations.empty else 0

        return time, avg_stint

    def calculate_possession_fields(
        clean_series: pd.Series,
        contested_series: pd.Series,
        label: str = "",
    ):
        # ---- Player possession ----
        clean_or_contested_series = clean_series | contested_series

        # Player possession metrics
        clean, stint_clean = compute_stats(clean_series)
        contested, stint_contested = compute_stats(contested_series)
        clean_or_contested, stint_clean_or_contested = compute_stats(
            clean_or_contested_series
        )

        label = label + " " if label else ""
        features[f"Time with {label}Full Possession"] = clean
        features[f"Time with {label}Contested Possession"] = contested
        features[f"Time with {label}Possession"] = clean_or_contested
        features[f"Average Stint with {label}Full Possession"] = stint_clean
        features[f"Average Stint with {label}Contested Possession"] = stint_contested
        features[f"Average Stint with {label}Possession"] = stint_clean_or_contested
        return None

    # ---- Player possession ----
    player_clean = game[f"{main_player}_in_possession"].astype(bool)
    player_contested = game[f"{main_player}_in_contested_possession"].astype(bool)
    calculate_possession_fields(
        clean_series=player_clean,
        contested_series=player_contested,
    )

    # ---- Offensive pressure ----
    offensive_mask = game[f"{main_player}_offensive_half"].astype(bool)
    player_offense_clean = player_clean & offensive_mask
    player_offense_contested = player_contested & offensive_mask
    calculate_possession_fields(
        clean_series=player_offense_clean,
        contested_series=player_offense_contested,
        label="Offensive Pressure",
    )

    shallow_offensive_mask = game[f"{main_player}_offensive_half"].astype(bool) & game[
        f"{main_player}_neutral_third"
    ].astype(bool)
    player_shallow_offense_clean = player_clean & shallow_offensive_mask
    player_shallow_offense_contested = player_contested & shallow_offensive_mask
    calculate_possession_fields(
        clean_series=player_shallow_offense_clean,
        contested_series=player_shallow_offense_contested,
        label="Light Offensive Pressure",
    )

    deep_offensive_mask = game[f"{main_player}_offensive_third"].astype(bool)
    player_deep_offense_clean = player_clean & deep_offensive_mask
    player_deep_offense_contested = player_contested & deep_offensive_mask
    calculate_possession_fields(
        clean_series=player_deep_offense_clean,
        contested_series=player_deep_offense_contested,
        label="Heavy Offensive Pressure",
    )

    # ---- Defensive pressure (opponent in possession in defensive half) ----
    defensive_mask = game[f"{main_player}_defensive_half"].astype(bool)
    player_offense_clean = player_clean & defensive_mask
    player_offense_contested = player_contested & defensive_mask
    calculate_possession_fields(
        clean_series=player_offense_clean,
        contested_series=player_offense_contested,
        label="Defensive Pressure",
    )

    shallow_defensive_mask = game[f"{main_player}_defensive_half"].astype(bool) & game[
        f"{main_player}_neutral_third"
    ].astype(bool)
    player_shallow_offense_clean = player_clean & shallow_defensive_mask
    player_shallow_offense_contested = player_contested & shallow_defensive_mask
    calculate_possession_fields(
        clean_series=player_shallow_offense_clean,
        contested_series=player_shallow_offense_contested,
        label="Light Defensive Pressure",
    )

    deep_defensive_mask = game[f"{main_player}_defensive_third"].astype(bool)
    player_deep_offense_clean = player_clean & deep_defensive_mask
    player_deep_offense_contested = player_contested & deep_defensive_mask
    calculate_possession_fields(
        clean_series=player_deep_offense_clean,
        contested_series=player_deep_offense_contested,
        label="Heavy Defensive Pressure",
    )

    aggregate_possession_sequences(
        features=features, game=game, main_player=main_player, teams=teams
    )

    return features
