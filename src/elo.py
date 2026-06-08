from __future__ import annotations

from typing import Iterable

BASE_ELO = 1000
K_FACTOR = 16
ELO_MODEL_VERSION = "lifetime_team_average_v2"
SEASON_K_FACTOR = 32
SEASONAL_ELO_MODEL_VERSION = "season_team_average_v1"


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def team_rating(player_ratings: Iterable[float]) -> float:
    ratings = list(player_ratings)
    return sum(ratings) / len(ratings)


def update_team_elos(team_a_ratings: list[float], team_b_ratings: list[float], winner: str, k_factor: float = K_FACTOR) -> tuple[list[float], list[float]]:
    team_a_avg = team_rating(team_a_ratings)
    team_b_avg = team_rating(team_b_ratings)

    exp_a = expected_score(team_a_avg, team_b_avg)
    exp_b = expected_score(team_b_avg, team_a_avg)

    actual_a = 1.0 if winner == "A" else 0.0
    actual_b = 1.0 if winner == "B" else 0.0

    delta_a = k_factor * (actual_a - exp_a)
    delta_b = k_factor * (actual_b - exp_b)

    per_player_a = delta_a / len(team_a_ratings)
    per_player_b = delta_b / len(team_b_ratings)

    new_a = [round(r + per_player_a, 2) for r in team_a_ratings]
    new_b = [round(r + per_player_b, 2) for r in team_b_ratings]
    return new_a, new_b
