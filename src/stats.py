from __future__ import annotations

import pandas as pd

from .elo import BASE_ELO


def current_elo_map(players_df: pd.DataFrame, history_df: pd.DataFrame) -> dict[str, float]:
    elo_map = {str(pid): float(BASE_ELO) for pid in players_df["player_id"].astype(str).tolist()}
    if history_df.empty:
        return elo_map
    history_df = history_df.copy()
    history_df["player_id"] = history_df["player_id"].astype(str)
    history_df["recorded_at"] = pd.to_datetime(history_df["recorded_at"], errors="coerce")
    latest = history_df.sort_values("recorded_at").groupby("player_id", as_index=False).tail(1)
    for _, row in latest.iterrows():
        elo_map[str(row["player_id"])] = float(row["new_elo"])
    return elo_map


def build_player_stats(players_df: pd.DataFrame, matches_df: pd.DataFrame, events_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    if players_df.empty:
        return pd.DataFrame()

    elo_map = current_elo_map(players_df, history_df)
    stats_rows = []

    matches_df = matches_df.copy()
    events_df = events_df.copy()
    if not events_df.empty:
        events_df["player_id"] = events_df["player_id"].astype(str)
        events_df["points_awarded"] = pd.to_numeric(events_df["points_awarded"], errors="coerce").fillna(0)

    for _, player in players_df.iterrows():
        pid = str(player["player_id"])
        name = player["name"]

        player_matches = matches_df[
            matches_df["team_a_players"].fillna("").str.contains(pid)
            | matches_df["team_b_players"].fillna("").str.contains(pid)
        ].copy()

        wins = 0
        losses = 0
        for _, match in player_matches.iterrows():
            in_a = pid in str(match["team_a_players"]).split("|")
            in_b = pid in str(match["team_b_players"]).split("|")
            if match["status"] != "Completed":
                continue
            if (in_a and match["winner"] == "A") or (in_b and match["winner"] == "B"):
                wins += 1
            elif match["winner"] in {"A", "B"}:
                losses += 1

        player_events = events_df[events_df["player_id"] == pid] if not events_df.empty else pd.DataFrame()
        good_shots = int((player_events["event_type"] == "good_shot").sum()) if not player_events.empty else 0
        bad_shots = int((player_events["event_type"] == "bad_shot").sum()) if not player_events.empty else 0
        service_faults = int((player_events["event_type"] == "service_fault").sum()) if not player_events.empty else 0
        highlights = int((player_events["event_type"] == "highlight").sum()) if not player_events.empty else 0
        points_won = int(player_events["points_awarded"].sum()) if not player_events.empty else 0
        matches_played = wins + losses
        win_rate = round((wins / matches_played) * 100, 1) if matches_played else 0.0
        shot_balance = good_shots - bad_shots

        stats_rows.append(
            {
                "player_id": pid,
                "name": name,
                "elo": round(float(elo_map.get(pid, BASE_ELO)), 2),
                "matches_played": matches_played,
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate,
                "points_won": points_won,
                "good_shots": good_shots,
                "bad_shots": bad_shots,
                "service_faults": service_faults,
                "highlights": highlights,
                "shot_balance": shot_balance,
            }
        )

    return pd.DataFrame(stats_rows).sort_values(["elo", "win_rate", "wins"], ascending=[False, False, False])
