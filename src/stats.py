from __future__ import annotations

import pandas as pd

from .elo import BASE_ELO


def _coalesced_text_column(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=str)
    out = pd.Series([""] * len(df), index=df.index, dtype=object)
    for col in columns:
        if col not in df.columns:
            continue
        values = df[col].fillna("").astype(str).str.strip()
        out = out.where(out.astype(str).str.strip() != "", values)
    return out.astype(str)


def _match_datetime_from_columns(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="datetime64[ns]")
    date_part = _coalesced_text_column(df, ["scheduled_date", "match_date"])
    time_part = _coalesced_text_column(df, ["scheduled_time", "match_time"])
    combined = (date_part + " " + time_part).str.strip()
    combined = combined.where(date_part.str.strip() != "", "")
    return pd.to_datetime(combined, errors="coerce")


def current_elo_map(players_df: pd.DataFrame, history_df: pd.DataFrame, matches_df: pd.DataFrame | None = None) -> dict[str, float]:
    """Return each player's current Elo using the same match-ordering idea as Elo rebuild.

    Important: Supabase/API result order is not guaranteed, so do not rely on the
    last row returned from elo_history. If matches_df is provided, sort history by:
    1) match created_at, 2) match date/time, 3) completed_at, 4) recorded_at, 5) match_id.
    """
    elo_map = {str(pid): float(BASE_ELO) for pid in players_df["player_id"].astype(str).tolist()}
    if history_df.empty:
        return elo_map

    history = history_df.copy()
    history["player_id"] = history["player_id"].astype(str)
    history["match_id"] = history["match_id"].fillna("").astype(str)
    history["new_elo"] = pd.to_numeric(history["new_elo"], errors="coerce")
    history["_recorded_sort_key"] = pd.to_datetime(history.get("recorded_at", ""), errors="coerce")

    if matches_df is not None and not matches_df.empty:
        matches = matches_df.copy()
        matches["match_id"] = matches["match_id"].fillna("").astype(str)
        keep_cols = [
            col for col in [
                "match_id", "created_at", "scheduled_date", "scheduled_time",
                "match_date", "match_time", "completed_at"
            ]
            if col in matches.columns
        ]
        matches = matches[keep_cols].drop_duplicates("match_id", keep="last")
        history = history.merge(matches, on="match_id", how="left", suffixes=("", "_match"))
        history["_created_sort_key"] = pd.to_datetime(history.get("created_at", ""), errors="coerce")
        history["_match_datetime_sort_key"] = _match_datetime_from_columns(history)
        history["_completed_sort_key"] = pd.to_datetime(history.get("completed_at", ""), errors="coerce")
        sort_cols = ["_created_sort_key", "_match_datetime_sort_key", "_completed_sort_key", "_recorded_sort_key", "match_id"]
    else:
        sort_cols = ["_recorded_sort_key", "match_id"]

    latest = history.sort_values(sort_cols, na_position="last").groupby("player_id", as_index=False).tail(1)
    for _, row in latest.iterrows():
        if pd.isna(row.get("new_elo")):
            continue
        elo_map[str(row["player_id"])] = float(row["new_elo"])
    return elo_map


def build_player_stats(
    players_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    events_df: pd.DataFrame,
    history_df: pd.DataFrame,
    participants_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if players_df.empty:
        return pd.DataFrame()

    elo_map = current_elo_map(players_df, history_df, matches_df)
    stats_rows = []

    matches_df = matches_df.copy()
    events_df = events_df.copy()
    participants_df = participants_df.copy() if participants_df is not None else pd.DataFrame(columns=["match_id", "player_id", "team", "slot"])
    if not events_df.empty:
        events_df["player_id"] = events_df["player_id"].fillna("").astype(str)
        events_df["points_awarded"] = pd.to_numeric(events_df["points_awarded"], errors="coerce").fillna(0)
    if not participants_df.empty:
        participants_df["player_id"] = participants_df["player_id"].astype(str)
        participants_df["match_id"] = participants_df["match_id"].astype(str)

    for _, player in players_df.iterrows():
        pid = str(player["player_id"])
        name = player["name"]

        if not participants_df.empty:
            match_ids = participants_df.loc[participants_df["player_id"] == pid, "match_id"].astype(str).unique().tolist()
            player_matches = matches_df[matches_df["match_id"].astype(str).isin(match_ids)].copy()
        else:
            player_matches = matches_df[
                matches_df["team_a_players"].fillna("").str.contains(pid)
                | matches_df["team_b_players"].fillna("").str.contains(pid)
            ].copy()

        wins = 0
        losses = 0
        for _, match in player_matches.iterrows():
            if match["status"] != "Completed":
                continue

            if not participants_df.empty:
                p_rows = participants_df[(participants_df["match_id"].astype(str) == str(match["match_id"])) & (participants_df["player_id"] == pid)]
                if p_rows.empty:
                    continue
                team = str(p_rows.iloc[0]["team"])
                if (team == "A" and match["winner"] == "A") or (team == "B" and match["winner"] == "B"):
                    wins += 1
                elif match["winner"] in {"A", "B"}:
                    losses += 1
            else:
                in_a = pid in str(match["team_a_players"]).split("|")
                in_b = pid in str(match["team_b_players"]).split("|")
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
