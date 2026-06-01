from __future__ import annotations

import pandas as pd

from .elo import BASE_ELO


def _safe_int(value, default: int = 0) -> int:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null", "nat"}:
        return default
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return default


def build_elo_timeline_df(matches_view: pd.DataFrame, elo_history_df: pd.DataFrame, players_lookup: dict[str, str]) -> pd.DataFrame:
    if elo_history_df.empty:
        return pd.DataFrame()
    history = elo_history_df.copy()
    history["player_id"] = history["player_id"].fillna("").astype(str)
    history["match_id"] = history["match_id"].fillna("").astype(str)
    history["recorded_at_dt"] = pd.to_datetime(history["recorded_at"], errors="coerce")
    for col in ["old_elo", "new_elo", "delta"]:
        history[col] = pd.to_numeric(history[col], errors="coerce").fillna(0)
    history["player"] = history["player_id"].map(players_lookup).fillna(history["player_id"])

    match_cols = [
        "match_id", "match_label", "scheduled_date", "scheduled_time", "team_a_names", "team_b_names",
        "team_a_score", "team_b_score", "winner_label", "status", "video_url", "notes"
    ]
    if not matches_view.empty:
        available = [c for c in match_cols if c in matches_view.columns]
        history = history.merge(matches_view[available], on="match_id", how="left")
    for col in match_cols:
        if col not in history.columns:
            history[col] = ""

    history["match_date_dt"] = pd.to_datetime(history["scheduled_date"], errors="coerce")
    history["match_date_dt"] = history["match_date_dt"].fillna(history["recorded_at_dt"].dt.normalize())
    history["match_date_label"] = history["match_date_dt"].dt.strftime("%Y-%m-%d").fillna(history["recorded_at"].astype(str).str[:10]).fillna("")
    history["score_label"] = history.apply(
        lambda r: f"{_safe_int(r.get('team_a_score'))} - {_safe_int(r.get('team_b_score'))}",
        axis=1,
    )

    # Chart-specific x position: use the match date, but spread multiple matches
    # on the same day horizontally so imported batches are readable.
    match_order = history[["match_id", "match_date_dt", "recorded_at_dt", "recorded_at"]].drop_duplicates("match_id").copy()
    match_order = match_order.sort_values(["match_date_dt", "recorded_at_dt", "recorded_at", "match_id"])
    match_order["date_group"] = match_order["match_date_dt"].dt.date.astype(str)
    match_order["day_order"] = match_order.groupby("date_group").cumcount()
    match_order["day_count"] = match_order.groupby("date_group")["match_id"].transform("count")
    # Spread between 02:00 and 22:00 within the same displayed date.
    match_order["spread_minutes"] = 120 + ((match_order["day_order"] + 1) * (20 * 60 / (match_order["day_count"] + 1)))
    match_order["chart_x_dt"] = match_order["match_date_dt"].dt.normalize() + pd.to_timedelta(match_order["spread_minutes"], unit="m")
    history = history.merge(match_order[["match_id", "chart_x_dt"]], on="match_id", how="left")
    history["delta_label"] = history["delta"].apply(lambda x: f"{float(x):+.0f}")
    history["hover_details"] = history.apply(
        lambda r: (
            f"<b>{r.get('player', '')}</b><br>"
            f"Elo: {float(r.get('old_elo', BASE_ELO)):.0f} → {float(r.get('new_elo', BASE_ELO)):.0f} ({r.get('delta_label', '')})<br>"
            f"Match: {r.get('team_a_names', '')} vs {r.get('team_b_names', '')}<br>"
            f"Score: {r.get('score_label', '')}<br>"
            f"Winner: {r.get('winner_label', '') or '—'}<br>"
            f"Date: {r.get('match_date_label', '')}"
        ),
        axis=1,
    )
    return history.sort_values(["chart_x_dt", "recorded_at_dt", "recorded_at", "match_id", "player"])


def build_partner_matrix_df(matches_df: pd.DataFrame, participants_df: pd.DataFrame, elo_history_df: pd.DataFrame, players_lookup: dict[str, str]) -> pd.DataFrame:
    if matches_df.empty or participants_df.empty:
        return pd.DataFrame()
    matches = matches_df.copy()
    participants = participants_df.copy()
    history = elo_history_df.copy()
    matches["match_id"] = matches["match_id"].astype(str)
    participants["match_id"] = participants["match_id"].astype(str)
    participants["player_id"] = participants["player_id"].astype(str)
    participants["team"] = participants["team"].astype(str)
    if not history.empty:
        history["match_id"] = history["match_id"].astype(str)
        history["player_id"] = history["player_id"].astype(str)
        history["delta"] = pd.to_numeric(history["delta"], errors="coerce").fillna(0)

    completed = matches[matches["status"].astype(str).str.lower().eq("completed") & matches["winner"].astype(str).isin(["A", "B"])].copy()
    rows: list[dict] = []
    for _, match in completed.iterrows():
        match_id = str(match["match_id"])
        p_match = participants[participants["match_id"] == match_id]
        if p_match.empty:
            continue
        h_match = history[history["match_id"] == match_id] if not history.empty else pd.DataFrame()
        for team, team_rows in p_match.groupby("team"):
            player_ids = team_rows["player_id"].astype(str).tolist()
            if len(player_ids) < 2:
                continue
            won = str(match.get("winner", "")) == str(team)
            for pid in player_ids:
                delta_rows = h_match[h_match["player_id"] == pid] if not h_match.empty else pd.DataFrame()
                delta = float(delta_rows["delta"].sum()) if not delta_rows.empty else 0.0
                for partner_id in player_ids:
                    if partner_id == pid:
                        continue
                    rows.append({
                        "player_id": pid,
                        "player": players_lookup.get(pid, pid),
                        "partner_id": partner_id,
                        "partner": players_lookup.get(partner_id, partner_id),
                        "match_id": match_id,
                        "won": won,
                        "delta": delta,
                        "gained": max(delta, 0.0),
                        "lost": min(delta, 0.0),
                    })
    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(rows)
    out = raw.groupby(["player_id", "player", "partner_id", "partner"], as_index=False).agg(
        matches=("match_id", "nunique"),
        wins=("won", "sum"),
        net_elo=("delta", "sum"),
        elo_gained=("gained", "sum"),
        elo_lost=("lost", "sum"),
    )
    out["losses"] = out["matches"] - out["wins"]
    out["win_rate"] = (out["wins"] / out["matches"] * 100).round(1)
    out["elo_lost_abs"] = out["elo_lost"].abs()
    out["hover"] = out.apply(
        lambda r: (
            f"<b>{r['player']} + {r['partner']}</b><br>"
            f"Matches: {int(r['matches'])}<br>"
            f"Win rate: {float(r['win_rate']):.1f}%<br>"
            f"Wins/Losses: {int(r['wins'])}/{int(r['losses'])}<br>"
            f"Net Elo for {r['player']}: {float(r['net_elo']):+.0f}<br>"
            f"Elo gained: +{float(r['elo_gained']):.0f}<br>"
            f"Elo lost: -{float(r['elo_lost_abs']):.0f}"
        ),
        axis=1,
    )
    return out


def build_player_relationship_insights(selected_pid: str, matches_df: pd.DataFrame, participants_df: pd.DataFrame, elo_history_df: pd.DataFrame, players_lookup: dict[str, str]) -> dict[str, object]:
    insights: dict[str, object] = {"worst_opponent": None, "best_partner": None, "best_partner_win_rate": None}
    if not selected_pid or matches_df.empty or participants_df.empty:
        return insights

    matches = matches_df.copy()
    participants = participants_df.copy()
    history = elo_history_df.copy()
    matches["match_id"] = matches["match_id"].astype(str)
    participants["match_id"] = participants["match_id"].astype(str)
    participants["player_id"] = participants["player_id"].astype(str)
    participants["team"] = participants["team"].astype(str)
    if not history.empty:
        history["match_id"] = history["match_id"].astype(str)
        history["player_id"] = history["player_id"].astype(str)
        history["delta"] = pd.to_numeric(history["delta"], errors="coerce").fillna(0)

    partner_rows = []
    opponent_rows = []
    player_matches = participants[participants["player_id"] == str(selected_pid)]
    for _, p_row in player_matches.iterrows():
        match_id = str(p_row["match_id"])
        team = str(p_row["team"])
        match_rows = matches[matches["match_id"] == match_id]
        if match_rows.empty:
            continue
        match = match_rows.iloc[0]
        if str(match.get("status", "")).lower() != "completed":
            continue
        h_rows = history[(history["match_id"] == match_id) & (history["player_id"] == str(selected_pid))] if not history.empty else pd.DataFrame()
        delta = float(h_rows["delta"].sum()) if not h_rows.empty else 0.0
        won = str(match.get("winner", "")) == team
        participants_this_match = participants[participants["match_id"] == match_id]
        for _, other in participants_this_match.iterrows():
            other_pid = str(other["player_id"])
            if other_pid == str(selected_pid):
                continue
            other_team = str(other["team"])
            row = {"id": other_pid, "name": players_lookup.get(other_pid, other_pid), "delta": delta, "won": won, "match_id": match_id}
            if other_team == team:
                partner_rows.append(row)
            else:
                opponent_rows.append(row)

    if opponent_rows:
        opp = pd.DataFrame(opponent_rows)
        opp_loss = opp.groupby(["id", "name"], as_index=False).agg(
            net_elo=("delta", "sum"),
            elo_lost=("delta", lambda s: float(s[s < 0].sum())),
            matches=("match_id", "nunique"),
        )
        opp_loss["elo_lost_abs"] = opp_loss["elo_lost"].abs()
        candidates = opp_loss[opp_loss["elo_lost_abs"] > 0].sort_values(["elo_lost_abs", "matches"], ascending=[False, False])
        if not candidates.empty:
            insights["worst_opponent"] = candidates.iloc[0].to_dict()

    if partner_rows:
        partner = pd.DataFrame(partner_rows)
        partner_summary = partner.groupby(["id", "name"], as_index=False).agg(
            matches=("match_id", "nunique"),
            wins=("won", "sum"),
            net_elo=("delta", "sum"),
        )
        partner_summary["win_rate"] = (partner_summary["wins"] / partner_summary["matches"] * 100).round(1)
        positive = partner_summary[partner_summary["net_elo"] > 0].sort_values(["net_elo", "win_rate", "matches"], ascending=[False, False, False])
        if not positive.empty:
            insights["best_partner"] = positive.iloc[0].to_dict()
        best_wr = partner_summary.sort_values(["win_rate", "matches", "net_elo"], ascending=[False, False, False])
        if not best_wr.empty:
            insights["best_partner_win_rate"] = best_wr.iloc[0].to_dict()

    return insights
