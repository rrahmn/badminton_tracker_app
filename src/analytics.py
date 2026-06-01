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
    out["avg_net_elo"] = (out["net_elo"] / out["matches"]).round(2)
    out["avg_elo_gained"] = (out["elo_gained"] / out["matches"]).round(2)
    out["avg_elo_lost"] = (out["elo_lost"].abs() / out["matches"]).round(2)
    out["elo_lost_abs"] = out["elo_lost"].abs()
    out["hover"] = out.apply(
        lambda r: (
            f"<b>{r['player']} + {r['partner']}</b><br>"
            f"Matches: {int(r['matches'])}<br>"
            f"Win rate: {float(r['win_rate']):.1f}%<br>"
            f"Wins/Losses: {int(r['wins'])}/{int(r['losses'])}<br>"
            f"Avg Elo / match for {r['player']}: {float(r['avg_net_elo']):+.1f}<br>"
            f"Net Elo: {float(r['net_elo']):+.0f}<br>"
            f"Elo gained: +{float(r['elo_gained']):.0f} total / +{float(r['avg_elo_gained']):.1f} avg<br>"
            f"Elo lost: -{float(r['elo_lost_abs']):.0f} total / -{float(r['avg_elo_lost']):.1f} avg"
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


def _expected_win_rate(own_elo: float, opponent_elo: float) -> float:
    return 1 / (1 + 10 ** ((opponent_elo - own_elo) / 400))


def _safe_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _match_old_elo_map(match_id: str, history: pd.DataFrame) -> dict[str, float]:
    if history.empty:
        return {}
    rows = history[history["match_id"].astype(str) == str(match_id)].copy()
    if rows.empty:
        return {}
    rows["old_elo"] = pd.to_numeric(rows["old_elo"], errors="coerce").fillna(BASE_ELO)
    return {str(row["player_id"]): float(row["old_elo"]) for _, row in rows.iterrows()}


def build_player_loss_risk_df(
    selected_pid: str,
    matches_df: pd.DataFrame,
    participants_df: pd.DataFrame,
    elo_history_df: pd.DataFrame,
    players_lookup: dict[str, str],
) -> pd.DataFrame:
    """Doubles-only player context chart data.

    x = partner Elo - selected player's Elo at the match.
    y = opponent team Elo - selected player's team Elo at the match.
    z = actual loss rate - expected loss rate, in percentage points.
    """
    if not selected_pid or matches_df.empty or participants_df.empty:
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
        p_match = participants[participants["match_id"] == match_id].copy()
        selected_rows = p_match[p_match["player_id"] == str(selected_pid)]
        if selected_rows.empty:
            continue
        selected_team = str(selected_rows.iloc[0]["team"])
        team_ids = p_match[p_match["team"] == selected_team]["player_id"].astype(str).tolist()
        opponent_ids = p_match[p_match["team"] != selected_team]["player_id"].astype(str).tolist()
        partner_ids = [pid for pid in team_ids if pid != str(selected_pid)]
        # This chart is specifically about doubles partner context.
        if not partner_ids or not opponent_ids:
            continue

        old_elos = _match_old_elo_map(match_id, history)
        player_elo = old_elos.get(str(selected_pid), BASE_ELO)
        partner_elos = [old_elos.get(pid, BASE_ELO) for pid in partner_ids]
        team_elos = [old_elos.get(pid, BASE_ELO) for pid in team_ids]
        opponent_elos = [old_elos.get(pid, BASE_ELO) for pid in opponent_ids]
        partner_elo = sum(partner_elos) / len(partner_elos)
        own_team_elo = sum(team_elos) / len(team_elos)
        opponent_team_elo = sum(opponent_elos) / len(opponent_elos)

        expected_win = _expected_win_rate(own_team_elo, opponent_team_elo)
        expected_loss = 1 - expected_win
        actual_loss = 0 if str(match.get("winner", "")) == selected_team else 1
        loss_risk_pp = (actual_loss - expected_loss) * 100

        team_a_score = _safe_int(match.get("team_a_score"))
        team_b_score = _safe_int(match.get("team_b_score"))
        selected_score = team_a_score if selected_team == "A" else team_b_score
        opponent_score = team_b_score if selected_team == "A" else team_a_score
        total_score = max(1, selected_score + opponent_score)
        actual_score_share = selected_score / total_score
        score_performance_pp = (actual_score_share - expected_win) * 100

        h_rows = history[(history["match_id"] == match_id) & (history["player_id"] == str(selected_pid))] if not history.empty else pd.DataFrame()
        delta = float(h_rows["delta"].sum()) if not h_rows.empty else 0.0

        rows.append({
            "match_id": match_id,
            "scheduled_date": str(match.get("scheduled_date", "") or ""),
            "x_partner_relative_elo": partner_elo - player_elo,
            "y_opponent_relative_elo": opponent_team_elo - own_team_elo,
            "player_elo": player_elo,
            "partner_elo": partner_elo,
            "own_team_elo": own_team_elo,
            "opponent_team_elo": opponent_team_elo,
            "expected_loss_rate": expected_loss * 100,
            "actual_loss_rate": actual_loss * 100,
            "loss_risk_pp": loss_risk_pp,
            "expected_score_share": expected_win * 100,
            "actual_score_share": actual_score_share * 100,
            "score_performance_pp": score_performance_pp,
            "selected_score": selected_score,
            "opponent_score": opponent_score,
            "result": "Loss" if actual_loss else "Win",
            "elo_delta": delta,
            "partner_names": " / ".join(players_lookup.get(pid, pid) for pid in partner_ids),
            "opponent_names": " / ".join(players_lookup.get(pid, pid) for pid in opponent_ids),
            "score_label": f"{selected_score} - {opponent_score}",
            "winner": str(match.get("winner", "") or ""),
        })

    return pd.DataFrame(rows)


def build_player_head_to_head_df(
    selected_pid: str,
    matches_df: pd.DataFrame,
    participants_df: pd.DataFrame,
    elo_history_df: pd.DataFrame,
    players_lookup: dict[str, str],
) -> pd.DataFrame:
    if not selected_pid or matches_df.empty or participants_df.empty:
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
        selected_rows = p_match[p_match["player_id"] == str(selected_pid)]
        if selected_rows.empty:
            continue
        selected_team = str(selected_rows.iloc[0]["team"])
        won = str(match.get("winner", "")) == selected_team
        h_rows = history[(history["match_id"] == match_id) & (history["player_id"] == str(selected_pid))] if not history.empty else pd.DataFrame()
        delta = float(h_rows["delta"].sum()) if not h_rows.empty else 0.0
        opponents = p_match[p_match["team"] != selected_team]
        for _, opponent in opponents.iterrows():
            opp_pid = str(opponent["player_id"])
            rows.append({
                "opponent_id": opp_pid,
                "opponent": players_lookup.get(opp_pid, opp_pid),
                "match_id": match_id,
                "won": won,
                "delta": delta,
            })

    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    out = raw.groupby(["opponent_id", "opponent"], as_index=False).agg(
        matches=("match_id", "nunique"),
        wins=("won", "sum"),
        net_elo=("delta", "sum"),
        elo_lost=("delta", lambda s: float(s[s < 0].sum())),
        elo_gained=("delta", lambda s: float(s[s > 0].sum())),
    )
    out["losses"] = out["matches"] - out["wins"]
    out["win_rate"] = (out["wins"] / out["matches"] * 100).round(1)
    out["elo_lost_abs"] = out["elo_lost"].abs()
    out["hover"] = out.apply(
        lambda r: (
            f"<b>vs {r['opponent']}</b><br>"
            f"Matches: {int(r['matches'])}<br>"
            f"Wins/Losses: {int(r['wins'])}/{int(r['losses'])}<br>"
            f"Win rate: {float(r['win_rate']):.1f}%<br>"
            f"Net Elo: {float(r['net_elo']):+.0f}<br>"
            f"Elo gained/lost: +{float(r['elo_gained']):.0f} / -{float(r['elo_lost_abs']):.0f}"
        ),
        axis=1,
    )
    return out.sort_values(["win_rate", "matches"], ascending=[False, False])


def build_player_match_timeline_df(
    selected_pid: str,
    matches_df: pd.DataFrame,
    participants_df: pd.DataFrame,
    elo_history_df: pd.DataFrame,
    players_lookup: dict[str, str],
) -> pd.DataFrame:
    """All completed matches for one player, enriched with Elo expectation, score performance and form metrics."""
    if not selected_pid or matches_df.empty or participants_df.empty:
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
        history["old_elo"] = pd.to_numeric(history["old_elo"], errors="coerce").fillna(BASE_ELO)
        history["new_elo"] = pd.to_numeric(history["new_elo"], errors="coerce").fillna(BASE_ELO)

    completed = matches[matches["status"].astype(str).str.lower().eq("completed") & matches["winner"].astype(str).isin(["A", "B"])].copy()
    rows: list[dict] = []
    for _, match in completed.iterrows():
        match_id = str(match["match_id"])
        p_match = participants[participants["match_id"] == match_id].copy()
        selected_rows = p_match[p_match["player_id"] == str(selected_pid)]
        if selected_rows.empty:
            continue
        selected_team = str(selected_rows.iloc[0]["team"])
        opponent_team = "B" if selected_team == "A" else "A"
        team_ids = p_match[p_match["team"] == selected_team]["player_id"].astype(str).tolist()
        opponent_ids = p_match[p_match["team"] == opponent_team]["player_id"].astype(str).tolist()
        if not team_ids or not opponent_ids:
            continue

        old_elos = _match_old_elo_map(match_id, history)
        own_team_elo = sum(old_elos.get(pid, BASE_ELO) for pid in team_ids) / len(team_ids)
        opponent_team_elo = sum(old_elos.get(pid, BASE_ELO) for pid in opponent_ids) / len(opponent_ids)
        expected_win = _expected_win_rate(own_team_elo, opponent_team_elo)

        team_a_score = _safe_int(match.get("team_a_score"))
        team_b_score = _safe_int(match.get("team_b_score"))
        selected_score = team_a_score if selected_team == "A" else team_b_score
        opponent_score = team_b_score if selected_team == "A" else team_a_score
        total_score = max(1, selected_score + opponent_score)
        actual_score_share = selected_score / total_score
        score_performance_pp = (actual_score_share - expected_win) * 100
        won = str(match.get("winner", "")) == selected_team
        margin = selected_score - opponent_score
        h_rows = history[(history["match_id"] == match_id) & (history["player_id"] == str(selected_pid))] if not history.empty else pd.DataFrame()
        elo_delta = float(h_rows["delta"].sum()) if not h_rows.empty else 0.0
        old_elo = float(h_rows["old_elo"].iloc[0]) if not h_rows.empty and "old_elo" in h_rows else BASE_ELO
        new_elo = float(h_rows["new_elo"].iloc[-1]) if not h_rows.empty and "new_elo" in h_rows else old_elo + elo_delta
        match_date = pd.to_datetime(str(match.get("scheduled_date", "") or ""), errors="coerce")
        if pd.isna(match_date):
            match_date = pd.to_datetime(str(match.get("completed_at", "") or match.get("created_at", "") or ""), errors="coerce")

        rows.append({
            "match_id": match_id,
            "match_date": match_date,
            "match_date_label": match_date.strftime("%Y-%m-%d") if not pd.isna(match_date) else "",
            "selected_team": selected_team,
            "partner_names": " / ".join(players_lookup.get(pid, pid) for pid in team_ids if pid != str(selected_pid)) or "Singles",
            "opponent_names": " / ".join(players_lookup.get(pid, pid) for pid in opponent_ids),
            "team_score": selected_score,
            "opponent_score": opponent_score,
            "score_label": f"{selected_score} - {opponent_score}",
            "video_url": str(match.get("video_url", "") or ""),
            "notes": str(match.get("notes", "") or ""),
            "score_margin": margin,
            "won": won,
            "result": "Win" if won else "Loss",
            "expected_win_rate": expected_win * 100,
            "actual_score_share": actual_score_share * 100,
            "score_performance_pp": score_performance_pp,
            "elo_delta": elo_delta,
            "old_elo": old_elo,
            "new_elo": new_elo,
            "close_game": abs(margin) <= 2 or (selected_score >= 20 and opponent_score >= 20),
            "own_team_elo": own_team_elo,
            "opponent_team_elo": opponent_team_elo,
            "opponent_relative_elo": opponent_team_elo - own_team_elo,
        })

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["match_date", "match_id"])
    out["match_number"] = range(1, len(out) + 1)
    out["rolling_elo_delta_5"] = out["elo_delta"].rolling(5, min_periods=1).sum()
    out["rolling_win_rate_5"] = out["won"].astype(float).rolling(5, min_periods=1).mean() * 100
    out["rolling_score_performance_5"] = out["score_performance_pp"].rolling(5, min_periods=1).mean()
    return out


def build_player_clutch_summary_df(match_timeline_df: pd.DataFrame) -> pd.DataFrame:
    if match_timeline_df.empty:
        return pd.DataFrame()
    rows = []
    for label, subset in [
        ("Close games", match_timeline_df[match_timeline_df["close_game"]]),
        ("Other games", match_timeline_df[~match_timeline_df["close_game"]]),
    ]:
        if subset.empty:
            rows.append({"category": label, "matches": 0, "wins": 0, "win_rate": 0.0, "net_elo": 0.0, "avg_score_performance": 0.0})
        else:
            rows.append({
                "category": label,
                "matches": int(len(subset)),
                "wins": int(subset["won"].sum()),
                "win_rate": round(float(subset["won"].mean() * 100), 1),
                "net_elo": round(float(subset["elo_delta"].sum()), 1),
                "avg_score_performance": round(float(subset["score_performance_pp"].mean()), 1),
            })
    return pd.DataFrame(rows)


def _normalised_pair(ids: list[str]) -> tuple[str, ...]:
    return tuple(sorted(str(x) for x in ids if str(x)))


def _team_score_for_match(match: pd.Series, team: str) -> tuple[int, int]:
    team_a_score = _safe_int(match.get("team_a_score"))
    team_b_score = _safe_int(match.get("team_b_score"))
    if str(team) == "A":
        return team_a_score, team_b_score
    return team_b_score, team_a_score


def build_player_context_setup_options(
    selected_pid: str,
    matches_df: pd.DataFrame,
    participants_df: pd.DataFrame,
    players_lookup: dict[str, str],
) -> pd.DataFrame:
    """Unique doubles setups the selected player has appeared in.

    A setup is: selected player + partner vs opponent pair.
    """
    if not selected_pid or matches_df.empty or participants_df.empty:
        return pd.DataFrame()

    matches = matches_df.copy()
    participants = participants_df.copy()
    matches["match_id"] = matches["match_id"].astype(str)
    participants["match_id"] = participants["match_id"].astype(str)
    participants["player_id"] = participants["player_id"].astype(str)
    participants["team"] = participants["team"].astype(str)

    completed = matches[matches["status"].astype(str).str.lower().eq("completed") & matches["winner"].astype(str).isin(["A", "B"])].copy()
    rows: list[dict] = []
    for _, match in completed.iterrows():
        match_id = str(match["match_id"])
        p_match = participants[participants["match_id"] == match_id]
        selected_rows = p_match[p_match["player_id"] == str(selected_pid)]
        if selected_rows.empty:
            continue
        selected_team = str(selected_rows.iloc[0]["team"])
        team_ids = p_match[p_match["team"] == selected_team]["player_id"].astype(str).tolist()
        opponent_ids = p_match[p_match["team"] != selected_team]["player_id"].astype(str).tolist()
        partner_ids = [pid for pid in team_ids if pid != str(selected_pid)]
        if len(partner_ids) != 1 or len(opponent_ids) != 2:
            continue
        partner_id = partner_ids[0]
        opponent_key = _normalised_pair(opponent_ids)
        opponent_label = " / ".join(players_lookup.get(pid, pid) for pid in opponent_key)
        rows.append({
            "partner_id": partner_id,
            "partner": players_lookup.get(partner_id, partner_id),
            "opponent_key": "|".join(opponent_key),
            "opponent_ids": list(opponent_key),
            "opponents": opponent_label,
            "setup_label": f"{players_lookup.get(partner_id, partner_id)} vs {opponent_label}",
            "selected_match_count": 1,
        })
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    return raw.groupby(["partner_id", "partner", "opponent_key", "opponents", "setup_label"], as_index=False).agg(
        selected_match_count=("selected_match_count", "sum")
    ).sort_values(["partner", "opponents"])


def build_replacement_benchmark_df(
    selected_pid: str,
    partner_id: str,
    opponent_ids: list[str],
    matches_df: pd.DataFrame,
    participants_df: pd.DataFrame,
    elo_history_df: pd.DataFrame,
    players_lookup: dict[str, str],
) -> pd.DataFrame:
    """Compare all players who occupied the selected player's slot.

    Setup definition: candidate + same partner vs same opponent pair.
    Uses score performance vs Elo expectation so a close loss can still rate positively.
    """
    if not partner_id or not opponent_ids or matches_df.empty or participants_df.empty:
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
        history["old_elo"] = pd.to_numeric(history["old_elo"], errors="coerce").fillna(BASE_ELO)

    opponent_key = _normalised_pair(opponent_ids)
    completed = matches[matches["status"].astype(str).str.lower().eq("completed") & matches["winner"].astype(str).isin(["A", "B"])].copy()
    rows: list[dict] = []

    for _, match in completed.iterrows():
        match_id = str(match["match_id"])
        p_match = participants[participants["match_id"] == match_id]
        if p_match.empty:
            continue
        for team, team_rows in p_match.groupby("team"):
            team_ids = team_rows["player_id"].astype(str).tolist()
            if str(partner_id) not in team_ids:
                continue
            candidates = [pid for pid in team_ids if pid != str(partner_id)]
            if len(candidates) != 1:
                continue
            candidate_id = candidates[0]
            opponent_team_ids = p_match[p_match["team"] != str(team)]["player_id"].astype(str).tolist()
            if _normalised_pair(opponent_team_ids) != opponent_key:
                continue

            old_elos = _match_old_elo_map(match_id, history)
            own_team_elo = sum(old_elos.get(pid, BASE_ELO) for pid in team_ids) / len(team_ids)
            opponent_team_elo = sum(old_elos.get(pid, BASE_ELO) for pid in opponent_team_ids) / len(opponent_team_ids)
            expected_win = _expected_win_rate(own_team_elo, opponent_team_elo)
            team_score, opp_score = _team_score_for_match(match, str(team))
            total_score = max(1, team_score + opp_score)
            actual_score_share = team_score / total_score
            score_performance_pp = (actual_score_share - expected_win) * 100
            won = str(match.get("winner", "")) == str(team)
            h_rows = history[(history["match_id"] == match_id) & (history["player_id"] == str(candidate_id))] if not history.empty else pd.DataFrame()
            elo_delta = float(h_rows["delta"].sum()) if not h_rows.empty else 0.0
            match_date = pd.to_datetime(str(match.get("scheduled_date", "") or ""), errors="coerce")
            if pd.isna(match_date):
                match_date = pd.to_datetime(str(match.get("completed_at", "") or match.get("created_at", "") or ""), errors="coerce")

            rows.append({
                "candidate_id": candidate_id,
                "candidate": players_lookup.get(candidate_id, candidate_id),
                "is_selected_player": str(candidate_id) == str(selected_pid),
                "partner_id": str(partner_id),
                "partner": players_lookup.get(str(partner_id), str(partner_id)),
                "opponents": " / ".join(players_lookup.get(pid, pid) for pid in opponent_key),
                "match_id": match_id,
                "match_date": match_date,
                "match_date_label": match_date.strftime("%Y-%m-%d") if not pd.isna(match_date) else "",
                "score_label": f"{team_score} - {opp_score}",
                "result": "Win" if won else "Loss",
                "won": won,
                "expected_score_share": expected_win * 100,
                "actual_score_share": actual_score_share * 100,
                "score_performance_pp": score_performance_pp,
                "elo_delta": elo_delta,
                "video_url": str(match.get("video_url", "") or ""),
                "notes": str(match.get("notes", "") or ""),
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["candidate", "match_date", "match_id"])


def summarise_replacement_benchmark(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    if benchmark_df.empty:
        return pd.DataFrame()
    summary = benchmark_df.groupby(["candidate_id", "candidate", "is_selected_player"], as_index=False).agg(
        matches=("match_id", "nunique"),
        wins=("won", "sum"),
        avg_performance_pp=("score_performance_pp", "mean"),
        total_elo=("elo_delta", "sum"),
        avg_elo=("elo_delta", "mean"),
    )
    summary["losses"] = summary["matches"] - summary["wins"]
    summary["win_rate"] = (summary["wins"] / summary["matches"] * 100).round(1)
    others = summary[~summary["is_selected_player"]]
    others_avg = float(others["avg_performance_pp"].mean()) if not others.empty else 0.0
    summary["replacement_value_pp"] = summary["avg_performance_pp"] - others_avg
    return summary.sort_values(["avg_performance_pp", "matches", "win_rate"], ascending=[False, False, False])
