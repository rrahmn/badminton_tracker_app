from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import plotly.express as px
import streamlit as st

from src.elo import BASE_ELO, update_team_elos
from src.stats import build_player_stats, current_elo_map
from src.storage import CSVStorage, DATA_FILES


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
st.set_page_config(page_title="Badminton Tracker", layout="wide")
st.title("🏸 Badminton Tracker")
st.caption("Track singles and doubles matches, player events, Elo and exports.")

storage = CSVStorage(DATA_DIR)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_load(name: str) -> pd.DataFrame:
    df = storage.load(name)
    for col in DATA_FILES[name]:
        if col not in df.columns:
            df[col] = None

    if name == "matches":
        text_cols = [
            "match_id",
            "created_at",
            "completed_at",
            "match_type",
            "team_a_players",
            "team_b_players",
            "winner",
            "status",
        ]
        int_cols = ["points_to_win", "team_a_score", "team_b_score"]
        for col in text_cols:
            df[col] = df[col].fillna("").astype(str)
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    elif name == "events":
        text_cols = ["event_id", "match_id", "timestamp", "team", "player_id", "event_type", "note"]
        int_cols = ["event_index", "points_awarded"]
        for col in text_cols:
            df[col] = df[col].fillna("").astype(str)
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    elif name == "players":
        df["player_id"] = df["player_id"].fillna("").astype(str)
        df["name"] = df["name"].fillna("").astype(str)
        df["created_at"] = df["created_at"].fillna("").astype(str)
        df["is_active"] = df["is_active"].fillna(True)

    elif name == "elo_history":
        text_cols = ["history_id", "match_id", "player_id", "recorded_at"]
        num_cols = ["old_elo", "new_elo", "delta"]
        for col in text_cols:
            df[col] = df[col].fillna("").astype(str)
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def refresh_state() -> None:
    st.session_state.players_df = safe_load("players")
    st.session_state.matches_df = safe_load("matches")
    st.session_state.events_df = safe_load("events")
    st.session_state.elo_history_df = safe_load("elo_history")


if "booted" not in st.session_state:
    st.session_state.booted = True
    refresh_state()

players_df = st.session_state.players_df
matches_df = st.session_state.matches_df
events_df = st.session_state.events_df
elo_history_df = st.session_state.elo_history_df


def save_df(name: str, df: pd.DataFrame) -> None:
    storage.save(name, df)
    refresh_state()


def player_name_map() -> dict[str, str]:
    return {str(row["player_id"]): row["name"] for _, row in st.session_state.players_df.iterrows()}


def get_active_match() -> pd.Series | None:
    live = st.session_state.matches_df[st.session_state.matches_df["status"] == "In Progress"]
    if live.empty:
        return None
    return live.sort_values("created_at").iloc[-1]


def get_match_events(match_id: str) -> pd.DataFrame:
    df = st.session_state.events_df
    if df.empty:
        return df.copy()
    return df[df["match_id"] == match_id].sort_values("event_index")


def parse_players(cell: str) -> list[str]:
    return [x for x in str(cell).split("|") if x]


def record_event(match_id: str, team: str, player_id: str, event_type: str, points_awarded: int = 0, note: str = "") -> None:
    current_events = get_match_events(match_id)
    row = {
        "event_id": str(uuid4()),
        "match_id": match_id,
        "timestamp": now_iso(),
        "event_index": int(current_events["event_index"].max() + 1) if not current_events.empty else 1,
        "team": team,
        "player_id": player_id,
        "event_type": event_type,
        "points_awarded": points_awarded,
        "note": note,
    }
    new_events = pd.concat([st.session_state.events_df, pd.DataFrame([row])], ignore_index=True)
    storage.save("events", new_events)

    matches = st.session_state.matches_df.copy()
    idx = matches.index[matches["match_id"] == match_id]
    if len(idx) == 1 and points_awarded:
        score_col = "team_a_score" if team == "A" else "team_b_score"
        matches.loc[idx, score_col] = pd.to_numeric(matches.loc[idx, score_col], errors="coerce").fillna(0).astype(int) + points_awarded
        storage.save("matches", matches)
    refresh_state()


def undo_last_event(match_id: str) -> None:
    match_events = get_match_events(match_id)
    if match_events.empty:
        st.warning("No events to undo.")
        return
    last_event = match_events.iloc[-1]

    events = st.session_state.events_df.copy()
    events = events[events["event_id"] != last_event["event_id"]]
    storage.save("events", events)

    if int(last_event.get("points_awarded", 0) or 0) > 0:
        matches = st.session_state.matches_df.copy()
        idx = matches.index[matches["match_id"] == match_id]
        if len(idx) == 1:
            score_col = "team_a_score" if last_event["team"] == "A" else "team_b_score"
            current_val = int(pd.to_numeric(matches.loc[idx, score_col], errors="coerce").fillna(0).iloc[0])
            matches.loc[idx, score_col] = max(0, current_val - int(last_event["points_awarded"]))
            storage.save("matches", matches)
    refresh_state()


def complete_match(match_id: str) -> None:
    matches = st.session_state.matches_df.copy()
    idx = matches.index[matches["match_id"] == match_id]
    if len(idx) != 1:
        return
    row = matches.loc[idx[0]]
    team_a_score = int(row["team_a_score"])
    team_b_score = int(row["team_b_score"])
    if team_a_score == team_b_score:
        st.error("A match needs a winner. Scores cannot be tied.")
        return

    winner = "A" if team_a_score > team_b_score else "B"
    matches["winner"] = matches["winner"].astype(object)
    matches["status"] = matches["status"].astype(object)
    matches["completed_at"] = matches["completed_at"].astype(object)
    matches.loc[idx, "winner"] = winner
    matches.loc[idx, "status"] = "Completed"
    matches.loc[idx, "completed_at"] = now_iso()
    storage.save("matches", matches)

    players_df = st.session_state.players_df.copy()
    elo_history = st.session_state.elo_history_df.copy()
    elo_map = current_elo_map(players_df, elo_history)

    team_a_ids = parse_players(row["team_a_players"])
    team_b_ids = parse_players(row["team_b_players"])
    team_a_old = [elo_map.get(pid, BASE_ELO) for pid in team_a_ids]
    team_b_old = [elo_map.get(pid, BASE_ELO) for pid in team_b_ids]
    team_a_new, team_b_new = update_team_elos(team_a_old, team_b_old, winner)

    history_rows = []
    for pid, old, new in zip(team_a_ids, team_a_old, team_a_new):
        history_rows.append({
            "history_id": str(uuid4()),
            "match_id": match_id,
            "player_id": pid,
            "old_elo": old,
            "new_elo": new,
            "delta": round(new - old, 2),
            "recorded_at": now_iso(),
        })
    for pid, old, new in zip(team_b_ids, team_b_old, team_b_new):
        history_rows.append({
            "history_id": str(uuid4()),
            "match_id": match_id,
            "player_id": pid,
            "old_elo": old,
            "new_elo": new,
            "delta": round(new - old, 2),
            "recorded_at": now_iso(),
        })

    elo_history = pd.concat([elo_history, pd.DataFrame(history_rows)], ignore_index=True)
    storage.save("elo_history", elo_history)
    refresh_state()


def export_zip_bytes() -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in DATA_FILES:
            zf.write(DATA_DIR / f"{name}.csv", arcname=f"{name}.csv")
    mem.seek(0)
    return mem.getvalue()


with st.sidebar:
    st.header("Players")
    with st.form("add_player_form", clear_on_submit=True):
        new_player_name = st.text_input("Add player")
        submitted = st.form_submit_button("Save player")
        if submitted:
            if not new_player_name.strip():
                st.error("Enter a player name.")
            elif players_df["name"].astype(str).str.lower().eq(new_player_name.strip().lower()).any():
                st.error("That player already exists.")
            else:
                storage.append_row(
                    "players",
                    {
                        "player_id": str(uuid4()),
                        "name": new_player_name.strip(),
                        "created_at": now_iso(),
                        "is_active": True,
                    },
                )
                refresh_state()
                st.success(f"Added {new_player_name.strip()}")

    st.divider()
    st.header("Data")
    st.download_button(
        "Export data (.zip)",
        data=export_zip_bytes(),
        file_name="badminton_tracker_data.zip",
        mime="application/zip",
    )

    uploaded = st.file_uploader("Import data zip", type=["zip"])
    if uploaded is not None:
        try:
            with zipfile.ZipFile(uploaded) as zf:
                payload = {}
                for name, columns in DATA_FILES.items():
                    with zf.open(f"{name}.csv") as f:
                        df = pd.read_csv(f)
                        for column in columns:
                            if column not in df.columns:
                                df[column] = None
                        payload[name] = df[columns]
                storage.replace_all(payload)
                refresh_state()
                st.success("Data imported.")
        except Exception as exc:
            st.error(f"Could not import zip: {exc}")


players_lookup = player_name_map()
stats_df = build_player_stats(players_df, matches_df, events_df, elo_history_df)


tab1, tab2, tab3, tab4 = st.tabs(["Live Match", "Players & Elo", "Stats", "Match History"])

with tab1:
    st.subheader("Live Match Recorder")
    active_match = get_active_match()

    if active_match is None:
        st.info("No live match. Create one below.")
        if players_df.empty:
            st.warning("Add players first.")
        else:
            with st.form("create_match_form"):
                match_type = st.selectbox("Match type", ["Singles", "Doubles"])
                points_to_win = st.number_input("Points to win", min_value=1, max_value=99, value=21)
                names = players_df["name"].sort_values().tolist()
                team_a = st.multiselect("Team A", names, max_selections=2)
                team_b = st.multiselect("Team B", names, max_selections=2)
                create_match = st.form_submit_button("Create match")

                if create_match:
                    expected = 1 if match_type == "Singles" else 2
                    if len(team_a) != expected or len(team_b) != expected:
                        st.error(f"{match_type} requires {expected} player(s) on each side.")
                    elif set(team_a) & set(team_b):
                        st.error("A player cannot be on both sides.")
                    else:
                        name_to_id = {row["name"]: str(row["player_id"]) for _, row in players_df.iterrows()}
                        row = {
                            "match_id": str(uuid4()),
                            "created_at": now_iso(),
                            "completed_at": "",
                            "match_type": match_type,
                            "points_to_win": int(points_to_win),
                            "team_a_players": "|".join(name_to_id[n] for n in team_a),
                            "team_b_players": "|".join(name_to_id[n] for n in team_b),
                            "team_a_score": 0,
                            "team_b_score": 0,
                            "winner": "",
                            "status": "In Progress",
                        }
                        storage.append_row("matches", row)
                        refresh_state()
                        st.success("Match created.")
                        st.rerun()
    else:
        team_a_ids = parse_players(active_match["team_a_players"])
        team_b_ids = parse_players(active_match["team_b_players"])
        team_a_names = [players_lookup.get(pid, pid) for pid in team_a_ids]
        team_b_names = [players_lookup.get(pid, pid) for pid in team_b_ids]

        a_col, mid_col, b_col = st.columns([1, 0.6, 1])
        with a_col:
            st.markdown("### Team A")
            st.markdown(" / ".join(team_a_names))
            st.metric("Score", int(active_match["team_a_score"]))
        with mid_col:
            st.markdown("### vs")
            st.write(f"{active_match['match_type']} · First to {int(active_match['points_to_win'])}")
        with b_col:
            st.markdown("### Team B")
            st.markdown(" / ".join(team_b_names))
            st.metric("Score", int(active_match["team_b_score"]))

        st.markdown("#### Record point")
        p1, p2 = st.columns(2)
        with p1:
            scorer_a = st.selectbox("Who scored for Team A?", team_a_names, key="scorer_a")
            if st.button("+1 Team A point", use_container_width=True):
                scorer_id = next(pid for pid, name in players_lookup.items() if name == scorer_a)
                record_event(str(active_match["match_id"]), "A", scorer_id, "point", points_awarded=1)
                st.rerun()
        with p2:
            scorer_b = st.selectbox("Who scored for Team B?", team_b_names, key="scorer_b")
            if st.button("+1 Team B point", use_container_width=True):
                scorer_id = next(pid for pid, name in players_lookup.items() if name == scorer_b)
                record_event(str(active_match["match_id"]), "B", scorer_id, "point", points_awarded=1)
                st.rerun()

        st.markdown("#### Record player event")
        event_cols = st.columns(4)
        all_live_names = team_a_names + team_b_names
        selected_player_name = st.selectbox("Player", all_live_names)
        selected_team = "A" if selected_player_name in team_a_names else "B"
        selected_pid = next(pid for pid, name in players_lookup.items() if name == selected_player_name)
        if event_cols[0].button("Good shot", use_container_width=True):
            record_event(str(active_match["match_id"]), selected_team, selected_pid, "good_shot")
            st.rerun()
        if event_cols[1].button("Bad shot", use_container_width=True):
            record_event(str(active_match["match_id"]), selected_team, selected_pid, "bad_shot")
            st.rerun()
        if event_cols[2].button("Service fault", use_container_width=True):
            record_event(str(active_match["match_id"]), selected_team, selected_pid, "service_fault")
            st.rerun()
        if event_cols[3].button("Undo last event", use_container_width=True):
            undo_last_event(str(active_match["match_id"]))
            st.rerun()

        controls = st.columns(2)
        reached_limit = int(active_match["team_a_score"]) >= int(active_match["points_to_win"]) or int(active_match["team_b_score"]) >= int(active_match["points_to_win"])
        if controls[0].button("Complete match", disabled=not reached_limit, use_container_width=True):
            complete_match(str(active_match["match_id"]))
            st.success("Match completed and Elo updated.")
            st.rerun()
        if controls[1].button("Force complete now", use_container_width=True):
            complete_match(str(active_match["match_id"]))
            st.success("Match completed and Elo updated.")
            st.rerun()

        event_log = get_match_events(str(active_match["match_id"])).copy()
        if not event_log.empty:
            event_log["player"] = event_log["player_id"].astype(str).map(players_lookup)
            st.markdown("#### Event log")
            st.dataframe(event_log[["event_index", "timestamp", "team", "player", "event_type", "points_awarded"]], use_container_width=True)

with tab2:
    st.subheader("Players & Elo")
    if stats_df.empty:
        st.info("Add players and complete matches to see Elo.")
    else:
        leaderboard = stats_df[["name", "elo", "matches_played", "wins", "losses", "win_rate"]].copy()
        st.dataframe(leaderboard, use_container_width=True)
        elo_chart = px.bar(leaderboard.sort_values("elo", ascending=False), x="name", y="elo", title="Current Elo")
        st.plotly_chart(elo_chart, use_container_width=True)

with tab3:
    st.subheader("Player Stats")
    if stats_df.empty:
        st.info("No stats yet.")
    else:
        st.dataframe(stats_df, use_container_width=True)
        metric_choice = st.selectbox("Chart", ["points_won", "good_shots", "bad_shots", "service_faults", "shot_balance"])
        chart_df = stats_df[["name", metric_choice]].sort_values(metric_choice, ascending=False)
        stat_chart = px.bar(chart_df, x="name", y=metric_choice, title=f"{metric_choice.replace('_', ' ').title()} by player")
        st.plotly_chart(stat_chart, use_container_width=True)

with tab4:
    st.subheader("Match History")
    history = matches_df.copy()
    if history.empty:
        st.info("No matches yet.")
    else:
        history["team_a_names"] = history["team_a_players"].apply(lambda x: " / ".join(players_lookup.get(pid, pid) for pid in parse_players(x)))
        history["team_b_names"] = history["team_b_players"].apply(lambda x: " / ".join(players_lookup.get(pid, pid) for pid in parse_players(x)))
        history["winner_label"] = history.apply(
            lambda row: row["team_a_names"] if row["winner"] == "A" else (row["team_b_names"] if row["winner"] == "B" else ""),
            axis=1,
        )
        st.dataframe(
            history[[
                "created_at",
                "match_type",
                "points_to_win",
                "team_a_names",
                "team_b_names",
                "team_a_score",
                "team_b_score",
                "winner_label",
                "status",
            ]],
            use_container_width=True,
        )

        completed = history[history["status"] == "Completed"]
        if not completed.empty:
            completed["played_on"] = pd.to_datetime(completed["completed_at"], errors="coerce").dt.date.astype(str)
            match_chart = px.histogram(completed, x="played_on", title="Matches completed over time")
            st.plotly_chart(match_chart, use_container_width=True)
