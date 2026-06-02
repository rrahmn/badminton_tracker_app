from __future__ import annotations

import re
from difflib import get_close_matches
from datetime import date, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from uuid import uuid4

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.auth import auth_is_configured, get_display_name, get_role, logout, require_editor, require_login
from src.analytics import build_elo_timeline_df, build_partner_matrix_df, build_player_relationship_insights, build_player_loss_risk_df, build_player_head_to_head_df, build_player_match_timeline_df, build_player_clutch_summary_df, build_player_context_setup_options, build_replacement_benchmark_df, summarise_replacement_benchmark, build_player_elo_relationship_timeline_df
from src.elo import BASE_ELO, ELO_MODEL_VERSION, K_FACTOR, update_team_elos
from src.stats import build_player_stats, current_elo_map
from src.storage import CSVStorage, SupabaseStorage, DATA_FILES


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
st.set_page_config(page_title="Badminton Tracker", layout="wide")
st.title("🏸 Badminton Tracker")
st.caption("Track singles and doubles matches, clips, player events and Elo.")

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 1.4rem; max-width: 1400px;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(120,120,120,.15);}
[data-testid="stMetricValue"] {font-size: 1.45rem;}
div[data-baseweb="tab-list"] {gap: .35rem; flex-wrap: wrap;}
div[data-baseweb="tab"] {background: rgba(255,255,255,.04); border: 1px solid rgba(120,120,120,.18); padding: .45rem .8rem; border-radius: 999px;}
div[data-baseweb="tab"][aria-selected="true"] {background: rgba(34,197,94,.12); border-color: rgba(34,197,94,.38);}
.st-emotion-cache-1r6slb0, .st-emotion-cache-13ln4jf {border-radius: 18px;}
[data-testid="stDataFrame"] {border: 1px solid rgba(120,120,120,.12); border-radius: 16px; overflow: hidden;}
[data-testid="stVerticalBlock"] div[data-testid="stForm"] {border: 1px solid rgba(120,120,120,.12); border-radius: 18px; padding: 1rem; background: rgba(255,255,255,.02);}
</style>
""", unsafe_allow_html=True)

require_login()
current_role = get_role()

def build_storage():
    try:
        supabase_cfg = st.secrets.get("supabase", {})
    except Exception:
        supabase_cfg = {}
    url = str(supabase_cfg.get("url", "") or "").strip()
    key = str(supabase_cfg.get("key", "") or "").strip()
    if url and key:
        return SupabaseStorage(url=url, key=key), "Supabase"
    return CSVStorage(DATA_DIR), "CSV"


storage, storage_backend = build_storage()

TIME_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2})$|^(\d+)$")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def format_seconds(value: int | float | str | None) -> str:
    try:
        total = int(float(value or 0))
    except (TypeError, ValueError):
        total = 0
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def parse_time_to_seconds(raw: str) -> int | None:
    text = str(raw or "").strip()
    if not text:
        return None
    match = TIME_RE.match(text)
    if not match:
        return None
    if match.group(4):
        return int(match.group(4))
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def extract_youtube_video_id(url: str) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
    except Exception:
        return ""

    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").strip("/")
    query = parse_qs(parsed.query)

    if "youtu.be" in host:
        return path.split("/")[0] if path else ""
    if "youtube.com" in host:
        if path == "watch":
            return query.get("v", [""])[0]
        if path.startswith("embed/"):
            return path.split("/", 1)[1].split("/")[0]
        if path.startswith("shorts/"):
            return path.split("/", 1)[1].split("/")[0]
    return ""


def build_youtube_embed_url(url: str) -> str:
    video_id = extract_youtube_video_id(url)
    if not video_id:
        return ""
    return (
        f"https://www.youtube.com/embed/{video_id}"
        "?rel=0&modestbranding=1&playsinline=1"
    )


def render_youtube_video(url: str, height: int = 520) -> None:
    embed_url = build_youtube_embed_url(url)
    if embed_url:
        st.components.v1.iframe(embed_url, height=height, scrolling=False)
    elif url:
        st.info("The saved video URL could not be embedded here. Use the open video link below.")


def build_youtube_timestamp_url(url: str, seconds: int | None) -> str:
    if not url or seconds is None:
        return ""
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    video_id = extract_youtube_video_id(url)
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}&t={int(seconds)}s"
    query["t"] = [str(int(seconds))]
    new_query = urlencode(query, doseq=True)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))


def safe_load(name: str) -> pd.DataFrame:
    df = storage.load(name)
    for col in DATA_FILES[name]:
        if col not in df.columns:
            df[col] = None

    if name == "matches":
        text_cols = [
            "match_id", "created_at", "completed_at", "match_type", "team_a_players", "team_b_players",
            "winner", "status", "video_url", "scheduled_date", "scheduled_time", "notes"
        ]
        int_cols = ["points_to_win", "team_a_score", "team_b_score"]
        for col in text_cols:
            df[col] = df[col].fillna("").astype(str)
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    elif name == "events":
        # Backward compatibility for older event schema.
        if "video_timestamp_seconds" in df.columns:
            start_empty = df["video_start_seconds"].isna() | (df["video_start_seconds"].astype(str) == "")
            df.loc[start_empty, "video_start_seconds"] = df.loc[start_empty, "video_timestamp_seconds"]
        if "video_timestamp_label" in df.columns:
            start_label_empty = df["video_start_label"].isna() | (df["video_start_label"].astype(str) == "")
            df.loc[start_label_empty, "video_start_label"] = df.loc[start_label_empty, "video_timestamp_label"]
        if "video_link" in df.columns:
            clip_empty = df["clip_url"].isna() | (df["clip_url"].astype(str) == "")
            df.loc[clip_empty, "clip_url"] = df.loc[clip_empty, "video_link"]

        text_cols = [
            "event_id", "match_id", "timestamp", "team", "player_id", "event_type", "note",
            "video_start_label", "video_end_label", "clip_url"
        ]
        int_cols = ["event_index", "points_awarded", "video_start_seconds", "video_end_seconds"]
        for col in text_cols:
            df[col] = df[col].fillna("").astype(str)
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        end_missing = df["video_end_seconds"] <= 0
        df.loc[end_missing, "video_end_seconds"] = df.loc[end_missing, "video_start_seconds"]
        end_label_missing = df["video_end_label"].eq("")
        df.loc[end_label_missing, "video_end_label"] = df.loc[end_label_missing, "video_start_label"]

    elif name == "players":
        df["player_id"] = df["player_id"].fillna("").astype(str)
        df["name"] = df["name"].fillna("").astype(str)
        df["created_at"] = df["created_at"].fillna("").astype(str)
        df["is_active"] = df["is_active"].fillna(True)

    elif name == "elo_history":
        text_cols = ["history_id", "match_id", "player_id", "recorded_at", "elo_model_version"]
        num_cols = ["old_elo", "new_elo", "delta", "k_factor_used"]
        for col in text_cols:
            df[col] = df[col].fillna("").astype(str)
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    elif name == "match_participants":
        text_cols = ["match_id", "player_id", "team"]
        int_cols = ["slot"]
        for col in text_cols:
            df[col] = df[col].fillna("").astype(str)
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df[DATA_FILES[name]]


def refresh_state() -> None:
    st.session_state.players_df = safe_load("players")
    st.session_state.matches_df = safe_load("matches")
    st.session_state.events_df = safe_load("events")
    st.session_state.elo_history_df = safe_load("elo_history")
    st.session_state.match_participants_df = safe_load("match_participants")


if "booted" not in st.session_state:
    st.session_state.booted = True
    refresh_state()

players_df = st.session_state.players_df
matches_df = st.session_state.matches_df
events_df = st.session_state.events_df
elo_history_df = st.session_state.elo_history_df
match_participants_df = st.session_state.match_participants_df


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


def save_df(name: str, df: pd.DataFrame) -> None:
    storage.save(name, df)
    refresh_state()


def get_match_participants(match_id: str) -> pd.DataFrame:
    df = st.session_state.match_participants_df
    if df.empty:
        return df.copy()
    out = df[df["match_id"].astype(str) == str(match_id)].copy()
    if not out.empty:
        out = out.sort_values(["team", "slot", "player_id"])
    return out


def get_match_team_ids(match_row: pd.Series, team: str) -> list[str]:
    team = str(team)
    match_id = str(match_row.get("match_id", ""))
    participants = get_match_participants(match_id)
    if not participants.empty:
        return participants.loc[participants["team"] == team, "player_id"].astype(str).tolist()
    legacy_col = "team_a_players" if team == "A" else "team_b_players"
    return parse_players(str(match_row.get(legacy_col, "")))


def create_match_participants_rows(match_id: str, team_a_ids: list[str], team_b_ids: list[str]) -> pd.DataFrame:
    rows = []
    for slot, pid in enumerate(team_a_ids, start=1):
        rows.append({"match_id": match_id, "player_id": str(pid), "team": "A", "slot": slot})
    for slot, pid in enumerate(team_b_ids, start=1):
        rows.append({"match_id": match_id, "player_id": str(pid), "team": "B", "slot": slot})
    return pd.DataFrame(rows, columns=DATA_FILES["match_participants"])


def update_match_details(match_id: str, *, video_url: str, scheduled_date: str, scheduled_time: str, notes: str) -> None:
    matches = st.session_state.matches_df.copy()
    idx = matches.index[matches["match_id"] == match_id]
    if len(idx) != 1:
        return
    for col in ["video_url", "scheduled_date", "scheduled_time", "notes"]:
        matches[col] = matches[col].astype(object)
    matches.loc[idx, "video_url"] = (video_url or "").strip()
    matches.loc[idx, "scheduled_date"] = (scheduled_date or "").strip()
    matches.loc[idx, "scheduled_time"] = (scheduled_time or "").strip()
    matches.loc[idx, "notes"] = (notes or "").strip()
    save_df("matches", matches)


def record_event(
    match_id: str,
    team: str,
    player_id: str,
    event_type: str,
    points_awarded: int = 0,
    note: str = "",
    video_start_seconds: int | None = None,
    video_end_seconds: int | None = None,
    update_match_score: bool = True,
) -> None:
    current_events = get_match_events(match_id)
    match_lookup = st.session_state.matches_df[st.session_state.matches_df["match_id"].astype(str) == str(match_id)]
    video_url = str(match_lookup.iloc[0].get("video_url", "") or "") if not match_lookup.empty else ""

    start_seconds = int(video_start_seconds) if video_start_seconds is not None else 0
    end_seconds = int(video_end_seconds) if video_end_seconds is not None else start_seconds
    if end_seconds < start_seconds:
        start_seconds, end_seconds = end_seconds, start_seconds

    row = {
        "event_id": str(uuid4()),
        "match_id": match_id,
        "timestamp": now_iso(),
        "event_index": int(current_events["event_index"].max() + 1) if not current_events.empty else 1,
        "team": team,
        "player_id": str(player_id or ""),
        "event_type": event_type,
        "points_awarded": points_awarded,
        "note": note,
        "video_start_seconds": start_seconds,
        "video_end_seconds": end_seconds,
        "video_start_label": format_seconds(start_seconds) if start_seconds > 0 else "",
        "video_end_label": format_seconds(end_seconds) if end_seconds > 0 else "",
        "clip_url": build_youtube_timestamp_url(video_url, start_seconds) if video_url and start_seconds > 0 else "",
    }
    new_events = pd.concat([st.session_state.events_df, pd.DataFrame([row])], ignore_index=True)
    storage.save("events", new_events)

    matches = st.session_state.matches_df.copy()
    idx = matches.index[matches["match_id"] == match_id]
    if update_match_score and len(idx) == 1 and points_awarded:
        score_col = "team_a_score" if team == "A" else "team_b_score"
        current_val = int(pd.to_numeric(matches.loc[idx, score_col], errors="coerce").fillna(0).iloc[0])
        matches.loc[idx, score_col] = current_val + points_awarded
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


def match_has_elo_update(match_id: str) -> bool:
    history = st.session_state.elo_history_df
    if history.empty:
        return False
    return history["match_id"].astype(str).eq(str(match_id)).any()


def apply_elo_for_match(match_id: str, winner: str, match_row: pd.Series | None = None) -> None:
    if match_has_elo_update(match_id):
        return

    if match_row is None:
        matches = st.session_state.matches_df
        rows = matches[matches["match_id"].astype(str) == str(match_id)]
        if rows.empty:
            return
        match_row = rows.iloc[0]

    players_df = st.session_state.players_df.copy()
    elo_history = st.session_state.elo_history_df.copy()
    elo_map = current_elo_map(players_df, elo_history)

    team_a_ids = get_match_team_ids(match_row, "A")
    team_b_ids = get_match_team_ids(match_row, "B")
    if not team_a_ids or not team_b_ids or winner not in {"A", "B"}:
        return

    team_a_old = [elo_map.get(pid, BASE_ELO) for pid in team_a_ids]
    team_b_old = [elo_map.get(pid, BASE_ELO) for pid in team_b_ids]
    team_a_new, team_b_new = update_team_elos(team_a_old, team_b_old, winner)

    history_rows = []
    for pid, old, new in zip(team_a_ids, team_a_old, team_a_new):
        history_rows.append({
            "history_id": str(uuid4()),
            "match_id": match_id,
            "player_id": pid,
            "old_elo": int(round(old)),
            "new_elo": int(round(new)),
            "delta": int(round(new - old)),
            "recorded_at": now_iso(),
            "elo_model_version": ELO_MODEL_VERSION,
            "k_factor_used": int(round(K_FACTOR)),
        })
    for pid, old, new in zip(team_b_ids, team_b_old, team_b_new):
        history_rows.append({
            "history_id": str(uuid4()),
            "match_id": match_id,
            "player_id": pid,
            "old_elo": int(round(old)),
            "new_elo": int(round(new)),
            "delta": int(round(new - old)),
            "recorded_at": now_iso(),
            "elo_model_version": ELO_MODEL_VERSION,
            "k_factor_used": int(round(K_FACTOR)),
        })

    if history_rows:
        elo_history = pd.concat([elo_history, pd.DataFrame(history_rows)], ignore_index=True)
        storage.save("elo_history", elo_history)
        refresh_state()



def _match_sort_key(row: pd.Series) -> pd.Timestamp:
    date_part = str(row.get("scheduled_date", "") or "").strip()
    time_part = str(row.get("scheduled_time", "") or "").strip()
    candidates = []
    if date_part:
        candidates.append(f"{date_part} {time_part}".strip())
    candidates.extend([
        str(row.get("completed_at", "") or "").strip(),
        str(row.get("created_at", "") or "").strip(),
    ])
    for candidate in candidates:
        if not candidate:
            continue
        parsed = pd.to_datetime(candidate, errors="coerce")
        if not pd.isna(parsed):
            return parsed
    return pd.Timestamp.max


def recalculate_elo_history() -> int:
    require_editor()

    completed_matches = st.session_state.matches_df.copy()
    if completed_matches.empty:
        storage.save("elo_history", pd.DataFrame(columns=DATA_FILES["elo_history"]))
        refresh_state()
        return 0

    completed_matches = completed_matches[
        completed_matches["status"].astype(str).str.lower().eq("completed")
        & completed_matches["winner"].astype(str).isin(["A", "B"])
    ].copy()

    if completed_matches.empty:
        storage.save("elo_history", pd.DataFrame(columns=DATA_FILES["elo_history"]))
        refresh_state()
        return 0

    completed_matches["_elo_sort_key"] = completed_matches.apply(_match_sort_key, axis=1)
    completed_matches = completed_matches.sort_values(["_elo_sort_key", "created_at", "match_id"])

    elo_map = {str(row["player_id"]): BASE_ELO for _, row in st.session_state.players_df.iterrows()}
    rebuilt_rows: list[dict] = []

    for _, match_row in completed_matches.iterrows():
        match_id = str(match_row.get("match_id", "") or "").strip()
        winner = str(match_row.get("winner", "") or "").strip()
        team_a_ids = get_match_team_ids(match_row, "A")
        team_b_ids = get_match_team_ids(match_row, "B")
        if not match_id or winner not in {"A", "B"} or not team_a_ids or not team_b_ids:
            continue

        team_a_old = [elo_map.get(pid, BASE_ELO) for pid in team_a_ids]
        team_b_old = [elo_map.get(pid, BASE_ELO) for pid in team_b_ids]
        team_a_new, team_b_new = update_team_elos(team_a_old, team_b_old, winner)
        recorded_at = str(match_row.get("completed_at", "") or "").strip() or str(match_row.get("created_at", "") or "").strip() or now_iso()

        for pid, old, new in zip(team_a_ids, team_a_old, team_a_new):
            old_i = int(round(old))
            new_i = int(round(new))
            rebuilt_rows.append({
                "history_id": str(uuid4()),
                "match_id": match_id,
                "player_id": pid,
                "old_elo": old_i,
                "new_elo": new_i,
                "delta": new_i - old_i,
                "recorded_at": recorded_at,
                "elo_model_version": ELO_MODEL_VERSION,
                "k_factor_used": int(round(K_FACTOR)),
            })
            elo_map[pid] = new_i

        for pid, old, new in zip(team_b_ids, team_b_old, team_b_new):
            old_i = int(round(old))
            new_i = int(round(new))
            rebuilt_rows.append({
                "history_id": str(uuid4()),
                "match_id": match_id,
                "player_id": pid,
                "old_elo": old_i,
                "new_elo": new_i,
                "delta": new_i - old_i,
                "recorded_at": recorded_at,
                "elo_model_version": ELO_MODEL_VERSION,
                "k_factor_used": int(round(K_FACTOR)),
            })
            elo_map[pid] = new_i

    rebuilt = pd.DataFrame(rebuilt_rows, columns=DATA_FILES["elo_history"])
    storage.save("elo_history", rebuilt)
    refresh_state()
    return len(completed_matches)

def complete_match(match_id: str) -> None:
    matches = st.session_state.matches_df.copy()
    idx = matches.index[matches["match_id"].astype(str) == str(match_id)]
    if len(idx) != 1:
        return
    row = matches.loc[idx[0]]
    team_a_score = int(row["team_a_score"])
    team_b_score = int(row["team_b_score"])
    if team_a_score == team_b_score:
        st.error("A match needs a winner. Scores cannot be tied.")
        return

    winner = "A" if team_a_score > team_b_score else "B"
    for col in ["winner", "status", "completed_at"]:
        matches[col] = matches[col].astype(object)
    matches.loc[idx, "winner"] = winner
    matches.loc[idx, "status"] = "Completed"
    if not str(matches.loc[idx[0], "completed_at"] or "").strip():
        matches.loc[idx, "completed_at"] = now_iso()
    storage.save("matches", matches)
    refresh_state()

    updated_row = st.session_state.matches_df[st.session_state.matches_df["match_id"].astype(str) == str(match_id)].iloc[0]
    apply_elo_for_match(match_id, winner, updated_row)



def _normalise_name_text(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _build_player_alias_map(players_df: pd.DataFrame) -> dict[str, str]:
    aliases: dict[str, str] = {}
    manual_aliases = {
        "viet": "Viet To",
        "viet to": "Viet To",
        "umar": "Umar Hussain",
        "umar hussain": "Umar Hussain",
        "rianur": "Rianur Rahman",
        "rian": "Rianur Rahman",
        "rianur rahman": "Rianur Rahman",
        "emad": "Emad Uddin",
        "emad uddin": "Emad Uddin",
        "ibrahim": "Ibrahim Yusuf",
        "ibrahim yusuf": "Ibrahim Yusuf",
        "salman": "Salman Ahmad",
        "salman ahmad": "Salman Ahmad",
        "tahmid": "Tahmid Khan",
        "tahmid khan": "Tahmid Khan",
        "morgan": "Morgan Chai",
        "morgan chai": "Morgan Chai",
    }
    name_to_id = {str(row["name"]): str(row["player_id"]) for _, row in players_df.iterrows()}
    for alias, canonical in manual_aliases.items():
        if canonical in name_to_id:
            aliases[_normalise_name_text(alias)] = name_to_id[canonical]
    for _, row in players_df.iterrows():
        pid = str(row["player_id"])
        name = str(row["name"])
        norm = _normalise_name_text(name)
        if norm:
            aliases[norm] = pid
        parts = norm.split()
        if parts:
            aliases.setdefault(parts[0], pid)
    return aliases


def _split_team_entry(team_text: str) -> list[str]:
    text = str(team_text or "").strip()
    text = re.sub(r"\b(an)\b", "and", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*&\s*|\s*/\s*|\s*\+\s*|\s*,\s*", " and ", text)
    parts = [p.strip() for p in re.split(r"\band\b", text, flags=re.IGNORECASE) if p.strip()]
    return parts or ([text] if text else [])


def _resolve_player_name(raw_name: str, alias_map: dict[str, str], players_lookup: dict[str, str]) -> tuple[str | None, str | None]:
    norm = _normalise_name_text(raw_name)
    if not norm:
        return None, "Blank player name"
    if norm in alias_map:
        return alias_map[norm], None
    close = get_close_matches(norm, list(alias_map.keys()), n=1, cutoff=0.82)
    if close:
        return alias_map[close[0]], None
    return None, f"Could not match player '{raw_name}'"


def _parse_score(score_raw: str) -> tuple[int | None, int | None, str | None]:
    text = str(score_raw or "").strip()
    match = re.search(r"(\d+)\s*[-–—:]\s*(\d+)", text)
    if not match:
        return None, None, f"Could not parse score '{score_raw}'"
    return int(match.group(1)), int(match.group(2)), None


def _parse_import_date(date_raw: str, default_year: int) -> tuple[str | None, str | None]:
    text = str(date_raw or "").strip()
    if not text:
        return None, "Missing date"
    parsed = pd.to_datetime(f"{text}-{default_year}", format="%d-%b-%Y", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(f"{text} {default_year}", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None, f"Could not parse date '{date_raw}'"
    return parsed.date().isoformat(), None


def _make_import_duplicate_key(row: dict) -> tuple:
    team_a = tuple(sorted(row.get("team_a_ids", [])))
    team_b = tuple(sorted(row.get("team_b_ids", [])))
    return (
        row.get("scheduled_date", ""),
        team_a,
        team_b,
        int(row.get("team_a_score", 0)),
        int(row.get("team_b_score", 0)),
        str(row.get("video_url", "") or "").strip(),
    )


def _existing_completed_match_keys(matches_df: pd.DataFrame) -> set[tuple]:
    keys: set[tuple] = set()
    if matches_df.empty:
        return keys
    for _, match in matches_df.iterrows():
        if str(match.get("status", "")).lower() != "completed":
            continue
        row = {
            "scheduled_date": str(match.get("scheduled_date", "") or ""),
            "team_a_ids": get_match_team_ids(match, "A"),
            "team_b_ids": get_match_team_ids(match, "B"),
            "team_a_score": int(match.get("team_a_score", 0) or 0),
            "team_b_score": int(match.get("team_b_score", 0) or 0),
            "video_url": str(match.get("video_url", "") or "").strip(),
        }
        keys.add(_make_import_duplicate_key(row))
    return keys


def parse_completed_matches_import(uploaded_df: pd.DataFrame, players_df: pd.DataFrame, matches_df: pd.DataFrame, players_lookup: dict[str, str], default_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["Date", "Team A", "Team B", "Score", "Link", "Match Type"]
    missing = [c for c in required if c not in uploaded_df.columns]
    if missing:
        return pd.DataFrame(), pd.DataFrame([{"row_number": "—", "error": f"Missing required columns: {', '.join(missing)}"}])

    alias_map = _build_player_alias_map(players_df)
    existing_keys = _existing_completed_match_keys(matches_df)
    seen_import_keys: set[tuple] = set()
    parsed_rows: list[dict] = []
    errors: list[dict] = []

    for row_index, row in uploaded_df.reset_index(drop=True).iterrows():
        row_number = int(row_index) + 2  # spreadsheet row number including header
        row_errors: list[str] = []
        match_type = str(row.get("Match Type", "") or "").strip().title()
        if match_type not in {"Singles", "Doubles"}:
            row_errors.append(f"Invalid Match Type '{row.get('Match Type')}'")

        scheduled_date, date_error = _parse_import_date(row.get("Date", ""), default_year)
        if date_error:
            row_errors.append(date_error)

        team_a_parts = _split_team_entry(row.get("Team A", ""))
        team_b_parts = _split_team_entry(row.get("Team B", ""))
        expected_players = 1 if match_type == "Singles" else 2
        if match_type in {"Singles", "Doubles"}:
            if len(team_a_parts) != expected_players:
                row_errors.append(f"Team A expected {expected_players} player(s), found {len(team_a_parts)}: {team_a_parts}")
            if len(team_b_parts) != expected_players:
                row_errors.append(f"Team B expected {expected_players} player(s), found {len(team_b_parts)}: {team_b_parts}")

        team_a_ids: list[str] = []
        team_b_ids: list[str] = []
        for raw_player in team_a_parts:
            pid, err = _resolve_player_name(raw_player, alias_map, players_lookup)
            if err:
                row_errors.append(err)
            elif pid:
                team_a_ids.append(pid)
        for raw_player in team_b_parts:
            pid, err = _resolve_player_name(raw_player, alias_map, players_lookup)
            if err:
                row_errors.append(err)
            elif pid:
                team_b_ids.append(pid)

        if set(team_a_ids) & set(team_b_ids):
            row_errors.append("Same player appears on both teams")

        team_a_score, team_b_score, score_error = _parse_score(row.get("Score", ""))
        if score_error:
            row_errors.append(score_error)
        elif team_a_score == team_b_score:
            row_errors.append("Completed match score cannot be tied")

        video_url = str(row.get("Link", "") or "").strip()
        if not video_url:
            row_errors.append("Missing YouTube/video link")

        parsed = {
            "spreadsheet_row": row_number,
            "scheduled_date": scheduled_date or "",
            "scheduled_time": f"00:{(row_index + 1) // 60:02d}:{(row_index + 1) % 60:02d}",
            "match_type": match_type,
            "team_a_raw": str(row.get("Team A", "") or ""),
            "team_b_raw": str(row.get("Team B", "") or ""),
            "team_a_ids": team_a_ids,
            "team_b_ids": team_b_ids,
            "team_a_names": " / ".join(players_lookup.get(pid, pid) for pid in team_a_ids),
            "team_b_names": " / ".join(players_lookup.get(pid, pid) for pid in team_b_ids),
            "team_a_score": team_a_score if team_a_score is not None else 0,
            "team_b_score": team_b_score if team_b_score is not None else 0,
            "winner": "A" if (team_a_score or 0) > (team_b_score or 0) else "B",
            "video_url": video_url,
            "source_score": str(row.get("Score", "") or ""),
            "duplicate": False,
        }
        if not row_errors:
            import_key = _make_import_duplicate_key(parsed)
            parsed["duplicate"] = import_key in existing_keys or import_key in seen_import_keys
            seen_import_keys.add(import_key)
        if row_errors:
            errors.append({"row_number": row_number, "error": "; ".join(row_errors)})
        parsed_rows.append(parsed)

    return pd.DataFrame(parsed_rows), pd.DataFrame(errors)


def import_completed_matches(parsed_df: pd.DataFrame) -> tuple[int, int]:
    require_editor()
    imported = 0
    skipped = 0
    for _, row in parsed_df.iterrows():
        if bool(row.get("duplicate", False)):
            skipped += 1
            continue
        match_id = str(uuid4())
        team_a_ids = list(row.get("team_a_ids", []))
        team_b_ids = list(row.get("team_b_ids", []))
        match_row = {
            "match_id": match_id,
            "created_at": now_iso(),
            "completed_at": now_iso(),
            "match_type": str(row.get("match_type", "")),
            "points_to_win": max(int(row.get("team_a_score", 0)), int(row.get("team_b_score", 0))),
            "team_a_players": "|".join(team_a_ids),
            "team_b_players": "|".join(team_b_ids),
            "team_a_score": int(row.get("team_a_score", 0)),
            "team_b_score": int(row.get("team_b_score", 0)),
            "winner": str(row.get("winner", "")),
            "status": "Completed",
            "video_url": str(row.get("video_url", "") or ""),
            "scheduled_date": str(row.get("scheduled_date", "") or ""),
            "scheduled_time": str(row.get("scheduled_time", "") or ""),
            "notes": f"Imported via CSV | Source row: {int(row.get('spreadsheet_row', 0))}",
        }
        storage.append_row("matches", match_row)
        for participant in create_match_participants_rows(match_id, team_a_ids, team_b_ids).to_dict("records"):
            storage.append_row("match_participants", participant)
        imported += 1
    refresh_state()
    if imported:
        recalculate_elo_history()
    return imported, skipped

def parse_date_str(raw: str) -> date:
    try:
        return date.fromisoformat(str(raw))
    except Exception:
        return date.today()


def save_clip_presets(start_text: str = "", end_text: str = "") -> None:
    st.session_state.clip_start = start_text
    st.session_state.clip_end = end_text


def add_match_display_columns(df: pd.DataFrame, players_lookup: dict[str, str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["team_a_names"] = out.apply(lambda row: " / ".join(players_lookup.get(pid, pid) for pid in get_match_team_ids(row, "A")), axis=1)
    out["team_b_names"] = out.apply(lambda row: " / ".join(players_lookup.get(pid, pid) for pid in get_match_team_ids(row, "B")), axis=1)
    out["winner_label"] = out.apply(
        lambda row: row["team_a_names"] if row["winner"] == "A" else (row["team_b_names"] if row["winner"] == "B" else ""),
        axis=1,
    )
    out["match_label"] = out.apply(
        lambda row: f"{row['scheduled_date'] or row['created_at'][:10]} | {row['team_a_names']} vs {row['team_b_names']} | {row['status']}",
        axis=1,
    )
    return out


def build_event_review_df(events_df: pd.DataFrame, matches_df: pd.DataFrame, players_lookup: dict[str, str]) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()

    review = events_df.copy()
    review["player"] = review["player_id"].astype(str).map(players_lookup)
    review["player"] = review.apply(lambda row: row["player"] if str(row.get("player", "")).strip() else f"Team {row['team']} (unattributed)", axis=1)
    review["clip_range"] = review.apply(
        lambda row: f"{row['video_start_label'] or format_seconds(row['video_start_seconds'])} → {row['video_end_label'] or format_seconds(row['video_end_seconds'])}",
        axis=1,
    )
    review["event_type_label"] = review["event_type"].astype(str).str.replace("_", " ").str.title()

    matches_view = add_match_display_columns(matches_df, players_lookup)
    match_cols = [
        "match_id", "match_label", "scheduled_date", "scheduled_time", "match_type", "points_to_win",
        "team_a_names", "team_b_names", "team_a_score", "team_b_score", "winner_label", "status", "notes", "video_url",
    ]
    if matches_view.empty:
        for col in match_cols:
            if col not in review.columns:
                review[col] = ""
        return review

    merged = review.merge(matches_view[match_cols], on="match_id", how="left")
    merged["match_info"] = merged.apply(
        lambda row: f"{row['scheduled_date'] or '—'} {row['scheduled_time'] or ''} | {row.get('team_a_names', '')} vs {row.get('team_b_names', '')}".strip(),
        axis=1,
    )
    return merged


def filter_events_by_player_name(df: pd.DataFrame, search_text: str, player_choice: str) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    if search_text:
        out = out[out["player"].astype(str).str.contains(search_text, case=False, na=False)]
    if player_choice and player_choice != "All players":
        out = out[out["player"] == player_choice]
    return out


def render_event_review_page(title: str, review_df: pd.DataFrame, event_type: str, players_lookup: dict[str, str]) -> None:
    st.subheader(title)
    subset = review_df[review_df["event_type"] == event_type].copy() if not review_df.empty else pd.DataFrame()
    if subset.empty:
        st.info(f"No {title.lower()} recorded yet.")
        return

    search_col, player_col = st.columns([1.2, 1])
    search_text = search_col.text_input(f"Search player in {title.lower()}", key=f"search_{event_type}", placeholder="Type part of a player name")
    player_options = ["All players"] + sorted(subset["player"].dropna().astype(str).unique().tolist())
    player_choice = player_col.selectbox("Filter by player", player_options, key=f"player_filter_{event_type}")
    subset = filter_events_by_player_name(subset, search_text, player_choice)

    if subset.empty:
        st.info("No events match that filter.")
        return

    subset = subset.sort_values(["scheduled_date", "scheduled_time", "event_index", "timestamp"], ascending=[False, False, True, False])
    st.caption(f"{len(subset)} clip(s)")
    st.dataframe(
        subset[[
            "player", "scheduled_date", "scheduled_time", "match_type", "team_a_names", "team_b_names",
            "team_a_score", "team_b_score", "winner_label", "status", "clip_range", "note", "clip_url"
        ]],
        use_container_width=True,
        column_config={
            "scheduled_date": "Match date",
            "scheduled_time": "Match time",
            "team_a_names": "Team A",
            "team_b_names": "Team B",
            "team_a_score": "A score",
            "team_b_score": "B score",
            "winner_label": "Winner",
            "clip_range": "Clip",
            "clip_url": st.column_config.LinkColumn("Clip link", display_text="Open clip"),
        },
        hide_index=True,
    )



def render_elo_history_page(players_df: pd.DataFrame, matches_df: pd.DataFrame, elo_history_df: pd.DataFrame, stats_df: pd.DataFrame, players_lookup: dict[str, str]) -> None:
    st.subheader("Elo leaderboard & history")
    if stats_df.empty:
        st.info("Add players and complete matches to see Elo.")
        return

    leaderboard = stats_df[["name", "elo", "matches_played", "wins", "losses", "win_rate"]].copy()
    top_cols = st.columns([1.1, 1.4])
    with top_cols[0]:
        st.markdown("#### Current leaderboard")
        st.dataframe(leaderboard.sort_values("elo", ascending=False), use_container_width=True, hide_index=True)
    with top_cols[1]:
        elo_bar = px.bar(
            leaderboard.sort_values("elo", ascending=True),
            x="elo",
            y="name",
            orientation="h",
            title="Current Elo by player",
            labels={"elo": "Current Elo", "name": "Player"},
        )
        elo_bar.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(elo_bar, use_container_width=True)

    st.markdown("#### Elo history")
    matches_view = add_match_display_columns(matches_df, players_lookup)
    timeline = build_elo_timeline_df(matches_view, elo_history_df, players_lookup)
    if timeline.empty:
        st.info("No Elo history yet. Complete a match to start building the timeline.")
        return

    # Defensive fallback: older analytics.py versions may not create these chart columns.
    # Keep the app running and still use match dates on the x-axis.
    if "recorded_at_dt" not in timeline.columns:
        timeline["recorded_at_dt"] = pd.to_datetime(timeline.get("recorded_at", ""), errors="coerce")
    if "match_date_dt" not in timeline.columns:
        timeline["match_date_dt"] = pd.to_datetime(timeline.get("scheduled_date", ""), errors="coerce")
        timeline["match_date_dt"] = timeline["match_date_dt"].fillna(timeline["recorded_at_dt"].dt.normalize())
    if "chart_x_dt" not in timeline.columns:
        order = timeline[["match_id", "match_date_dt", "recorded_at_dt"]].drop_duplicates("match_id").copy()
        order = order.sort_values(["match_date_dt", "recorded_at_dt", "match_id"])
        order["date_group"] = order["match_date_dt"].dt.date.astype(str)
        order["day_order"] = order.groupby("date_group").cumcount()
        order["day_count"] = order.groupby("date_group")["match_id"].transform("count")
        order["spread_minutes"] = 120 + ((order["day_order"] + 1) * (20 * 60 / (order["day_count"] + 1)))
        order["chart_x_dt"] = order["match_date_dt"].dt.normalize() + pd.to_timedelta(order["spread_minutes"], unit="m")
        timeline = timeline.merge(order[["match_id", "chart_x_dt"]], on="match_id", how="left")

    player_options = sorted(timeline["player"].dropna().astype(str).unique().tolist())
    default_players = player_options[: min(8, len(player_options))]
    selected_players = st.multiselect(
        "Show players",
        player_options,
        default=default_players,
        help="Filter players in or out to keep the chart readable.",
    )
    filtered = timeline[timeline["player"].isin(selected_players)].copy() if selected_players else timeline.iloc[0:0].copy()
    if filtered.empty:
        st.info("Select at least one player to show the Elo line chart.")
        return

    fig = go.Figure()
    x_title = "Match date (matches on same day spread horizontally)"
    for player_name, group in filtered.groupby("player", sort=True):
        group = group.sort_values(["chart_x_dt", "recorded_at_dt", "recorded_at", "match_id"])
        x_values = group["chart_x_dt"].fillna(group["match_date_dt"]).fillna(group["recorded_at_dt"])
        if x_values.isna().all():
            x_values = list(range(1, len(group) + 1))
            x_title = "Elo event order"
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=group["new_elo"],
                mode="lines+markers",
                name=player_name,
                customdata=group[["hover_details", "match_id"]],
                hovertemplate="%{customdata[0]}<extra></extra>",
                marker=dict(size=8, line=dict(width=1)),
            )
        )
    fig.update_layout(
        title="Elo movement over time",
        xaxis_title=x_title,
        yaxis_title="Elo rating",
        hovermode="closest",
        legend_title="Player",
        height=520,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Hover over a marker to see the match, score, winner and Elo change. Streamlit does not natively open pages from Plotly clicks, so use the selector below to inspect the exact match.")

    event_options = filtered.sort_values(["chart_x_dt", "recorded_at_dt", "recorded_at", "player"], ascending=[False, False, False, True]).copy()
    event_options["elo_event_label"] = event_options.apply(
        lambda r: f"{r.get('player', '')}: {r.get('delta_label', '')} Elo | {r.get('match_date_label', '')} | {r.get('team_a_names', '')} vs {r.get('team_b_names', '')} | {r.get('score_label', '')}",
        axis=1,
    )
    selected_event = st.selectbox("Inspect Elo event / match", event_options["elo_event_label"].tolist(), key="elo_event_inspector")
    if selected_event:
        event_row = event_options[event_options["elo_event_label"] == selected_event].iloc[0]
        st.markdown("##### Selected match summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Player Elo change", str(event_row.get("delta_label", "")))
        c2.metric("New Elo", f"{float(event_row.get('new_elo', BASE_ELO)):.0f}")
        c3.metric("Score", str(event_row.get("score_label", "")))
        c4.metric("Winner", str(event_row.get("winner_label", "—") or "—"))
        st.write(f"**Match:** {event_row.get('team_a_names', '')} vs {event_row.get('team_b_names', '')}")
        if str(event_row.get("video_url", "") or "").strip():
            st.markdown(f"**Video:** [Open video]({event_row.get('video_url')})")
        st.write(f"**Notes:** {event_row.get('notes', '') or '—'}")

    st.divider()
    render_player_elo_relationship_impact_page(players_df, matches_df, match_participants_df, elo_history_df, players_lookup)



def _option_id_from_name(players_df: pd.DataFrame, name: str, default: str = "") -> str:
    if not name or name.startswith("Any"):
        return default
    rows = players_df[players_df["name"].astype(str) == str(name)]
    if rows.empty:
        return default
    return str(rows.iloc[0]["player_id"])


def _id_pipe_contains(pipe_value: str, player_id: str) -> bool:
    if not player_id:
        return True
    return str(player_id) in [x for x in str(pipe_value or "").split("|") if x]


def render_player_elo_relationship_impact_page(players_df: pd.DataFrame, matches_df: pd.DataFrame, participants_df: pd.DataFrame, elo_history_df: pd.DataFrame, players_lookup: dict[str, str]) -> None:
    st.markdown("#### Relationship Elo impact")
    st.caption("Choose one player and optional relationship filters. The two charts use the same filtered games: per-match impact on the left, cumulative impact on the right.")

    if players_df.empty:
        st.info("Add players first.")
        return

    def _split_pipe_ids(value: str) -> list[str]:
        return [x for x in str(value or "").split("|") if x]

    def _names_from_id_series(series: pd.Series) -> list[str]:
        ids: set[str] = set()
        for value in series.fillna("").astype(str):
            ids.update(_split_pipe_ids(value))
        return sorted(players_lookup.get(pid, pid) for pid in ids if pid in players_lookup)

    def _keep_valid_selectbox_value(key: str, options: list[str], default: str) -> None:
        if st.session_state.get(key) not in options:
            st.session_state[key] = default

    name_options = players_df["name"].astype(str).sort_values().tolist()
    f1, f2 = st.columns([1.2, 1])
    evaluated_name = f1.selectbox("Player being evaluated", name_options, key="relationship_impact_player")
    evaluated_pid = _option_id_from_name(players_df, evaluated_name)
    format_filter = f2.selectbox("Format", ["All", "Singles", "Doubles"], key="relationship_impact_format")

    timeline = build_player_elo_relationship_timeline_df(evaluated_pid, matches_df, participants_df, elo_history_df, players_lookup)
    if timeline.empty:
        st.info("No completed matches with Elo history for this player yet.")
        return

    option_base = timeline.copy()
    if format_filter != "All":
        option_base = option_base[option_base["format"] == format_filter]

    if option_base.empty:
        st.warning("No games exist for this player with the selected format.")
        return

    partner_options = ["Any partner"] + _names_from_id_series(option_base["partner_ids"])
    _keep_valid_selectbox_value("relationship_impact_partner", partner_options, "Any partner")
    p_col, o1_col, o2_col = st.columns(3)
    partner_name = p_col.selectbox("Partner filter", partner_options, key="relationship_impact_partner")
    partner_id = _option_id_from_name(players_df, partner_name)

    opponent_option_base = option_base.copy()
    if partner_id:
        opponent_option_base = opponent_option_base[opponent_option_base["partner_ids"].apply(lambda value: _id_pipe_contains(value, partner_id))]

    opponent1_options = ["Any opponent"] + _names_from_id_series(opponent_option_base["opponent_ids"])
    _keep_valid_selectbox_value("relationship_impact_opp1", opponent1_options, "Any opponent")
    opponent1_name = o1_col.selectbox("Opponent 1 filter", opponent1_options, key="relationship_impact_opp1")
    opponent1_id = _option_id_from_name(players_df, opponent1_name)

    opponent2_option_base = opponent_option_base.copy()
    if opponent1_id:
        opponent2_option_base = opponent2_option_base[opponent2_option_base["opponent_ids"].apply(lambda value: _id_pipe_contains(value, opponent1_id))]
    opponent2_options_raw = _names_from_id_series(opponent2_option_base["opponent_ids"])
    opponent2_options_raw = [name for name in opponent2_options_raw if name != opponent1_name]
    opponent2_options = ["Any opponent"] + opponent2_options_raw
    _keep_valid_selectbox_value("relationship_impact_opp2", opponent2_options, "Any opponent")
    opponent2_name = o2_col.selectbox("Opponent 2 filter", opponent2_options, key="relationship_impact_opp2")
    opponent2_id = _option_id_from_name(players_df, opponent2_name)

    opponent_ids = [pid for pid in [opponent1_id, opponent2_id] if pid]

    filtered = option_base.copy()
    if partner_id:
        filtered = filtered[filtered["partner_ids"].apply(lambda value: _id_pipe_contains(value, partner_id))]
    for opponent_id in opponent_ids:
        filtered = filtered[filtered["opponent_ids"].apply(lambda value, oid=opponent_id: _id_pipe_contains(value, oid))]

    if filtered.empty:
        st.warning("No matches match those filters. Try removing one filter or switching format to All.")
        return

    filtered = filtered.sort_values(["chart_x_dt", "match_row_order", "match_id"]).copy()
    filtered["running_total"] = filtered["elo_delta"].cumsum()
    filtered["positive_elo"] = filtered["elo_delta"].where(filtered["elo_delta"] > 0)
    filtered["negative_elo"] = filtered["elo_delta"].where(filtered["elo_delta"] < 0)
    filtered["win_marker_y"] = filtered["elo_delta"].where(filtered["result"] == "Win")
    filtered["loss_marker_y"] = filtered["elo_delta"].where(filtered["result"] == "Loss")

    total_delta = float(filtered["elo_delta"].sum())
    avg_delta = float(filtered["elo_delta"].mean())
    wins = int((filtered["result"] == "Win").sum())
    matches = int(len(filtered))
    win_rate = (wins / matches * 100) if matches else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Net Elo impact", f"{total_delta:+.0f}")
    m2.metric("Average per game", f"{avg_delta:+.1f}")
    m3.metric("Win rate", f"{win_rate:.1f}%")
    m4.metric("Matches", matches)

    title_parts = [evaluated_name]
    if partner_id:
        title_parts.append(f"with {partner_name}")
    if opponent_ids:
        title_parts.append("vs " + " + ".join([players_lookup.get(pid, pid) for pid in opponent_ids]))
    if format_filter != "All":
        title_parts.append(f"({format_filter})")
    title_suffix = " ".join(title_parts)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        fig_match = go.Figure()
        fig_match.add_trace(
            go.Bar(
                x=filtered["chart_x_dt"],
                y=filtered["positive_elo"],
                name="Elo gained",
                marker_color="#16803c",
                customdata=filtered[["hover_details", "video_url"]],
                hovertemplate="%{customdata[0]}<extra></extra>",
            )
        )
        fig_match.add_trace(
            go.Bar(
                x=filtered["chart_x_dt"],
                y=filtered["negative_elo"],
                name="Elo lost",
                marker_color="#b91c1c",
                customdata=filtered[["hover_details", "video_url"]],
                hovertemplate="%{customdata[0]}<extra></extra>",
            )
        )
        fig_match.add_trace(
            go.Scatter(
                x=filtered["chart_x_dt"],
                y=filtered["win_marker_y"],
                name="Win",
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="#0f172a", line=dict(width=1, color="#ffffff")),
                customdata=filtered[["hover_details", "video_url"]],
                hovertemplate="%{customdata[0]}<extra></extra>",
            )
        )
        fig_match.add_trace(
            go.Scatter(
                x=filtered["chart_x_dt"],
                y=filtered["loss_marker_y"],
                name="Loss",
                mode="markers",
                marker=dict(symbol="x", size=10, color="#0f172a", line=dict(width=2, color="#0f172a")),
                customdata=filtered[["hover_details", "video_url"]],
                hovertemplate="%{customdata[0]}<extra></extra>",
            )
        )
        fig_match.add_hline(y=0, line_width=1, line_dash="dash", line_color="rgba(120,120,120,.7)")
        fig_match.update_layout(
            title="Per-match Elo impact",
            xaxis_title="Game date, ordered within day",
            yaxis_title="Elo change",
            legend_title="Legend",
            hovermode="closest",
            height=430,
            margin=dict(l=20, r=10, t=60, b=40),
            bargap=0.38,
        )
        st.plotly_chart(fig_match, use_container_width=True)

    with chart_cols[1]:
        fig_cum = go.Figure()
        fig_cum.add_trace(
            go.Scatter(
                x=filtered["chart_x_dt"],
                y=filtered["running_total"],
                name="Cumulative Elo impact",
                mode="lines+markers",
                line=dict(width=3, color="#2563eb"),
                marker=dict(size=8, color="#2563eb", line=dict(width=1, color="#ffffff")),
                customdata=filtered[["hover_details", "running_total", "video_url"]],
                hovertemplate="%{customdata[0]}<br>Cumulative impact: %{customdata[1]:+.0f}<extra></extra>",
            )
        )
        fig_cum.add_hline(y=0, line_width=1, line_dash="dash", line_color="rgba(120,120,120,.7)")
        fig_cum.update_layout(
            title="Cumulative Elo impact",
            xaxis_title="Game date, ordered within day",
            yaxis_title="Running total Elo",
            legend_title="Legend",
            hovermode="closest",
            height=430,
            margin=dict(l=20, r=10, t=60, b=40),
        )
        st.plotly_chart(fig_cum, use_container_width=True)

    st.caption("Both charts use the same filters. Left = what each game did to Elo. Right = how those games add up over time. Hover points/bars for score, format, partner/opponents and video URL.")

    review_cols = [
        "match_date_label", "format", "partner_names", "opponent_names", "result", "score_label",
        "elo_delta", "running_total", "old_elo", "new_elo", "video_url"
    ]
    with st.expander("Show matching games", expanded=False):
        st.dataframe(
            filtered[review_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "match_date_label": "Date",
                "partner_names": "Partner",
                "opponent_names": "Opponents",
                "score_label": "Score",
                "elo_delta": st.column_config.NumberColumn("Elo change", format="%+.0f"),
                "running_total": st.column_config.NumberColumn("Running total", format="%+.0f"),
                "old_elo": st.column_config.NumberColumn("Old Elo", format="%.0f"),
                "new_elo": st.column_config.NumberColumn("New Elo", format="%.0f"),
                "video_url": st.column_config.LinkColumn("Video", display_text="Open video"),
            },
        )


def render_partner_matrix_page(players_df: pd.DataFrame, matches_df: pd.DataFrame, participants_df: pd.DataFrame, elo_history_df: pd.DataFrame, players_lookup: dict[str, str]) -> None:
    st.subheader("Partner evaluation matrix")
    st.caption("Rows show the player being evaluated. Columns show their doubles partner. The cell colour is the average net Elo change per match for the row player when paired with that partner.")
    matrix_df = build_partner_matrix_df(matches_df, participants_df, elo_history_df, players_lookup)
    if matrix_df.empty:
        st.info("No doubles partner data yet. Complete doubles matches to build this matrix.")
        return

    players = sorted(set(matrix_df["player"].dropna().astype(str).tolist()) | set(matrix_df["partner"].dropna().astype(str).tolist()))
    selected = st.multiselect("Players to include", players, default=players[: min(12, len(players))])
    filtered = matrix_df[matrix_df["player"].isin(selected) & matrix_df["partner"].isin(selected)].copy() if selected else matrix_df.iloc[0:0].copy()
    if filtered.empty:
        st.info("Select at least two players with completed doubles matches together.")
        return

    z = filtered.pivot(index="player", columns="partner", values="avg_net_elo").reindex(index=selected, columns=selected)
    hover = filtered.pivot(index="player", columns="partner", values="hover").reindex(index=selected, columns=selected)
    for same in set(z.index).intersection(set(z.columns)):
        z.loc[same, same] = None
        hover.loc[same, same] = ""

    max_abs = float(pd.to_numeric(filtered["avg_net_elo"], errors="coerce").abs().max() or 1)
    fig = go.Figure(data=go.Heatmap(
        z=z.values,
        x=z.columns.tolist(),
        y=z.index.tolist(),
        text=hover.values,
        hovertemplate="%{text}<extra></extra>",
        colorscale="RdYlGn",
        zmid=0,
        zmin=-max_abs,
        zmax=max_abs,
        colorbar=dict(title="Avg Elo / match"),
    ))
    fig.update_layout(
        title="Doubles partner impact on Elo per match",
        xaxis_title="Partner",
        yaxis_title="Player",
        height=max(520, 42 * len(z.index)),
        margin=dict(l=80, r=20, t=70, b=80),
    )
    fig.update_xaxes(side="top")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Partner detail table")
    table = filtered.copy().sort_values(["player", "avg_net_elo"], ascending=[True, False])
    st.dataframe(
        table[["player", "partner", "matches", "wins", "losses", "win_rate", "avg_net_elo", "avg_elo_gained", "avg_elo_lost", "elo_gained", "elo_lost_abs", "net_elo"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "win_rate": st.column_config.NumberColumn("Win rate %", format="%.1f"),
            "avg_net_elo": st.column_config.NumberColumn("Avg net Elo / match", format="%+.1f"),
            "avg_elo_gained": st.column_config.NumberColumn("Avg Elo gained / match", format="+%.1f"),
            "avg_elo_lost": st.column_config.NumberColumn("Avg Elo lost / match", format="%.1f"),
            "elo_gained": st.column_config.NumberColumn("Total Elo gained", format="+%.0f"),
            "elo_lost_abs": st.column_config.NumberColumn("Total Elo lost", format="%.0f"),
            "net_elo": st.column_config.NumberColumn("Total net Elo", format="%+.0f"),
        },
    )


def render_player_relationship_highlights(selected_pid: str, matches_df: pd.DataFrame, participants_df: pd.DataFrame, elo_history_df: pd.DataFrame, players_lookup: dict[str, str]) -> None:
    insights = build_player_relationship_insights(selected_pid, matches_df, participants_df, elo_history_df, players_lookup)
    st.markdown("#### Relationship highlights")
    c1, c2, c3 = st.columns(3)
    worst = insights.get("worst_opponent")
    if worst:
        c1.metric("Lost most Elo against", str(worst.get("name", "—")), f"-{float(worst.get('elo_lost_abs', 0)):.0f} Elo")
        c1.caption(f"Across {int(worst.get('matches', 0))} match(es).")
    else:
        c1.metric("Lost most Elo against", "—")
        c1.caption("No negative opponent Elo impact yet.")

    best_partner = insights.get("best_partner")
    if best_partner:
        c2.metric("Best Elo partner", str(best_partner.get("name", "—")), f"+{float(best_partner.get('net_elo', 0)):.0f} Elo")
        c2.caption(f"Win rate {float(best_partner.get('win_rate', 0)):.1f}% over {int(best_partner.get('matches', 0))} match(es).")
    else:
        c2.metric("Best Elo partner", "—")
        c2.caption("No positive partner Elo impact yet.")

    best_wr = insights.get("best_partner_win_rate")
    if best_wr:
        c3.metric("Best partner win rate", str(best_wr.get("name", "—")), f"{float(best_wr.get('win_rate', 0)):.1f}%")
        c3.caption(f"{int(best_wr.get('wins', 0))}/{int(best_wr.get('matches', 0))} wins, net {float(best_wr.get('net_elo', 0)):+.0f} Elo.")
    else:
        c3.metric("Best partner win rate", "—")
        c3.caption("No doubles partner history yet.")



def _build_loss_risk_surface(risk_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Create a smooth IDW surface for the player's doubles score-performance chart."""
    x = pd.to_numeric(risk_df["x_partner_relative_elo"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(risk_df["y_opponent_relative_elo"], errors="coerce").to_numpy(dtype=float)
    z_col = "score_performance_pp" if "score_performance_pp" in risk_df.columns else "loss_risk_pp"
    z = pd.to_numeric(risk_df[z_col], errors="coerce").to_numpy(dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[valid], y[valid], z[valid]

    if len(x) == 0:
        axis_limit = 200.0
        max_abs_z = 10.0
        grid_axis = np.linspace(-axis_limit, axis_limit, 45)
        xx, yy = np.meshgrid(grid_axis, grid_axis)
        zz = np.zeros_like(xx)
        return xx, yy, zz, axis_limit, max_abs_z

    axis_limit = max(100.0, float(np.nanmax(np.abs(np.concatenate([x, y])))) + 40.0)
    axis_limit = min(max(axis_limit, 160.0), 500.0)
    grid_axis = np.linspace(-axis_limit, axis_limit, 70)
    xx, yy = np.meshgrid(grid_axis, grid_axis)

    if len(x) == 1:
        zz = np.full_like(xx, z[0], dtype=float)
    else:
        # Inverse-distance weighting: nearby historical matches influence the colour more.
        dx = xx[..., None] - x[None, None, :]
        dy = yy[..., None] - y[None, None, :]
        dist_sq = dx * dx + dy * dy
        weights = 1.0 / np.power(dist_sq + 900.0, 1.15)
        zz = np.sum(weights * z[None, None, :], axis=2) / np.sum(weights, axis=2)

    max_abs_z = max(5.0, float(np.nanmax(np.abs(z))))
    return xx, yy, zz, axis_limit, max_abs_z


def render_player_loss_risk_heatmap(
    selected_pid: str,
    selected_name: str,
    matches_df: pd.DataFrame,
    participants_df: pd.DataFrame,
    elo_history_df: pd.DataFrame,
    players_lookup: dict[str, str],
) -> None:
    st.markdown("#### Performance map: better or worse than expected")
    timeline = build_player_match_timeline_df(selected_pid, matches_df, participants_df, elo_history_df, players_lookup)
    if timeline.empty:
        st.info("No completed match performance map yet for this player.")
        return

    chart_df = timeline.copy()
    chart_df["result_symbol"] = chart_df["result"].map({"Win": "circle", "Loss": "x"}).fillna("circle")
    chart_df["elo_impact_size"] = chart_df["elo_delta"].abs().clip(lower=6, upper=28) + 8
    chart_df["performance_band"] = pd.cut(
        chart_df["score_performance_pp"],
        bins=[-999, -10, -3, 3, 10, 999],
        labels=["Major underperformance", "Slight underperformance", "About expected", "Slight overperformance", "Major overperformance"],
    ).astype(str)

    max_x = max(100.0, float(chart_df["opponent_relative_elo"].abs().max() or 0) + 40)
    max_y = max(18.0, float(chart_df["score_performance_pp"].abs().max() or 0) + 5)
    max_y = min(max_y, 50.0)
    max_color = max(12.0, float(chart_df["score_performance_pp"].abs().max() or 0))

    st.caption(
        "Each dot is one match. Up = you scored better than Elo expected. Down = you scored worse than Elo expected. "
        "Right = harder match against stronger opponents. Left = easier match where you were favoured. "
        "Circle = win, X = loss, bigger marker = bigger Elo swing."
    )

    fig = go.Figure()

    # Background quadrant shading, deliberately subtle and labelled in plain English.
    fig.add_shape(type="rect", x0=-max_x, x1=0, y0=0, y1=max_y, fillcolor="rgba(46, 204, 113, 0.10)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0, x1=max_x, y0=0, y1=max_y, fillcolor="rgba(0, 128, 0, 0.13)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=-max_x, x1=0, y0=-max_y, y1=0, fillcolor="rgba(231, 76, 60, 0.15)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0, x1=max_x, y0=-max_y, y1=0, fillcolor="rgba(230, 126, 34, 0.12)", line_width=0, layer="below")

    for result, symbol, name in [("Win", "circle", "Wins"), ("Loss", "x", "Losses")]:
        subset = chart_df[chart_df["result"] == result]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["opponent_relative_elo"],
                y=subset["score_performance_pp"],
                mode="markers",
                name=name,
                marker=dict(
                    symbol=symbol,
                    size=subset["elo_impact_size"],
                    color=subset["score_performance_pp"],
                    colorscale=[
                        [0.00, "#b2182b"],
                        [0.35, "#ef8a62"],
                        [0.50, "#f7f7f7"],
                        [0.65, "#67a9cf"],
                        [1.00, "#2166ac"],
                    ],
                    cmin=-max_color,
                    cmax=max_color,
                    colorbar=dict(title="Performance<br>vs expected"),
                    line=dict(width=1.2, color="rgba(0,0,0,0.65)"),
                    opacity=0.9,
                ),
                customdata=subset[[
                    "match_date_label", "partner_names", "opponent_names", "score_label", "result",
                    "expected_win_rate", "actual_score_share", "score_performance_pp", "elo_delta", "performance_band",
                ]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Partner(s): %{customdata[1]}<br>"
                    "Opponent(s): %{customdata[2]}<br>"
                    "Score: %{customdata[3]}<br>"
                    "Result: %{customdata[4]}<br>"
                    "Match difficulty: %{x:+.0f} Elo<br>"
                    "Expected score share: %{customdata[5]:.1f}%<br>"
                    "Actual score share: %{customdata[6]:.1f}%<br>"
                    "Performance: %{customdata[7]:+.1f} pp (%{customdata[9]})<br>"
                    "Elo change: %{customdata[8]:+.0f}<extra></extra>"
                ),
            )
        )

    fig.add_hline(y=0, line_width=2, line_color="rgba(0,0,0,0.65)", annotation_text="as expected", annotation_position="bottom right")
    fig.add_vline(x=0, line_width=2, line_color="rgba(0,0,0,0.55)", annotation_text="even matchup", annotation_position="top")

    label_box = dict(bgcolor="rgba(255,255,255,0.86)", bordercolor="rgba(0,0,0,0.20)", borderwidth=1)
    fig.add_annotation(x=-max_x * 0.55, y=max_y * 0.72, showarrow=False, text="<b>Handled business</b><br>Favoured and scored well", **label_box)
    fig.add_annotation(x=max_x * 0.55, y=max_y * 0.72, showarrow=False, text="<b>Punching up</b><br>Hard match, strong performance", **label_box)
    fig.add_annotation(x=-max_x * 0.55, y=-max_y * 0.72, showarrow=False, text="<b>Slip-up zone</b><br>Favoured but underperformed", **label_box)
    fig.add_annotation(x=max_x * 0.55, y=-max_y * 0.72, showarrow=False, text="<b>Outmatched</b><br>Hard match and struggled", **label_box)

    fig.update_layout(
        title=f"{selected_name}: match performance vs Elo expectation",
        xaxis_title="Match difficulty: opponent team Elo minus your team Elo before the match",
        yaxis_title="Score performance vs expected (percentage points)",
        height=620,
        margin=dict(l=20, r=20, t=75, b=75),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(range=[-max_x, max_x], zeroline=False)
    fig.update_yaxes(range=[-max_y, max_y], zeroline=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### How to interpret this")
    c1, c2, c3 = st.columns(3)
    c1.info("**High dots** = you scored better than Elo expected. **Low dots** = underperformed.")
    c2.info("**Right side** = harder matches. **Left side** = matches you were expected to do well in.")
    c3.info("**Blue/green-ish dots** = better than expected. **Red/orange dots** = worse than expected.")

    st.markdown("##### Matches worth reviewing")
    review_rows = []
    if not chart_df.empty:
        best = chart_df.sort_values("score_performance_pp", ascending=False).head(3)
        worst = chart_df.sort_values("score_performance_pp", ascending=True).head(3)
        elo_loss = chart_df.sort_values("elo_delta", ascending=True).head(3)
        for reason, subset in [
            ("Best overperformance", best),
            ("Biggest underperformance", worst),
            ("Biggest Elo loss", elo_loss),
        ]:
            for _, r in subset.iterrows():
                review_rows.append({
                    "reason": reason,
                    "date": r.get("match_date_label", ""),
                    "partner": r.get("partner_names", ""),
                    "opponents": r.get("opponent_names", ""),
                    "score": r.get("score_label", ""),
                    "result": r.get("result", ""),
                    "performance_pp": round(float(r.get("score_performance_pp", 0)), 1),
                    "elo_delta": round(float(r.get("elo_delta", 0)), 0),
                    "video_url": r.get("video_url", ""),
                    "match_id": r.get("match_id", ""),
                })
    review_df = pd.DataFrame(review_rows).drop_duplicates(subset=["reason", "match_id"]) if review_rows else pd.DataFrame()
    if review_df.empty:
        st.info("No review-priority matches yet.")
    else:
        st.dataframe(
            review_df[["reason", "date", "partner", "opponents", "score", "result", "performance_pp", "elo_delta", "video_url", "match_id"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "performance_pp": st.column_config.NumberColumn("Perf vs expected pp", format="%+.1f"),
                "elo_delta": st.column_config.NumberColumn("Elo", format="%+.0f"),
                "video_url": st.column_config.LinkColumn("Video", display_text="Open video"),
            },
        )


def render_player_form_and_score_charts(
    selected_pid: str,
    selected_name: str,
    matches_df: pd.DataFrame,
    participants_df: pd.DataFrame,
    elo_history_df: pd.DataFrame,
    players_lookup: dict[str, str],
) -> None:
    st.markdown("#### Score performance and clutch")
    timeline = build_player_match_timeline_df(selected_pid, matches_df, participants_df, elo_history_df, players_lookup)
    if timeline.empty:
        st.info("No completed match timeline yet for this player.")
        return

    # Expected vs actual score share
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timeline["expected_win_rate"],
        y=timeline["actual_score_share"],
        mode="markers",
        marker=dict(
            size=10,
            color=timeline["score_performance_pp"],
            colorscale="RdYlGn",
            cmin=-max(10, float(timeline["score_performance_pp"].abs().max() or 10)),
            cmax=max(10, float(timeline["score_performance_pp"].abs().max() or 10)),
            colorbar=dict(title="Score<br>perf pp"),
            line=dict(width=1, color="rgba(0,0,0,0.45)"),
        ),
        customdata=timeline[["match_date_label", "partner_names", "opponent_names", "score_label", "result", "score_performance_pp", "elo_delta"]],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Partner(s): %{customdata[1]}<br>"
            "Opponent(s): %{customdata[2]}<br>"
            "Score: %{customdata[3]}<br>"
            "Result: %{customdata[4]}<br>"
            "Expected score share: %{x:.1f}%<br>"
            "Actual score share: %{y:.1f}%<br>"
            "Performance: %{customdata[5]:+.1f} pp<br>"
            "Elo change: %{customdata[6]:+.0f}<extra></extra>"
        ),
        name="Matches",
    ))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(color="rgba(0,0,0,0.45)", dash="dash"), name="As expected"))
    fig.update_layout(
        title=f"{selected_name}: actual score share vs Elo expectation",
        xaxis_title="Expected score share from Elo (%)",
        yaxis_title="Actual score share (%)",
        height=430,
        margin=dict(l=20, r=20, t=70, b=50),
    )
    fig.update_xaxes(range=[0, 100])
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Above the dashed line = scored better than Elo expected. Below the line = scored worse than expected.")

    clutch = build_player_clutch_summary_df(timeline)
    if not clutch.empty:
        st.markdown("#### Clutch: close-game win rate")
        clutch_fig = px.bar(
            clutch,
            x="category",
            y="win_rate",
            text="win_rate",
            title="Close games vs other games",
            labels={"category": "Game type", "win_rate": "Win rate %"},
        )
        clutch_fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        clutch_fig.update_layout(height=360, margin=dict(l=15, r=15, t=60, b=45), yaxis_range=[0, 100])
        st.plotly_chart(clutch_fig, use_container_width=True)
        st.dataframe(
            clutch[["category", "matches", "wins", "win_rate", "net_elo", "avg_score_performance"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "win_rate": st.column_config.NumberColumn("Win rate %", format="%.1f"),
                "net_elo": st.column_config.NumberColumn("Net Elo", format="%+.0f"),
                "avg_score_performance": st.column_config.NumberColumn("Avg score perf pp", format="%+.1f"),
            },
        )


def render_player_partner_chemistry_chart(
    selected_pid: str,
    selected_name: str,
    matches_df: pd.DataFrame,
    participants_df: pd.DataFrame,
    elo_history_df: pd.DataFrame,
    players_lookup: dict[str, str],
) -> None:
    st.markdown("#### Partner chemistry")
    partner_df = build_partner_matrix_df(matches_df, participants_df, elo_history_df, players_lookup)
    if partner_df.empty:
        st.info("No doubles partner chemistry data yet.")
        return
    player_partners = partner_df[partner_df["player_id"].astype(str) == str(selected_pid)].copy()
    if player_partners.empty:
        st.info("No completed doubles matches with partners yet for this player.")
        return
    player_partners = player_partners.sort_values("avg_net_elo", ascending=True)
    fig = px.bar(
        player_partners,
        x="avg_net_elo",
        y="partner",
        orientation="h",
        title=f"{selected_name}: average Elo impact by partner",
        labels={"avg_net_elo": "Avg Elo per match", "partner": "Partner"},
        hover_data={"matches": True, "win_rate": ':.1f', "net_elo": ':+.0f', "avg_net_elo": ':+.1f'},
    )
    fig.add_vline(x=0, line_dash="dash", line_color="rgba(0,0,0,0.45)")
    fig.update_layout(height=max(360, 36 * len(player_partners)), margin=dict(l=15, r=15, t=65, b=45))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Bars to the right mean this partner has historically gained you Elo on average. Bars to the left mean you have lost Elo with them on average.")


def render_player_replacement_benchmark(
    selected_pid: str,
    selected_name: str,
    matches_df: pd.DataFrame,
    participants_df: pd.DataFrame,
    elo_history_df: pd.DataFrame,
    players_lookup: dict[str, str],
) -> None:
    st.markdown("#### Same setup benchmark")
    st.caption(
        "This asks: with the same partner against the same opponent pair, did you do better or worse than the other players who took your place?"
    )

    setup_options = build_player_context_setup_options(selected_pid, matches_df, participants_df, players_lookup)
    if setup_options.empty:
        st.info("No exact doubles setup benchmark yet. You need completed doubles matches where the same partner/opponent setup appears.")
        return

    setup_options = setup_options.sort_values(["selected_match_count", "setup_label"], ascending=[False, True]).copy()
    setup_options["display_label"] = setup_options.apply(
        lambda r: f"{r['setup_label']} — you played this {int(r['selected_match_count'])} time(s)",
        axis=1,
    )
    selected_setup_label = st.selectbox(
        "Setup to compare",
        setup_options["display_label"].tolist(),
        key=f"replacement_setup_{selected_pid}",
    )
    setup_row = setup_options[setup_options["display_label"] == selected_setup_label].iloc[0]
    partner_id = str(setup_row["partner_id"])
    opponent_ids = str(setup_row["opponent_key"]).split("|")

    benchmark = build_replacement_benchmark_df(selected_pid, partner_id, opponent_ids, matches_df, participants_df, elo_history_df, players_lookup)
    if benchmark.empty:
        st.info("No benchmark matches found for this exact setup yet.")
        return

    summary = summarise_replacement_benchmark(benchmark)
    if summary.empty:
        st.info("No benchmark summary available yet.")
        return

    selected_summary = summary[summary["candidate_id"].astype(str) == str(selected_pid)]
    others = summary[summary["candidate_id"].astype(str) != str(selected_pid)]
    selected_avg = float(selected_summary["avg_performance_pp"].iloc[0]) if not selected_summary.empty else 0.0
    others_avg = float(others["avg_performance_pp"].mean()) if not others.empty else 0.0
    replacement_value = selected_avg - others_avg
    selected_matches = int(selected_summary["matches"].iloc[0]) if not selected_summary.empty else 0

    verdict = "Better than replacements" if replacement_value > 1 else ("Worse than replacements" if replacement_value < -1 else "About the same")
    verdict_delta = f"{replacement_value:+.1f} pp"

    m1, m2, m3 = st.columns(3)
    m1.metric("You in this setup", f"{selected_avg:+.1f} pp", f"{selected_matches} match(es)")
    m2.metric("Others in your place", f"{others_avg:+.1f} pp", f"{len(others)} replacement player(s)")
    m3.metric("Verdict", verdict, verdict_delta)

    st.caption(
        "Positive numbers mean the player scored better than Elo expected. Negative numbers mean they scored worse than Elo expected. "
        "The verdict compares you against the average of everyone else in the same role."
    )

    summary = summary.sort_values("avg_performance_pp", ascending=True).copy()
    summary["label"] = summary.apply(
        lambda r: f"⭐ {r['candidate']}" if str(r["candidate_id"]) == str(selected_pid) else str(r["candidate"]),
        axis=1,
    )

    fig = go.Figure()
    colors = summary["candidate_id"].astype(str).apply(lambda pid: "#2563eb" if pid == str(selected_pid) else "#9ca3af")
    fig.add_trace(go.Bar(
        x=summary["avg_performance_pp"],
        y=summary["label"],
        orientation="h",
        marker_color=colors,
        text=summary["avg_performance_pp"].apply(lambda x: f"{x:+.1f} pp"),
        textposition="outside",
        customdata=summary[["matches", "wins", "losses", "win_rate", "avg_elo", "total_elo", "replacement_value_pp"]],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Avg performance: %{x:+.1f} pp<br>"
            "Matches: %{customdata[0]}<br>"
            "Wins/Losses: %{customdata[1]}/%{customdata[2]}<br>"
            "Win rate: %{customdata[3]:.1f}%<br>"
            "Avg Elo change: %{customdata[4]:+.1f}<br>"
            "Total Elo: %{customdata[5]:+.0f}<extra></extra>"
        ),
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="rgba(0,0,0,0.45)")
    fig.update_layout(
        title=f"Who performs best with {setup_row['partner']} vs {setup_row['opponents']}?",
        xaxis_title="Average score performance vs expected",
        yaxis_title="Player in your place",
        height=max(360, 52 * len(summary)),
        margin=dict(l=20, r=45, t=70, b=45),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    compact = summary.sort_values("avg_performance_pp", ascending=False).copy()
    compact["player"] = compact["candidate"]
    compact["avg_perf"] = compact["avg_performance_pp"]
    compact["avg_elo_change"] = compact["avg_elo"]
    st.dataframe(
        compact[["player", "matches", "win_rate", "avg_perf", "avg_elo_change"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "player": "Player",
            "matches": "Matches",
            "win_rate": st.column_config.NumberColumn("Win rate %", format="%.1f"),
            "avg_perf": st.column_config.NumberColumn("Avg performance", format="%+.1f pp"),
            "avg_elo_change": st.column_config.NumberColumn("Avg Elo", format="%+.1f"),
        },
    )

    with st.expander("Show match-level details", expanded=False):
        match_table = benchmark.sort_values(["candidate", "match_date", "match_id"]).copy()
        st.dataframe(
            match_table[["candidate", "match_date_label", "score_label", "result", "score_performance_pp", "elo_delta", "video_url"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "candidate": "Player",
                "match_date_label": "Date",
                "score_label": "Score",
                "score_performance_pp": st.column_config.NumberColumn("Performance", format="%+.1f pp"),
                "elo_delta": st.column_config.NumberColumn("Elo", format="%+.0f"),
                "video_url": st.column_config.LinkColumn("Video", display_text="Open video"),
            },
        )


def render_player_head_to_head_matrix(
    selected_pid: str,
    selected_name: str,
    matches_df: pd.DataFrame,
    participants_df: pd.DataFrame,
    elo_history_df: pd.DataFrame,
    players_lookup: dict[str, str],
) -> None:
    st.markdown("#### Head-to-head opponent matrix")
    h2h = build_player_head_to_head_df(selected_pid, matches_df, participants_df, elo_history_df, players_lookup)
    if h2h.empty:
        st.info("No head-to-head opponent data yet.")
        return

    h2h = h2h.sort_values(["win_rate", "matches", "net_elo"], ascending=[False, False, False]).copy()
    z = [h2h["win_rate"].tolist()]
    text = [h2h["hover"].tolist()]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=h2h["opponent"].tolist(),
        y=[selected_name],
        text=text,
        hovertemplate="%{text}<extra></extra>",
        colorscale="RdYlGn",
        zmin=0,
        zmax=100,
        colorbar=dict(title="Win rate %"),
    ))
    fig.update_layout(
        title=f"{selected_name}: head-to-head win rate against opponents",
        xaxis_title="Opponent",
        yaxis_title="Player",
        height=280,
        margin=dict(l=70, r=20, t=70, b=100),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        h2h[["opponent", "matches", "wins", "losses", "win_rate", "elo_gained", "elo_lost_abs", "net_elo"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "win_rate": st.column_config.NumberColumn("Win rate %", format="%.1f"),
            "elo_gained": st.column_config.NumberColumn("Elo gained", format="+%.0f"),
            "elo_lost_abs": st.column_config.NumberColumn("Elo lost", format="%.0f"),
            "net_elo": st.column_config.NumberColumn("Net Elo", format="%+.0f"),
        },
    )

def render_player_explorer(players_df: pd.DataFrame, stats_df: pd.DataFrame, review_df: pd.DataFrame, matches_df: pd.DataFrame, players_lookup: dict[str, str]) -> None:
    st.subheader("Player Explorer")
    if players_df.empty:
        st.info("Add players first.")
        return

    search_text = st.text_input("Search player", key="player_explorer_search", placeholder="Type a player name")
    names = players_df["name"].astype(str).sort_values().tolist()
    filtered_names = [name for name in names if search_text.lower() in name.lower()] if search_text else names
    if not filtered_names:
        st.info("No players match that search.")
        return

    selected_name = st.selectbox("Select player", filtered_names, key="player_explorer_select")
    selected_row = players_df[players_df["name"] == selected_name].iloc[0]
    selected_pid = str(selected_row["player_id"])

    stat_row = stats_df[stats_df["player_id"] == selected_pid]
    if not stat_row.empty:
        stat_row = stat_row.iloc[0]
        metric_cols = st.columns(6)
        metric_cols[0].metric("Elo", f"{float(stat_row['elo']):.0f}")
        metric_cols[1].metric("Matches", int(stat_row["matches_played"]))
        metric_cols[2].metric("Wins", int(stat_row["wins"]))
        metric_cols[3].metric("Points", int(stat_row["points_won"]))
        metric_cols[4].metric("Good / Bad", f"{int(stat_row['good_shots'])} / {int(stat_row['bad_shots'])}")
        metric_cols[5].metric("Highlights", int(stat_row["highlights"]))

    render_player_relationship_highlights(selected_pid, matches_df, match_participants_df, elo_history_df, players_lookup)
    render_player_form_and_score_charts(selected_pid, selected_name, matches_df, match_participants_df, elo_history_df, players_lookup)
    render_player_partner_chemistry_chart(selected_pid, selected_name, matches_df, match_participants_df, elo_history_df, players_lookup)
    render_player_loss_risk_heatmap(selected_pid, selected_name, matches_df, match_participants_df, elo_history_df, players_lookup)
    render_player_replacement_benchmark(selected_pid, selected_name, matches_df, match_participants_df, elo_history_df, players_lookup)
    render_player_head_to_head_matrix(selected_pid, selected_name, matches_df, match_participants_df, elo_history_df, players_lookup)

    player_events = review_df[review_df["player_id"].astype(str) == selected_pid].copy() if not review_df.empty else pd.DataFrame()
    if not match_participants_df.empty:
        player_match_ids = match_participants_df.loc[match_participants_df["player_id"].astype(str) == selected_pid, "match_id"].astype(str).unique().tolist()
        player_matches = matches_df[matches_df["match_id"].astype(str).isin(player_match_ids)].copy()
    else:
        player_matches = matches_df[
            matches_df["team_a_players"].fillna("").astype(str).str.contains(selected_pid)
            | matches_df["team_b_players"].fillna("").astype(str).str.contains(selected_pid)
        ].copy()
    player_matches = add_match_display_columns(player_matches, players_lookup)

    st.markdown("#### Matches")
    if player_matches.empty:
        st.info("No matches recorded for this player yet.")
    else:
        st.dataframe(
            player_matches[[
                "scheduled_date", "scheduled_time", "match_type", "team_a_names", "team_b_names",
                "team_a_score", "team_b_score", "winner_label", "status", "video_url", "notes"
            ]].sort_values(["scheduled_date", "scheduled_time"], ascending=[False, False]),
            use_container_width=True,
            column_config={"video_url": st.column_config.LinkColumn("Video", display_text="Open video")},
            hide_index=True,
        )

    st.markdown("#### Events")
    if player_events.empty:
        st.info("No events logged for this player yet.")
        return

    event_filter_col, note_filter_col = st.columns([1, 1.4])
    event_types = ["All"] + sorted(player_events["event_type_label"].dropna().unique().tolist())
    event_filter = event_filter_col.selectbox("Event type", event_types, key="player_explorer_event_type")
    note_search = note_filter_col.text_input("Search event notes", key="player_explorer_note_search", placeholder="Optional note text")

    if event_filter != "All":
        player_events = player_events[player_events["event_type_label"] == event_filter]
    if note_search:
        player_events = player_events[player_events["note"].astype(str).str.contains(note_search, case=False, na=False)]

    if player_events.empty:
        st.info("No player events match that filter.")
        return

    summary = player_events.groupby("event_type_label", dropna=False).size().reset_index(name="count").sort_values("count", ascending=False)
    summary_chart = px.bar(summary, x="event_type_label", y="count", title=f"{selected_name} events by type")
    st.plotly_chart(summary_chart, use_container_width=True)

    st.dataframe(
        player_events[[
            "scheduled_date", "scheduled_time", "event_index", "event_type_label", "team", "clip_range",
            "team_a_names", "team_b_names", "note", "clip_url"
        ]].sort_values(["scheduled_date", "scheduled_time", "event_index"], ascending=[False, False, True]),
        use_container_width=True,
        column_config={
            "scheduled_date": "Match date",
            "scheduled_time": "Match time",
            "event_type_label": "Event",
            "team_a_names": "Team A",
            "team_b_names": "Team B",
            "clip_range": "Clip",
            "clip_url": st.column_config.LinkColumn("Clip link", display_text="Open clip"),
        },
        hide_index=True,
    )


def render_top_summary(players_df: pd.DataFrame, matches_df: pd.DataFrame, review_df: pd.DataFrame) -> None:
    total_players = 0 if players_df.empty else int(players_df["player_id"].nunique())
    total_matches = 0 if matches_df.empty else int(matches_df["match_id"].nunique())
    live_matches = 0 if matches_df.empty else int(matches_df["status"].eq("In Progress").sum())
    total_clips = 0 if review_df.empty else int(review_df[review_df["clip_url"].astype(str).ne("")]["event_id"].nunique())
    top_cols = st.columns(4)
    top_cols[0].metric("Players", total_players)
    top_cols[1].metric("Matches", total_matches)
    top_cols[2].metric("Live matches", live_matches)
    top_cols[3].metric("Tagged clips", total_clips)


with st.sidebar:
    st.markdown("### Session")
    st.write(f"Signed in as: {get_display_name()}")
    st.write(f"Role: {current_role}")
    st.button("Log out", on_click=logout, use_container_width=True)

    st.markdown("### Quick add player")
    with st.form("add_player_form", clear_on_submit=True):
        new_player_name = st.text_input("Player name", placeholder="Add a new player")
        submitted = st.form_submit_button("Save player", use_container_width=True)
        if submitted:
            require_editor()
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

    with st.expander("About this app", expanded=False):
        st.write("Use Live Match to tag clips, Matches to review one game in depth, and Player Explorer to trace one player across all games.")
        st.write("Tip: embed a YouTube link on the match and tag every event with a clip range so you can review it later.")


players_lookup = player_name_map()
stats_df = build_player_stats(players_df, matches_df, events_df, elo_history_df, match_participants_df)
review_df = build_event_review_df(events_df, matches_df, players_lookup)
render_top_summary(players_df, matches_df, review_df)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "🎬 Live Match", "🏅 Elo", "📊 Stats", "🤝 Partners", "🗂️ Matches", "✅ Good Shots", "❌ Bad Shots", "⭐ Highlights", "🧍 Player Explorer", "📅 Match History", "🛠️ Admin"
])

with tab1:
    st.subheader("Live Match Recorder")
    active_match = get_active_match()

    if active_match is None:
        st.info("No live match. Create one below.")
        if players_df.empty:
            st.warning("Add players first.")
        elif current_role != "admin":
            st.info("General Viewer mode: only admins can create matches.")
        else:
            create_mode = st.radio(
                "How do you want to save this match?",
                ["Create live match for annotation", "Save completed match with final score"],
                horizontal=False,
                key="create_match_mode",
                help="Choose completed match first if you already know the final score. The score boxes will appear before you save.",
            )
            if create_mode == "Save completed match with final score":
                st.info("Enter the final score below. This will save the result, update Elo immediately, and still let you annotate clips later from Matches.")

            with st.form("create_match_form", clear_on_submit=False):
                match_type = st.selectbox("Match type", ["Singles", "Doubles"])
                points_to_win = st.number_input("Points to win", min_value=1, max_value=99, value=21)
                names = players_df["name"].sort_values().tolist()
                team_a = st.multiselect("Team A", names, max_selections=2)
                team_b = st.multiselect("Team B", names, max_selections=2)
                create_cols = st.columns(2)
                match_date = create_cols[0].date_input("Match date", value=date.today())
                match_time = create_cols[1].time_input("Match time")
                video_url = st.text_input("YouTube video URL (optional)")
                notes = st.text_area("Match notes", placeholder="Venue, lineup notes, injuries, tactics, anything useful")
                final_a_score = 0
                final_b_score = 0
                if create_mode == "Save completed match with final score":
                    score_cols = st.columns(2)
                    final_a_score = score_cols[0].number_input("Final Team A score", min_value=0, max_value=99, value=21, key="final_team_a_score")
                    final_b_score = score_cols[1].number_input("Final Team B score", min_value=0, max_value=99, value=0, key="final_team_b_score")
                create_match = st.form_submit_button("Save match", type="primary")

                if not create_match:
                    # Reset the guard once the form is no longer actively submitting. This lets
                    # admins create another match later, but protects the first submit/rerun cycle.
                    st.session_state["create_match_submit_in_progress"] = False

                if create_match:
                    require_editor()
                    expected = 1 if match_type == "Singles" else 2
                    if len(team_a) != expected or len(team_b) != expected:
                        st.error(f"{match_type} requires {expected} player(s) on each side.")
                    elif set(team_a) & set(team_b):
                        st.error("A player cannot be on both sides.")
                    elif create_mode == "Save completed match with final score" and int(final_a_score) == int(final_b_score):
                        st.error("Completed matches need a winner, so the final scores cannot be tied.")
                    else:
                        create_signature = "|".join([
                            str(match_type),
                            str(points_to_win),
                            "~".join(sorted(team_a)),
                            "~".join(sorted(team_b)),
                            str(match_date),
                            str(match_time),
                            (video_url or "").strip(),
                            (notes or "").strip(),
                            str(create_mode),
                            str(int(final_a_score)) if create_mode == "Save completed match with final score" else "",
                            str(int(final_b_score)) if create_mode == "Save completed match with final score" else "",
                        ])

                        if st.session_state.get("create_match_submit_in_progress") and st.session_state.get("last_create_match_signature") == create_signature:
                            st.warning("Duplicate submit ignored. The match was already saved.")
                            st.stop()

                        st.session_state["create_match_submit_in_progress"] = True
                        st.session_state["last_create_match_signature"] = create_signature

                        name_to_id = {row["name"]: str(row["player_id"]) for _, row in players_df.iterrows()}
                        match_id_new = str(uuid4())
                        is_completed = create_mode == "Save completed match with final score"
                        winner = ""
                        if is_completed:
                            winner = "A" if int(final_a_score) > int(final_b_score) else "B"
                        row = {
                            "match_id": match_id_new,
                            "created_at": now_iso(),
                            "completed_at": now_iso() if is_completed else "",
                            "match_type": match_type,
                            "points_to_win": int(points_to_win),
                            "team_a_players": "|".join(name_to_id[n] for n in team_a),
                            "team_b_players": "|".join(name_to_id[n] for n in team_b),
                            "team_a_score": int(final_a_score) if is_completed else 0,
                            "team_b_score": int(final_b_score) if is_completed else 0,
                            "winner": winner,
                            "status": "Completed" if is_completed else "In Progress",
                            "video_url": (video_url or "").strip(),
                            "scheduled_date": str(match_date),
                            "scheduled_time": str(match_time),
                            "notes": (notes or "").strip(),
                        }
                        storage.append_row("matches", row)
                        participants_payload = pd.concat([st.session_state.match_participants_df, create_match_participants_rows(row["match_id"], parse_players(row["team_a_players"]), parse_players(row["team_b_players"]))], ignore_index=True)
                        storage.save("match_participants", participants_payload)
                        refresh_state()
                        if is_completed:
                            created_match = st.session_state.matches_df[st.session_state.matches_df["match_id"].astype(str) == match_id_new].iloc[0]
                            apply_elo_for_match(match_id_new, winner, created_match)
                            st.success("Completed match saved and Elo updated. You can annotate it later from Matches.")
                        else:
                            st.success("Live match created.")
                        st.rerun()
    else:
        team_a_ids = get_match_team_ids(active_match, "A")
        team_b_ids = get_match_team_ids(active_match, "B")
        team_a_names = [players_lookup.get(pid, pid) for pid in team_a_ids]
        team_b_names = [players_lookup.get(pid, pid) for pid in team_b_ids]
        all_live_names = team_a_names + team_b_names
        match_id = str(active_match["match_id"])
        active_video_url = str(active_match.get("video_url", "") or "")

        a_col, mid_col, b_col = st.columns([1, 0.8, 1])
        with a_col:
            st.markdown("### Team A")
            st.markdown(" / ".join(team_a_names))
            st.metric("Score", int(active_match["team_a_score"]))
        with mid_col:
            st.markdown("### Match")
            st.write(f"{active_match['match_type']} · First to {int(active_match['points_to_win'])}")
            st.write(f"Date: {active_match.get('scheduled_date', '') or '—'}")
            st.write(f"Time: {active_match.get('scheduled_time', '') or '—'}")
        with b_col:
            st.markdown("### Team B")
            st.markdown(" / ".join(team_b_names))
            st.metric("Score", int(active_match["team_b_score"]))

        details_expander = st.expander("Match details and video", expanded=True)
        with details_expander:
            if active_video_url:
                render_youtube_video(active_video_url)
                st.markdown(f"[Open video in YouTube]({active_video_url})")
            else:
                st.info("No YouTube video URL saved for this match yet.")

            meta1, meta2 = st.columns(2)
            new_date = meta1.date_input("Match date", value=parse_date_str(active_match.get("scheduled_date", "")), key=f"date_{match_id}")
            new_time = meta2.text_input("Match time", value=str(active_match.get("scheduled_time", "") or ""), key=f"time_{match_id}", placeholder="19:30")
            new_video = st.text_input("YouTube URL", value=active_video_url, key=f"video_{match_id}")
            new_notes = st.text_area("Match notes", value=str(active_match.get("notes", "") or ""), key=f"notes_{match_id}", height=100)
            if st.button("Save match details", disabled=current_role != "admin", use_container_width=True):
                require_editor()
                update_match_details(match_id, video_url=new_video, scheduled_date=str(new_date), scheduled_time=new_time, notes=new_notes)
                st.success("Match details saved.")
                st.rerun()

        st.markdown("#### Clip tagging console")
        st.caption("Enter a clip start and end for each event so you can generate clips later.")
        default_start = st.session_state.get("clip_start", "")
        default_end = st.session_state.get("clip_end", "")
        clip_cols = st.columns([1.3, 1.3, 0.8, 0.8, 0.9])
        clip_start_raw = clip_cols[0].text_input("Clip start", value=default_start, placeholder="01:24", key="clip_start_input")
        clip_end_raw = clip_cols[1].text_input("Clip end", value=default_end, placeholder="01:31", key="clip_end_input")
        start_seconds = parse_time_to_seconds(clip_start_raw)
        end_seconds = parse_time_to_seconds(clip_end_raw)
        clip_cols[2].metric("Start", format_seconds(start_seconds) if start_seconds is not None else "Invalid")
        clip_cols[3].metric("End", format_seconds(end_seconds) if end_seconds is not None else "Invalid")
        duration_label = "—"
        if start_seconds is not None and end_seconds is not None and end_seconds >= start_seconds:
            duration_label = f"{end_seconds - start_seconds}s"
        clip_cols[4].metric("Clip length", duration_label)

        selector_cols = st.columns([1.2, 1, 1])
        selected_player_name = selector_cols[0].selectbox("Selected player", all_live_names, key="selected_player_name")
        selected_team = "A" if selected_player_name in team_a_names else "B"
        selected_pid = next(pid for pid, name in players_lookup.items() if name == selected_player_name)
        scorer_a_options = ["Team point (unattributed)"] + team_a_names
        scorer_b_options = ["Team point (unattributed)"] + team_b_names
        scorer_a = selector_cols[1].selectbox("Who scored for Team A?", scorer_a_options, key="scorer_a")
        scorer_b = selector_cols[2].selectbox("Who scored for Team B?", scorer_b_options, key="scorer_b")

        valid_clip = start_seconds is not None and end_seconds is not None and end_seconds >= start_seconds
        if not valid_clip:
            st.warning("Enter a valid clip start and end. Examples: 84, 01:24, 00:01:24")
        if current_role != "admin":
            st.info("General Viewer mode: editing controls are disabled.")

        event_note = st.text_input("Event note", key="event_note", placeholder="Optional note, for example cross-court winner or service error under pressure")
        point_btn_cols = st.columns(2)
        if point_btn_cols[0].button("+1 Team A point", use_container_width=True, disabled=current_role != "admin"):
            require_editor()
            if not valid_clip:
                st.error("Enter a valid clip start and end first.")
            else:
                scorer_id = next((pid for pid, name in players_lookup.items() if name == scorer_a), None)
                record_event(match_id, "A", scorer_id, "point", points_awarded=1, video_start_seconds=start_seconds, video_end_seconds=end_seconds)
                save_clip_presets(format_seconds(end_seconds), format_seconds(end_seconds))
                st.rerun()
        if point_btn_cols[1].button("+1 Team B point", use_container_width=True, disabled=current_role != "admin"):
            require_editor()
            if not valid_clip:
                st.error("Enter a valid clip start and end first.")
            else:
                scorer_id = next((pid for pid, name in players_lookup.items() if name == scorer_b), None)
                record_event(match_id, "B", scorer_id, "point", points_awarded=1, video_start_seconds=start_seconds, video_end_seconds=end_seconds)
                save_clip_presets(format_seconds(end_seconds), format_seconds(end_seconds))
                st.rerun()

        event_cols = st.columns(5)
        event_specs = [
            ("Good shot", "good_shot"),
            ("Bad shot", "bad_shot"),
            ("Service fault", "service_fault"),
            ("Highlight", "highlight"),
            ("Undo last event", "undo"),
        ]
        for col, (label, code) in zip(event_cols, event_specs):
            if col.button(label, use_container_width=True, disabled=current_role != "admin"):
                require_editor()
                if code == "undo":
                    undo_last_event(match_id)
                    st.rerun()
                elif not valid_clip:
                    st.error("Enter a valid clip start and end first.")
                else:
                    record_event(match_id, selected_team, selected_pid, code, note=event_note, video_start_seconds=start_seconds, video_end_seconds=end_seconds)
                    save_clip_presets(format_seconds(end_seconds), format_seconds(end_seconds))
                    st.rerun()

        controls = st.columns(2)
        reached_limit = int(active_match["team_a_score"]) >= int(active_match["points_to_win"]) or int(active_match["team_b_score"]) >= int(active_match["points_to_win"])
        if controls[0].button("Complete match", disabled=(not reached_limit) or current_role != "admin", use_container_width=True):
            require_editor()
            complete_match(match_id)
            st.success("Match completed and Elo updated.")
            st.rerun()
        if controls[1].button("Force complete now", disabled=current_role != "admin", use_container_width=True):
            require_editor()
            complete_match(match_id)
            st.success("Match completed and Elo updated.")
            st.rerun()

        event_log = get_match_events(match_id).copy()
        if not event_log.empty:
            event_log["player"] = event_log["player_id"].astype(str).map(players_lookup)
            event_log["player"] = event_log.apply(lambda row: row["player"] if str(row.get("player", "")).strip() else f"Team {row['team']} (unattributed)", axis=1)
            event_log["clip_range"] = event_log.apply(lambda row: f"{row['video_start_label']} → {row['video_end_label']}".strip(" →"), axis=1)
            st.markdown("#### Event log")
            st.dataframe(
                event_log[["event_index", "timestamp", "team", "player", "event_type", "points_awarded", "clip_range", "note", "clip_url"]],
                use_container_width=True,
                column_config={
                    "clip_url": st.column_config.LinkColumn("Jump link", display_text="Open clip"),
                },
                hide_index=True,
            )

with tab2:
    render_elo_history_page(players_df, matches_df, elo_history_df, stats_df, players_lookup)

with tab3:
    st.subheader("Player Stats")
    if stats_df.empty:
        st.info("No stats yet.")
    else:
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        metric_choice = st.selectbox("Chart", ["points_won", "good_shots", "bad_shots", "service_faults", "highlights", "shot_balance"])
        chart_df = stats_df[["name", metric_choice]].sort_values(metric_choice, ascending=False)
        stat_chart = px.bar(chart_df, x="name", y=metric_choice, title=f"{metric_choice.replace('_', ' ').title()} by player")
        st.plotly_chart(stat_chart, use_container_width=True)

with tab4:
    render_partner_matrix_page(players_df, matches_df, match_participants_df, elo_history_df, players_lookup)

with tab5:
    st.subheader("Matches")
    matches_view = add_match_display_columns(matches_df, players_lookup)
    if matches_view.empty:
        st.info("No matches yet.")
    else:
        default_idx = len(matches_view) - 1
        selected_label = st.selectbox(
            "Select a match",
            matches_view["match_label"].tolist(),
            index=default_idx,
            key="match_detail_selector",
        )
        selected_match = matches_view[matches_view["match_label"] == selected_label].iloc[-1]
        selected_match_id = str(selected_match["match_id"])
        selected_events = get_match_events(selected_match_id).copy()

        summary_cols = st.columns(4)
        summary_cols[0].metric("Status", selected_match["status"])
        summary_cols[1].metric("Score", f"{selected_match['team_a_score']} - {selected_match['team_b_score']}")
        summary_cols[2].metric("Format", selected_match["match_type"])
        summary_cols[3].metric("Target", int(selected_match["points_to_win"]))

        st.markdown("#### Match details")
        details_left, details_right = st.columns(2)
        details_left.write(f"**Team A:** {selected_match['team_a_names']}")
        details_left.write(f"**Team B:** {selected_match['team_b_names']}")
        details_left.write(f"**Winner:** {selected_match['winner_label'] or '—'}")
        details_left.write(f"**Scheduled date:** {selected_match['scheduled_date'] or '—'}")
        details_left.write(f"**Scheduled time:** {selected_match['scheduled_time'] or '—'}")
        details_right.write(f"**Created:** {selected_match['created_at'] or '—'}")
        details_right.write(f"**Completed:** {selected_match['completed_at'] or '—'}")
        if str(selected_match.get('video_url', '') or '').strip():
            details_right.markdown(f"**Video:** [Open video]({selected_match['video_url']})")
        else:
            details_right.write("**Video:** —")
        st.write(f"**Notes:** {selected_match['notes'] or '—'}")

        st.markdown("#### Add annotations to this match")
        if current_role != "admin":
            st.info("General Viewer mode: annotations are read-only.")
        else:
            ann_team_a_ids = get_match_team_ids(selected_match, "A")
            ann_team_b_ids = get_match_team_ids(selected_match, "B")
            ann_team_a_names = [players_lookup.get(pid, pid) for pid in ann_team_a_ids]
            ann_team_b_names = [players_lookup.get(pid, pid) for pid in ann_team_b_ids]
            ann_all_names = ann_team_a_names + ann_team_b_names

            with st.form(f"annotate_match_{selected_match_id}"):
                ann_cols = st.columns([1, 1, 1.2])
                ann_start_raw = ann_cols[0].text_input("Clip start", placeholder="01:24", key=f"ann_start_{selected_match_id}")
                ann_end_raw = ann_cols[1].text_input("Clip end", placeholder="01:31", key=f"ann_end_{selected_match_id}")
                ann_event_type = ann_cols[2].selectbox(
                    "Event type",
                    ["good_shot", "bad_shot", "service_fault", "highlight", "point"],
                    format_func=lambda x: x.replace("_", " ").title(),
                    key=f"ann_type_{selected_match_id}",
                )

                ann_player_options = ["Team A (unattributed)", "Team B (unattributed)"] + ann_all_names
                ann_player_choice = st.selectbox("Player / team", ann_player_options, key=f"ann_player_{selected_match_id}")
                ann_note = st.text_input("Annotation note", placeholder="Optional note about the clip", key=f"ann_note_{selected_match_id}")
                submitted_annotation = st.form_submit_button("Add annotation", use_container_width=True)

                if submitted_annotation:
                    require_editor()
                    ann_start_seconds = parse_time_to_seconds(ann_start_raw)
                    ann_end_seconds = parse_time_to_seconds(ann_end_raw)
                    if ann_start_seconds is None or ann_end_seconds is None or ann_end_seconds < ann_start_seconds:
                        st.error("Enter a valid clip start and end. Examples: 84, 01:24, 00:01:24")
                    else:
                        if ann_player_choice == "Team A (unattributed)":
                            ann_team = "A"
                            ann_pid = ""
                        elif ann_player_choice == "Team B (unattributed)":
                            ann_team = "B"
                            ann_pid = ""
                        else:
                            ann_team = "A" if ann_player_choice in ann_team_a_names else "B"
                            ann_pid = next((pid for pid, name in players_lookup.items() if name == ann_player_choice), "")

                        points_awarded = 1 if ann_event_type == "point" else 0
                        record_event(
                            selected_match_id,
                            ann_team,
                            ann_pid,
                            ann_event_type,
                            points_awarded=points_awarded,
                            note=ann_note,
                            video_start_seconds=ann_start_seconds,
                            video_end_seconds=ann_end_seconds,
                            update_match_score=False,
                        )
                        st.success("Annotation added. Final score was not changed.")
                        st.rerun()

        st.markdown("#### Event data")
        if selected_events.empty:
            st.info("No events recorded for this match yet.")
        else:
            selected_events = build_event_review_df(selected_events, matches_df, players_lookup)
            st.dataframe(
                selected_events[[
                    "event_index", "timestamp", "team", "player", "event_type_label", "points_awarded",
                    "video_start_seconds", "video_end_seconds", "video_start_label", "video_end_label",
                    "clip_range", "note", "clip_url"
                ]],
                use_container_width=True,
                column_config={
                    "event_type_label": "Event",
                    "clip_url": st.column_config.LinkColumn("Jump link", display_text="Open clip"),
                },
                hide_index=True,
            )
            csv_bytes = selected_events.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Export this match's events (.csv)",
                data=csv_bytes,
                file_name=f"match_{selected_match_id}_events.csv",
                mime="text/csv",
            )

with tab6:
    render_event_review_page("Good Shots", review_df, "good_shot", players_lookup)

with tab7:
    render_event_review_page("Bad Shots", review_df, "bad_shot", players_lookup)

with tab8:
    render_event_review_page("Highlights", review_df, "highlight", players_lookup)

with tab9:
    render_player_explorer(players_df, stats_df, review_df, matches_df, players_lookup)


with tab10:
    st.subheader("Match History")
    history = matches_df.copy()
    if history.empty:
        st.info("No matches yet.")
    else:
        history = add_match_display_columns(history, players_lookup)
        history_display = history[[
            "scheduled_date", "scheduled_time", "match_type", "points_to_win", "team_a_names", "team_b_names",
            "team_a_score", "team_b_score", "winner_label", "status", "video_url", "notes"
        ]]
        st.dataframe(
            history_display,
            use_container_width=True,
            column_config={"video_url": st.column_config.LinkColumn("Video URL", display_text="Open video")},
            hide_index=True,
        )

        csv = history_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download match history CSV",
            data=csv,
            file_name="match_history.csv",
            mime="text/csv",
            key="download_match_history_csv",
        )

        completed = history[history["status"] == "Completed"].copy()
        if not completed.empty:
            completed["played_on"] = completed["scheduled_date"].replace("", pd.NA).fillna(completed["completed_at"].astype(str).str[:10])
            match_chart = px.histogram(completed, x="played_on", title="Matches completed over time")
            st.plotly_chart(match_chart, use_container_width=True)


with tab11:
    st.subheader("Admin tools")
    if current_role != "admin":
        st.info("General Viewer mode: admin tools are hidden.")
    else:
        st.markdown("### Recalculate Elo")
        st.write("Use this after deleting duplicate matches or changing match results. It clears Elo history and rebuilds it from completed matches in chronological order.")
        completed_count = int(matches_df[
            matches_df["status"].astype(str).str.lower().eq("completed")
            & matches_df["winner"].astype(str).isin(["A", "B"])
        ].shape[0]) if not matches_df.empty else 0
        st.caption(f"Completed matches eligible for Elo rebuild: {completed_count}")
        confirm_recalc = st.checkbox("I understand this will replace the existing Elo history", key="confirm_recalculate_elo")
        if st.button("Recalculate Elo from completed matches", disabled=not confirm_recalc, type="primary", use_container_width=True):
            processed = recalculate_elo_history()
            st.success(f"Elo recalculated from {processed} completed match(es).")
            st.rerun()

        st.divider()
        st.markdown("### Import completed matches from CSV")
        st.write("Upload a spreadsheet with columns: Date, Team A, Team B, Score, Link, Match Type. Matches are imported in spreadsheet order as completed matches, then Elo is rebuilt.")
        import_year = st.number_input("Year for dates like 04-Apr", min_value=2020, max_value=2100, value=date.today().year, step=1)
        uploaded_scores = st.file_uploader("Upload completed match CSV", type=["csv"], key="completed_match_csv_import")
        if uploaded_scores is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_scores)
                parsed_df, import_errors = parse_completed_matches_import(uploaded_df, players_df, matches_df, players_lookup, int(import_year))
                c1, c2, c3 = st.columns(3)
                c1.metric("Rows found", len(uploaded_df))
                c2.metric("Ready to import", int((~parsed_df.get("duplicate", pd.Series(dtype=bool))).sum()) if not parsed_df.empty else 0)
                c3.metric("Duplicates skipped", int(parsed_df.get("duplicate", pd.Series(dtype=bool)).sum()) if not parsed_df.empty else 0)

                if not import_errors.empty:
                    st.error("Fix these rows before importing.")
                    st.dataframe(import_errors, use_container_width=True, hide_index=True)
                elif parsed_df.empty:
                    st.warning("No valid rows found to import.")
                else:
                    preview = parsed_df.copy()
                    preview["team_a_ids"] = preview["team_a_ids"].apply(lambda x: "|".join(x) if isinstance(x, list) else x)
                    preview["team_b_ids"] = preview["team_b_ids"].apply(lambda x: "|".join(x) if isinstance(x, list) else x)
                    st.markdown("#### Import preview")
                    st.dataframe(
                        preview[[
                            "spreadsheet_row", "scheduled_date", "scheduled_time", "match_type", "team_a_names", "team_b_names",
                            "team_a_score", "team_b_score", "winner", "duplicate", "video_url"
                        ]],
                        use_container_width=True,
                        hide_index=True,
                        column_config={"video_url": st.column_config.LinkColumn("Video", display_text="Open")},
                    )
                    confirm_import = st.checkbox("I have reviewed the preview and want to import these completed matches", key="confirm_completed_csv_import")
                    importable_count = int((~parsed_df["duplicate"]).sum())
                    if st.button("Import completed matches and rebuild Elo", disabled=(not confirm_import or importable_count == 0), type="primary", use_container_width=True):
                        imported, skipped = import_completed_matches(parsed_df)
                        st.success(f"Imported {imported} match(es), skipped {skipped} duplicate(s), and rebuilt Elo.")
                        st.rerun()
            except Exception as exc:
                st.error(f"Could not parse/import this CSV: {exc}")
