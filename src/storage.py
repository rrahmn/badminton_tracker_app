from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import math
import pandas as pd


DATA_FILES: Dict[str, list[str]] = {
    "players": ["player_id", "name", "created_at", "is_active"],
    "matches": [
        "match_id",
        "created_at",
        "completed_at",
        "match_type",
        "points_to_win",
        "team_a_players",
        "team_b_players",
        "team_a_score",
        "team_b_score",
        "winner",
        "status",
        "video_url",
        "scheduled_date",
        "scheduled_time",
        "notes",
    ],
    "events": [
        "event_id",
        "match_id",
        "timestamp",
        "event_index",
        "team",
        "player_id",
        "event_type",
        "points_awarded",
        "note",
        "video_start_seconds",
        "video_end_seconds",
        "video_start_label",
        "video_end_label",
        "clip_url",
    ],
    "elo_history": [
        "history_id",
        "match_id",
        "player_id",
        "old_elo",
        "new_elo",
        "delta",
        "recorded_at",
        "elo_model_version",
        "k_factor_used",
    ],
    "match_participants": [
        "match_id",
        "player_id",
        "team",
        "slot",
    ],
}

DB_MATCH_COLUMNS = [
    "match_id",
    "created_at",
    "completed_at",
    "match_type",
    "points_to_win",
    "team_a_score",
    "team_b_score",
    "winner",
    "status",
    "video_url",
    "match_date",
    "match_time",
    "notes",
]

PRIMARY_KEYS: Dict[str, list[str]] = {
    "players": ["player_id"],
    "matches": ["match_id"],
    "events": ["event_id"],
    "elo_history": ["history_id"],
    "match_participants": ["match_id", "player_id", "team", "slot"],
}


class CSVStorage:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_files()

    def _ensure_files(self) -> None:
        for name, columns in DATA_FILES.items():
            path = self.data_dir / f"{name}.csv"
            if not path.exists():
                pd.DataFrame(columns=columns).to_csv(path, index=False)

    def load(self, name: str) -> pd.DataFrame:
        path = self.data_dir / f"{name}.csv"
        df = pd.read_csv(path)
        expected = DATA_FILES[name]
        for column in expected:
            if column not in df.columns:
                df[column] = None
        return df[expected]

    def save(self, name: str, df: pd.DataFrame) -> None:
        path = self.data_dir / f"{name}.csv"
        df.to_csv(path, index=False)

    def append_row(self, name: str, row: dict) -> None:
        df = self.load(name)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        self.save(name, df)

    def replace_all(self, payload: dict[str, pd.DataFrame]) -> None:
        for name, df in payload.items():
            self.save(name, df)


def _pythonize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rows.append({col: _pythonize_scalar(row[col]) for col in df.columns})
    return rows


def _normalize_db_nulls(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    nullable_text_blank_to_none: dict[str, set[str]] = {
        "matches": {"completed_at", "video_url", "notes", "winner", "status", "match_type", "match_date", "match_time"},
        "events": {"note", "player_id", "video_start_label", "video_end_label", "clip_url", "team", "event_type"},
        "elo_history": {"elo_model_version"},
        "players": set(),
        "match_participants": set(),
    }
    integer_fields: dict[str, set[str]] = {
        "matches": {"points_to_win", "team_a_score", "team_b_score"},
        "events": {"event_index", "points_awarded", "video_start_seconds", "video_end_seconds"},
        "elo_history": {"old_elo", "new_elo", "delta", "k_factor_used"},
        "match_participants": {"slot"},
    }
    for col in nullable_text_blank_to_none.get(name, set()):
        if normalized.get(col) == "":
            normalized[col] = None
    # typed / optional columns that must never be sent as empty strings
    for col in ["created_at", "completed_at", "timestamp", "recorded_at", "match_date", "match_time"]:
        if col in normalized and normalized.get(col) == "":
            normalized[col] = None

    for col in integer_fields.get(name, set()):
        if col not in normalized:
            continue
        value = normalized.get(col)
        if value in (None, ""):
            normalized[col] = None
            continue
        try:
            normalized[col] = int(float(value))
        except (TypeError, ValueError):
            pass
    return normalized


class SupabaseStorage:
    def __init__(self, url: str, key: str):
        try:
            from supabase import create_client
        except ImportError as exc:
            raise RuntimeError(
                "Supabase storage requires the 'supabase' package. Add it to requirements.txt and reinstall."
            ) from exc

        self.url = url
        self.key = key
        self.client = create_client(url, key)

    def _select_columns(self, name: str) -> str:
        if name == "matches":
            return ",".join(DB_MATCH_COLUMNS)
        return ",".join(DATA_FILES[name])

    def _chunked(self, records: list[dict[str, Any]], size: int = 500) -> list[list[dict[str, Any]]]:
        return [records[i : i + size] for i in range(0, len(records), size)]

    def _fetch_all_records(self, name: str) -> list[dict[str, Any]]:
        columns = self._select_columns(name)
        start = 0
        page_size = 1000
        rows: list[dict[str, Any]] = []
        while True:
            response = (
                self.client.table(name)
                .select(columns)
                .range(start, start + page_size - 1)
                .execute()
            )
            batch = response.data or []
            rows.extend(batch)
            if len(batch) < page_size:
                break
            start += page_size
        return rows

    def load(self, name: str) -> pd.DataFrame:
        if name == "matches":
            match_rows = self._fetch_all_records("matches")
            participants_rows = self._fetch_all_records("match_participants")
            df = pd.DataFrame(match_rows)
            if df.empty:
                df = pd.DataFrame(columns=DB_MATCH_COLUMNS)
            if "match_date" in df.columns:
                df = df.rename(columns={"match_date": "scheduled_date", "match_time": "scheduled_time"})
            participants_df = pd.DataFrame(participants_rows)
            team_map: dict[str, dict[str, str]] = {}
            if not participants_df.empty:
                for team in ["A", "B"]:
                    subset = participants_df[participants_df["team"] == team].copy()
                    if not subset.empty:
                        subset["slot"] = pd.to_numeric(subset["slot"], errors="coerce").fillna(0).astype(int)
                        grouped = subset.sort_values(["match_id", "slot"]).groupby("match_id")["player_id"].apply(lambda s: "|".join(str(x) for x in s if str(x)))
                        for match_id, players in grouped.items():
                            team_map.setdefault(str(match_id), {})[team] = players
            df["team_a_players"] = df.get("match_id", pd.Series(dtype=str)).astype(str).map(lambda mid: team_map.get(mid, {}).get("A", ""))
            df["team_b_players"] = df.get("match_id", pd.Series(dtype=str)).astype(str).map(lambda mid: team_map.get(mid, {}).get("B", ""))
            for column in DATA_FILES[name]:
                if column not in df.columns:
                    df[column] = None
            return df[DATA_FILES[name]]

        rows = self._fetch_all_records(name)
        df = pd.DataFrame(rows)
        for column in DATA_FILES[name]:
            if column not in df.columns:
                df[column] = None
        if df.empty:
            return pd.DataFrame(columns=DATA_FILES[name])
        return df[DATA_FILES[name]]

    def _delete_missing_rows(self, name: str, desired_df: pd.DataFrame) -> None:
        if name == "match_participants":
            # The database table has a synthetic `id` primary key, while the app uses a logical
            # composite key. Replace the table contents wholesale to keep the app schema unchanged.
            self.client.table(name).delete().neq("match_id", "__never__").execute()
            records = _df_to_records(desired_df[DATA_FILES[name]])
            if records:
                for batch in self._chunked(records):
                    self.client.table(name).insert(batch).execute()
            return

        pk_cols = PRIMARY_KEYS[name]
        current_df = self.load(name)
        desired_keys = {
            tuple(str(_pythonize_scalar(row[col]) or "") for col in pk_cols)
            for _, row in desired_df[pk_cols].iterrows()
        }
        current_keys = {
            tuple(str(_pythonize_scalar(row[col]) or "") for col in pk_cols)
            for _, row in current_df[pk_cols].iterrows()
        }
        keys_to_delete = current_keys - desired_keys
        for key_tuple in keys_to_delete:
            query = self.client.table(name).delete()
            for col, value in zip(pk_cols, key_tuple):
                query = query.eq(col, value)
            query.execute()

    def save(self, name: str, df: pd.DataFrame) -> None:
        normalized = df.copy()
        for col in DATA_FILES[name]:
            if col not in normalized.columns:
                normalized[col] = None
        normalized = normalized[DATA_FILES[name]]

        self._delete_missing_rows(name, normalized)
        if name == "match_participants":
            return

        if name == "matches":
            db_df = normalized.drop(columns=["team_a_players", "team_b_players"], errors="ignore").rename(columns={"scheduled_date": "match_date", "scheduled_time": "match_time"})
            records = [_normalize_db_nulls(name, rec) for rec in _df_to_records(db_df[DB_MATCH_COLUMNS])]
            if not records:
                return
            on_conflict = ",".join(PRIMARY_KEYS[name])
            for batch in self._chunked(records):
                self.client.table(name).upsert(batch, on_conflict=on_conflict).execute()
            return

        records = [_normalize_db_nulls(name, rec) for rec in _df_to_records(normalized)]
        if not records:
            return
        on_conflict = ",".join(PRIMARY_KEYS[name])
        for batch in self._chunked(records):
            self.client.table(name).upsert(batch, on_conflict=on_conflict).execute()

    def append_row(self, name: str, row: dict) -> None:
        payload = {col: _pythonize_scalar(row.get(col)) for col in DATA_FILES[name]}
        if name == "match_participants":
            self.client.table(name).insert(_normalize_db_nulls(name, payload)).execute()
            return
        if name == "matches":
            payload = {
                "match_id": payload.get("match_id"),
                "created_at": payload.get("created_at"),
                "completed_at": payload.get("completed_at"),
                "match_type": payload.get("match_type"),
                "points_to_win": payload.get("points_to_win"),
                "team_a_score": payload.get("team_a_score"),
                "team_b_score": payload.get("team_b_score"),
                "winner": payload.get("winner"),
                "status": payload.get("status"),
                "video_url": payload.get("video_url"),
                "match_date": payload.get("scheduled_date"),
                "match_time": payload.get("scheduled_time"),
                "notes": payload.get("notes"),
            }
        payload = _normalize_db_nulls(name, payload)
        on_conflict = ",".join(PRIMARY_KEYS[name])
        self.client.table(name).upsert(payload, on_conflict=on_conflict).execute()

    def replace_all(self, payload: dict[str, pd.DataFrame]) -> None:
        for name, df in payload.items():
            self.save(name, df)
