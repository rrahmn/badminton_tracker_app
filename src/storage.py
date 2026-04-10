from __future__ import annotations

from pathlib import Path
from typing import Dict
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
