# Badminton Tracker

A simple Streamlit app to record badminton games between friends.

## What it does

- Supports singles and doubles
- Records points per player and per game
- Records good shots, bad shots and service faults per player
- Tracks live event log with timestamps
- Supports undo for the last event
- Calculates Elo after completed matches
- Shows player stats and leaderboard
- Exports and imports app data as zip
- Stores data in CSV files so it is easy to inspect and later swap out for a database

## Project structure

```text
badminton_app/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── players.csv
│   ├── matches.csv
│   ├── events.csv
│   └── elo_history.csv
└── src/
    ├── elo.py
    ├── models.py
    ├── stats.py
    └── storage.py
```

## Run it

### 1. Create a virtual environment

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install packages

```bash
pip install -r requirements.txt
```

### 3. Start the app

```bash
streamlit run app.py
```

## How to use it

1. Add players in the left sidebar.
2. Create a singles or doubles match in the Live Match tab.
3. During the game:
   - add points to Team A or Team B and choose who scored
   - log good shots, bad shots or service faults for any player
   - undo the last event if needed
4. Complete the match once finished.
5. Elo and stats update automatically.
6. Export your data zip from the sidebar whenever you want a backup.
7. Import a zip later to restore or sync on another machine.

## Notes on Elo

- Every player starts at 1000
- Singles and doubles are both supported
- For doubles, the team average Elo is used and the rating change is split across both teammates

## Future upgrades

- MongoDB or BigQuery repository layer
- Authentication
- Cloud sync
- Rally-level tagging
- Shot heatmaps and richer analytics
- Mobile-first UI


## Authentication and roles

This version supports Streamlit OIDC login plus two roles:
- `editor`: can add players, create matches, record events, import data and complete matches
- `viewer`: read-only access to stats, history and exports

### Local setup
1. Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml`
2. Fill in your OIDC values and role email lists
3. Keep `.streamlit/secrets.toml` out of Git

### Streamlit Community Cloud
Add the same values in your app Secrets settings. Keep the repo public if you want. The allowed emails stay private in Streamlit secrets.

## Password roles

This app now uses two shared passwords stored in Streamlit secrets:

- `admin`: full access
- `viewer`: read-only access

Create `.streamlit/secrets.toml` locally with:

```toml
[passwords]
admin = "your-admin-password"
viewer = "your-viewer-password"
```

On Streamlit Community Cloud, put the same block into the app Secrets settings.
