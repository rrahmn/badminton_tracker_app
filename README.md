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
