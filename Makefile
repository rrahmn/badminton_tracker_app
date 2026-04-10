PYTHON ?= python
APP_FILE := app.py
VENV_DIR := .venv
SECRETS_EXAMPLE := .streamlit/secrets.example.toml
SECRETS_FILE := .streamlit/secrets.toml

ifeq ($(OS),Windows_NT)
VENV_PY := $(VENV_DIR)/Scripts/python.exe
else
VENV_PY := $(VENV_DIR)/bin/python
endif

.PHONY: help setup venv install secrets run clean

help:
	@echo Available targets:
	@echo   make setup   - Create virtual environment, install requirements, and create .streamlit/secrets.toml if missing
	@echo   make venv    - Create the virtual environment in $(VENV_DIR)
	@echo   make install - Install requirements into the virtual environment
	@echo   make secrets - Create $(SECRETS_FILE) from $(SECRETS_EXAMPLE) if it does not already exist
	@echo   make run     - Run the Streamlit app
	@echo   make clean   - Remove the virtual environment

setup: venv install secrets

venv:
	$(PYTHON) -m venv $(VENV_DIR)

install: venv
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.txt

secrets:
	$(PYTHON) -c "from pathlib import Path; src=Path(r'$(SECRETS_EXAMPLE)'); dst=Path(r'$(SECRETS_FILE)'); dst.parent.mkdir(parents=True, exist_ok=True); dst.write_text(src.read_text(encoding='utf-8'), encoding='utf-8') if src.exists() and not dst.exists() else None; print(f'Created {dst}' if src.exists() and dst.exists() and dst.read_text(encoding=\"utf-8\") == src.read_text(encoding=\"utf-8\") else (f'{dst} already exists' if dst.exists() else f'Missing template: {src}'))"

run: install secrets
	$(VENV_PY) -m streamlit run $(APP_FILE)

clean:
	$(PYTHON) -c "import shutil; from pathlib import Path; shutil.rmtree(Path(r'$(VENV_DIR)'), ignore_errors=True); print('Removed $(VENV_DIR)')"
