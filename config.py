import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
STATIC_PLOTS = BASE_DIR / "static" / "plots"

for d in (DATA_DIR, MODELS_DIR, STATIC_PLOTS):
    d.mkdir(parents=True, exist_ok=True)

NETFLIX_CSV = DATA_DIR / "netflix_titles.csv"
REVIEWS_DB = DATA_DIR / "reviews.db"
LOG_FILE = BASE_DIR / "app.log"

SECRET_KEY = "change-me-please-12345"
ADMIN_TOKEN = "admin123"  # для страницы логов

RANDOM_STATE = 42