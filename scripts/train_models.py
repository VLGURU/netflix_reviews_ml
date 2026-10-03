"""Обучение 5 моделей + GridSearchCV, сохранение метрик и ROC."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import pandas as pd
from config import MODELS_DIR
from src.data_loader import seed_db_if_empty, load_reviews_from_db
from src.sentiment import train_all_models
from src.logger import logger


def main():
    seed_db_if_empty()
    df = load_reviews_from_db()
    logger.info(f"Обучаю модели на {len(df)} отзывах...")

    results, roc_data, best_name = train_all_models(df, use_sentence_transformer=False)

    results.to_csv(MODELS_DIR / "metrics.csv", index=False)
    with open(MODELS_DIR / "roc.json", "w", encoding="utf-8") as f:
        json.dump(roc_data, f, ensure_ascii=False)

    print("\n=== Результаты ===")
    print(results.to_string(index=False))
    print(f"\nЛучшая модель: {best_name}")


if __name__ == "__main__":
    main()
