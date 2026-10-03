import random
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import NETFLIX_CSV, REVIEWS_DB, RANDOM_STATE
from src.logger import logger

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

POS_TEMPLATES = [
    "Отличный фильм, смотрел(а) на одном дыхании! {title} — рекомендую.",
    "Прекрасная игра актёров, сюжет держит в напряжении. {title} супер!",
    "Очень понравилось, буду пересматривать. {title} — шедевр.",
    "Захватывающе и красиво снято. {title} заслуживает внимания.",
    "Великолепно! {title} превзошёл все ожидания.",
    "Интересный сюжет и отличная музыка. Спасибо за {title}.",
    "Отличный выбор на вечер. {title} — 10 из 10.",
    "Понравилось, особенно финал. {title} рекомендую друзьям.",
]

NEU_TEMPLATES = [
    "Нормально, но ничего особенного. {title} на один раз.",
    "Средний фильм, есть плюсы и минусы. {title} — так себе.",
    "Ожидал(а) большего. {title} на троечку.",
    "Смотрибельно, но забывается. {title} — средне.",
    "Не плохо, но и не хорошо. {title} на любителя.",
]

NEG_TEMPLATES = [
    "Скучно, еле досмотрел(а). {title} — разочарование.",
    "Ужасный сюжет и плохая игра. Не рекомендую {title}.",
    "Потерянное время. {title} — худшее, что я видел(а).",
    "Не понравилось совсем. {title} — мимо.",
    "Слабый сценарий и предсказуемый финал. {title} не стоит.",
    "Разочарован(а). {title} — пустая трата времени.",
]

POS_WORDS = ["отличный", "прекрасный", "великолепно", "супер", "рекомендую", "шедевр"]
NEG_WORDS = ["ужасный", "скучно", "разочарование", "плохой", "мимо", "слабый"]


def load_netflix_titles() -> pd.DataFrame:
    df = pd.read_csv(NETFLIX_CSV)
    df = df[["show_id", "type", "title", "description", "release_year", "date_added"]].dropna(subset=["title"])
    df["description"] = df["description"].fillna("")
    return df


def generate_reviews(n_reviews: int = 2000, n_users: int = 150) -> pd.DataFrame:
    """Генерирует синтетический датасет отзывов на основе каталога Netflix."""
    titles_df = load_netflix_titles()
    titles = titles_df["title"].tolist()

    rows = []
    start_date = datetime(2023, 1, 1)
    for i in range(n_reviews):
        title = random.choice(titles)
        rating = random.choices([1, 2, 3, 4, 5], weights=[0.10, 0.10, 0.20, 0.30, 0.30])[0]
        if rating >= 4:
            template = random.choice(POS_TEMPLATES)
        elif rating == 3:
            template = random.choice(NEU_TEMPLATES)
        else:
            template = random.choice(NEG_TEMPLATES)

        # немного шума: 15% — шаблон не соответствует оценке
        if random.random() < 0.15:
            template = random.choice(POS_TEMPLATES + NEU_TEMPLATES + NEG_TEMPLATES)

        text = template.format(title=title)
        # добавим шум слов
        extra = random.choice([[], POS_WORDS, NEG_WORDS])
        if extra:
            text += " " + " ".join(random.sample(extra, k=min(2, len(extra))))

        user_id = f"user_{random.randint(1, n_users):04d}"
        date = start_date + timedelta(days=random.randint(0, 700), minutes=random.randint(0, 1439))

        # изредка добавим ПД (телефон/имя) — чтобы работала маскировка
        if random.random() < 0.10:
            text += f" Мой телефон +7-9{random.randint(10,99)}-{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(10,99)}."
        if random.random() < 0.08:
            text += " Пишет Иван Петров."

        rows.append({
            "review_id": i + 1,
            "text": text,
            "rating": rating,
            "product": title,
            "date": date.strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": user_id,
        })

    return pd.DataFrame(rows)


def init_db():
    conn = sqlite3.connect(REVIEWS_DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            rating INTEGER NOT NULL,
            product TEXT NOT NULL,
            date TEXT NOT NULL,
            user_id TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def seed_db_if_empty():
    init_db()
    conn = sqlite3.connect(REVIEWS_DB)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM reviews")
    count = cur.fetchone()[0]
    conn.close()
    if count == 0:
        logger.info("База пуста — генерирую синтетические отзывы...")
        df = generate_reviews()
        conn = sqlite3.connect(REVIEWS_DB)
        df[["text", "rating", "product", "date", "user_id"]].to_sql(
            "reviews", conn, if_exists="append", index=False
        )
        conn.commit()
        conn.close()
        logger.info(f"Загружено {len(df)} отзывов в SQLite.")
    else:
        logger.info(f"В базе уже {count} отзывов.")


def load_reviews_from_db() -> pd.DataFrame:
    init_db()
    conn = sqlite3.connect(REVIEWS_DB)
    df = pd.read_sql_query("SELECT * FROM reviews", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


if __name__ == "__main__":
    seed_db_if_empty()
    df = load_reviews_from_db()
    print(df.head())
    print(df.shape)