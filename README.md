# Netflix Reviews ML — Практическая работа №10

ML-продукт для анализа отзывов клиентов интернет-магазина на основе каталога Netflix.

**Автор:** Плохотный В.А., группа Т-11-24
**Дисциплина:** Технология разработки и защиты баз данных
**Преподаватель:** Д. В. Чекан

---

## Что умеет

- 🎯 **Классификация тональности** — 5 моделей (LogisticRegression, RandomForest, XGBoost, MLPClassifier, CatBoost) с GridSearchCV
- 🔒 **Защита ПД** — маскировка телефонов, имён, email
- 🧩 **Кластеризация** — KMeans + DBSCAN + t-SNE (SentenceTransformer)
- 👥 **Рекомендации** — user-based collaborative filtering
- 🌐 **Веб-интерфейс** — Flask, 5 страниц
- 📊 **Визуализация** — WordCloud, Plotly, matplotlib
- 📝 **Логирование** — `app.log`
- ✅ **9 unit-тестов** — pytest

---

## Датасет

- Источник: [Netflix Titles (TidyTuesday)](https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-04-20/netflix_titles.csv)
- На основе каталога сгенерировано **2000 отзывов** с оценками 1–5, датами и user_id
- Хранение: **SQLite** (`data/reviews.db`)

---

## Установка

```bash
git clone https://github.com/VLGURU/netflix_reviews_ml.git
cd netflix_reviews_ml

python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt

# macOS: libomp для XGBoost
brew install libomp

python app.py
```

Открыть: **http://127.0.0.1:5000**

---

## Обучение моделей

```bash
python -m scripts.train_models     # 5-15 минут
python -m scripts.build_clusters   # 5-10 минут
```

---

## Тесты

```bash
pytest tests/ -v
```

Ожидается: **9 passed**.

---

## Результаты моделей

| Модель | Accuracy | F1 |
|---|---|---|
| **LogisticRegression** ⭐ | **0.9325** | **0.9436** |
| RandomForest | 0.9325 | 0.9436 |
| CatBoost | 0.9275 | 0.9397 |
| MLPClassifier | 0.9175 | 0.9317 |
| XGBoost | 0.9150 | 0.9303 |

---

## Кластеризация (KMeans, 5 кластеров)

| Кластер | Размер | Позитив | Топ-слова |
|---|---|---|---|
| 0 | 193 | 84.5% | рекомендую, отличный, понравилось |
| 1 | 92 | 33.7% | ужасный, сюжет, игра |
| 2 | 45 | 64.4% | телефон, мой |
| 3 | 130 | 23.1% | скучно, слабый, разочарование |
| 4 | 40 | 90.0% | интересный, сюжет, музыка |

---

## Страницы веб-приложения

| URL | Что показывает |
|---|---|
| `/` | Главная: статистика, формы, рекомендации |
| `/product_stats` | Средний рейтинг, облака слов |
| `/models_compare` | Таблица метрик + ROC |
| `/clusters` | Plotly-визуализация кластеров |
| `/logs?token=admin123` | Логи (только админ) |

---

## Структура

```
netflix_reviews_ml/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── netflix_titles.csv
│   └── reviews.db
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── sentiment.py
│   ├── clustering.py
│   ├── recommend.py
│   ├── stats.py
│   └── logger.py
├── scripts/
│   ├── train_models.py
│   └── build_clusters.py
├── templates/
├── static/
├── tests/
└── models/
```