import os
import pytest
import pandas as pd
from src.preprocessing import mask_phone, mask_name, mask_email, mask_pii, clean_text, preprocess
from src.data_loader import generate_reviews, load_reviews_from_db, seed_db_if_empty, init_db
from src.recommend import recommend_for_user, build_user_item_matrix
from src.sentiment import make_label, prepare_data
from src.stats import top_negative_words, top_positive_products


# 1. Маскировка телефона
def test_mask_phone():
    text = "Мой телефон +7-912-345-67-89, звоните!"
    assert "[ТЕЛЕФОН]" in mask_phone(text)
    assert "+7" not in mask_phone(text)


# 2. Маскировка имени и email
def test_mask_name_and_email():
    assert mask_name("Пишет Иван Петров") == "Пишет [ИМЯ] [ИМЯ]"
    assert "[EMAIL]" in mask_email("почта user@example.com")
    full = mask_pii("Иван, +7-912-345-67-89, test@mail.ru")
    assert "[ИМЯ]" in full and "[ТЕЛЕФОН]" in full and "[EMAIL]" in full


# 3. Загрузка/генерация данных
def test_generate_reviews_structure():
    df = generate_reviews(n_reviews=50)
    assert len(df) == 50
    for col in ["text", "rating", "product", "date", "user_id"]:
        assert col in df.columns
    assert df["rating"].between(1, 5).all()


# 4. Метрики / метки
def test_make_label():
    assert make_label(5) == 1
    assert make_label(4) == 1
    assert make_label(3) == 0
    assert make_label(1) == 0


def test_prepare_data():
    df = pd.DataFrame({
        "text": ["Отличный фильм!", "Скучно и ужасно"],
        "rating": [5, 1],
        "product": ["A", "B"],
        "date": ["2024-01-01", "2024-01-02"],
        "user_id": ["u1", "u2"],
    })
    out = prepare_data(df)
    assert "clean" in out.columns
    assert "label" in out.columns
    assert list(out["label"]) == [1, 0]


# 5. Рекомендации
def test_recommend_new_user():
    df = pd.DataFrame({
        "user_id": ["u1"] * 3 + ["u2"] * 3,
        "product": ["A", "B", "C"] * 2,
        "rating": [5, 4, 3, 4, 5, 2],
    })
    recs = recommend_for_user("unknown_user", df, top_n=2)
    assert len(recs) == 2
    assert all(isinstance(r[0], str) for r in recs)


def test_build_user_item_matrix():
    df = pd.DataFrame({
        "user_id": ["u1", "u1", "u2"],
        "product": ["A", "B", "A"],
        "rating": [5, 3, 4],
    })
    m = build_user_item_matrix(df)
    assert m.shape[0] == 2
    assert "A" in m.columns and "B" in m.columns


# 6. Статистика
def test_top_negative_words():
    df = pd.DataFrame({
        "text": ["скучно ужасно плохо", "ужасно скучно", "отлично супер"],
        "rating": [1, 2, 5],
    })
    words = top_negative_words(df, top_n=3)
    assert any(w == "ужасно" for w, _ in words)


def test_top_positive_products():
    df = pd.DataFrame({
        "product": ["A", "A", "A", "B", "B", "B"],
        "rating": [5, 5, 4, 1, 1, 2],
    })
    top = top_positive_products(df, min_reviews=3, top_n=5)
    assert "A" in top.index
    assert top.loc["A", "pos_share"] > top.loc["B", "pos_share"]