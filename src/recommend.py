import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.logger import logger


def build_user_item_matrix(df):
    """Строит матрицу user x product со средними оценками."""
    return df.pivot_table(
        index="user_id", columns="product", values="rating", aggfunc="mean"
    ).fillna(0)


def _top_products_by_avg(df, top_n=5):
    top = df.groupby("product")["rating"].mean().sort_values(ascending=False).head(top_n)
    return [(p, float(r)) for p, r in top.items()]


def recommend_for_user(user_id, df, top_n=5):
    """
    User-based collaborative filtering.
    Возвращает список (product, score) — топ-N рекомендаций для пользователя.
    """
    matrix = build_user_item_matrix(df)

    if user_id not in matrix.index:
        logger.info(f"recommend_for_user | новый пользователь {user_id} — топ по среднему")
        return _top_products_by_avg(df, top_n)

    sim = cosine_similarity(matrix)
    sim_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)
    similar_users = sim_df[user_id].drop(user_id).sort_values(ascending=False).head(10).index

    user_rated = matrix.loc[user_id]
    user_rated = user_rated[user_rated > 0].index.tolist()

    scores = {}
    for other in similar_users:
        sim_score = sim_df.loc[user_id, other]
        for product, rating in matrix.loc[other].items():
            if rating > 0 and product not in user_rated:
                scores.setdefault(product, []).append(rating * sim_score)

    ranked = sorted(
        [(p, float(np.mean(v))) for p, v in scores.items()],
        key=lambda x: x[1], reverse=True,
    )[:top_n]

    if not ranked:
        logger.info(f"recommend_for_user | нет коллаборативных данных — топ по среднему")
        return _top_products_by_avg(df, top_n)

    return ranked