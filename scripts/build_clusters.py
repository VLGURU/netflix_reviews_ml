"""Кластеризация отзывов: KMeans + DBSCAN + PCA/t-SNE."""
import json
from config import MODELS_DIR
from src.data_loader import seed_db_if_empty, load_reviews_from_db
from src.clustering import build_clusters
from src.logger import logger


def main():
    seed_db_if_empty()
    df = load_reviews_from_db()
    logger.info(f"Кластеризую {len(df)} отзывов...")

    df_clustered, report = build_clusters(df, sample_size=500)

    with open(MODELS_DIR / "cluster_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== Отчёт по кластерам ===")
    for cl, info in report.items():
        print(f"Кластер {cl}: {info['size']} отзывов, позитив {info['pos_share']*100:.1f}%")
        print("  Топ-слова:", info["top_words"][:5])


if __name__ == "__main__":
    main()