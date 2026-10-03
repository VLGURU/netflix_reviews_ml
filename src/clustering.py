import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from collections import Counter
from config import STATIC_PLOTS, RANDOM_STATE
from src.preprocessing import preprocess
from src.sentiment import ST_MODEL_NAME
from src.logger import logger


def build_clusters(df, sample_size=500):
    df = df.copy()
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=RANDOM_STATE).reset_index(drop=True)
    df["clean"] = df["text"].astype(str).map(preprocess)

    logger.info("Кодирую отзывы SentenceTransformer...")
    st = SentenceTransformer(ST_MODEL_NAME)
    emb = st.encode(df["clean"].tolist(), show_progress_bar=False)

    logger.info("KMeans...")
    km = KMeans(n_clusters=5, random_state=RANDOM_STATE, n_init=10)
    df["kmeans"] = km.fit_predict(emb)

    logger.info("DBSCAN...")
    db = DBSCAN(eps=0.8, min_samples=5, metric="cosine")
    df["dbscan"] = db.fit_predict(emb)

    logger.info("PCA / t-SNE...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    emb_pca = pca.fit_transform(emb)
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, init="pca")
    emb_tsne = tsne.fit_transform(emb)

    df["pca_x"], df["pca_y"] = emb_pca[:, 0], emb_pca[:, 1]
    df["tsne_x"], df["tsne_y"] = emb_tsne[:, 0], emb_tsne[:, 1]

    # графики
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(df["pca_x"], df["pca_y"], c=df["kmeans"], cmap="tab10", alpha=0.6)
    axes[0].set_title("PCA + KMeans")
    axes[1].scatter(df["tsne_x"], df["tsne_y"], c=df["kmeans"], cmap="tab10", alpha=0.6)
    axes[1].set_title("t-SNE + KMeans")
    plt.tight_layout()
    plt.savefig(STATIC_PLOTS / "clusters.png", dpi=120)
    plt.close()

    # топ-слова и тональность по кластерам
    report = {}
    for cl in sorted(df["kmeans"].unique()):
        sub = df[df["kmeans"] == cl]
        words = " ".join(sub["clean"]).split()
        stop = {"и", "в", "на", "с", "не", "а", "но", "что", "это", "я"}
        words = [w for w in words if w not in stop and len(w) > 2]
        top = Counter(words).most_common(10)
        pos_share = (sub["rating"] >= 4).mean()
        report[int(cl)] = {
            "size": len(sub),
            "top_words": top,
            "pos_share": float(pos_share),
            "examples": sub["text"].head(5).tolist(),
        }

    df.to_csv("data/clusters.csv", index=False)
    return df, report