import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from config import STATIC_PLOTS
from src.preprocessing import clean_text

sns.set_style("whitegrid")


def top_positive_products(df, min_reviews=5, top_n=10):
    g = df.groupby("product").agg(
        n=("rating", "size"),
        pos_share=("rating", lambda s: (s >= 4).mean()),
        avg_rating=("rating", "mean"),
    )
    g = g[g["n"] >= min_reviews].sort_values("pos_share", ascending=False)
    return g.head(top_n)


def top_negative_words(df, top_n=5):
    neg = df[df["rating"] <= 2]["text"].astype(str).map(clean_text)
    words = " ".join(neg).split()
    stop = {"и", "в", "на", "с", "не", "а", "но", "что", "это", "я", "за", "по", "из", "так", "же"}
    words = [w for w in words if w not in stop and len(w) > 2]
    return Counter(words).most_common(top_n)


def make_wordcloud(df, path, positive=True):
    subset = df[df["rating"] >= 4] if positive else df[df["rating"] <= 2]
    text = " ".join(subset["text"].astype(str).map(clean_text))
    if not text.strip():
        text = "нет данных"
    wc = WordCloud(width=800, height=400, background_color="white",
                   colormap="Greens" if positive else "Reds").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_rating_distribution(df, path):
    plt.figure(figsize=(8, 4))
    sns.countplot(x="rating", data=df, palette="viridis")
    plt.title("Распределение оценок")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_reviews_over_time(df, path):
    d = df.copy()
    d["date"] = d["date"].dt.date
    daily = d.groupby("date").size()
    plt.figure(figsize=(10, 4))
    plt.plot(daily.index, daily.values, marker="o", color="#1f77b4")
    plt.title("Динамика отзывов по дням")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def full_report(df):
    """Сохраняет графики в static/plots."""
    plot_rating_distribution(df, STATIC_PLOTS / "rating_dist.png")
    plot_reviews_over_time(df, STATIC_PLOTS / "reviews_time.png")
    make_wordcloud(df, STATIC_PLOTS / "wordcloud_pos.png", positive=True)
    make_wordcloud(df, STATIC_PLOTS / "wordcloud_neg.png", positive=False)
    # student_report.png — сводный отчёт
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.countplot(x="rating", data=df, ax=axes[0, 0], palette="viridis")
    axes[0, 0].set_title("Распределение оценок")
    d = df.copy(); d["date"] = d["date"].dt.date
    daily = d.groupby("date").size()
    axes[0, 1].plot(daily.index, daily.values, color="#1f77b4")
    axes[0, 1].set_title("Динамика отзывов")
    axes[0, 1].tick_params(axis="x", rotation=45)
    pos_wc = WordCloud(width=400, height=200, background_color="white", colormap="Greens").generate(
        " ".join(df[df["rating"] >= 4]["text"].astype(str).map(clean_text)) or "нет")
    axes[1, 0].imshow(pos_wc); axes[1, 0].axis("off"); axes[1, 0].set_title("Облако слов (позитив)")
    neg_wc = WordCloud(width=400, height=200, background_color="white", colormap="Reds").generate(
        " ".join(df[df["rating"] <= 2]["text"].astype(str).map(clean_text)) or "нет")
    axes[1, 1].imshow(neg_wc); axes[1, 1].axis("off"); axes[1, 1].set_title("Облако слов (негатив)")
    plt.tight_layout()
    plt.savefig(STATIC_PLOTS / "student_report.png", dpi=120)
    plt.close()
    return STATIC_PLOTS / "student_report.png"