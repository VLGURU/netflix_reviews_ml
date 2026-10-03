import os
import uuid
from functools import wraps
from datetime import datetime

from flask import (Flask, render_template, request, redirect, url_for,
                   session, jsonify, make_response, abort)
import pandas as pd
import numpy as np

from config import (SECRET_KEY, ADMIN_TOKEN, STATIC_PLOTS, MODELS_DIR, REVIEWS_DB)
from src.data_loader import seed_db_if_empty, load_reviews_from_db, init_db
from src.sentiment import predict_sentiment, load_sentiment_model, SENT_MODEL_PATH
from src.recommend import recommend_for_user
from src.stats import full_report, top_positive_products, top_negative_words
from src.clustering import build_clusters
from src.logger import log_action, logger

app = Flask(__name__)
app.secret_key = SECRET_KEY


# ---------------------- утилиты ----------------------
def get_user_id():
    """Возвращает user_id из сессии, создаёт при первом входе."""
    if "user_id" not in session:
        session["user_id"] = f"web_{uuid.uuid4().hex[:8]}"
        log_action(session["user_id"], "new_session", "created")
    return session["user_id"]


def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = request.args.get("token") or request.cookies.get("admin_token")
        if token != ADMIN_TOKEN:
            abort(403, description="Доступ только для администратора. Добавьте ?token=admin123")
        return f(*args, **kwargs)
    return wrapper


# ---------------------- инициализация ----------------------
@app.before_request
def _ensure_db():
    if not os.path.exists(REVIEWS_DB):
        seed_db_if_empty()


# ---------------------- маршруты ----------------------
@app.route("/")
def index():
    user_id = get_user_id()
    df = load_reviews_from_db()
    log_action(user_id, "view_index", f"{len(df)} reviews")

    # рекомендации для текущего пользователя
    try:
        recs = recommend_for_user(user_id, df, top_n=5)
    except Exception as e:
        logger.warning(f"Ошибка рекомендаций: {e}")
        recs = []

    # метрики по товарам
    stats = df.groupby("product").agg(
        avg_rating=("rating", "mean"),
        n=("rating", "size"),
        pos_share=("rating", lambda s: (s >= 4).mean()),
    ).reset_index().sort_values("n", ascending=False).head(10)

    # графики (генерируем при первом запросе)
    report_path = STATIC_PLOTS / "student_report.png"
    if not report_path.exists():
        full_report(df)

    return render_template(
        "index.html",
        user_id=user_id,
        recs=recs,
        stats=stats.to_dict(orient="records"),
        n_reviews=len(df),
        avg_rating=float(df["rating"].mean()),
        pos_share=float((df["rating"] >= 4).mean()),
    )


@app.route("/predict", methods=["POST"])
def predict():
    user_id = get_user_id()
    text = request.form.get("text", "").strip()
    if not text:
        return jsonify({"error": "Пустой текст"}), 400
    try:
        label, conf = predict_sentiment(text)
    except Exception as e:
        logger.exception("predict_sentiment failed")
        return jsonify({"error": str(e)}), 500

    log_action(user_id, "predict", f"{label} ({conf:.2f})")
    return jsonify({"label": label, "confidence": round(conf * 100, 2)})


@app.route("/add_review", methods=["POST"])
def add_review():
    user_id = get_user_id()
    text = request.form.get("text", "").strip()
    rating = request.form.get("rating", type=int)
    product = request.form.get("product", "").strip() or "Неизвестный товар"

    if not text or not rating or not (1 <= rating <= 5):
        return jsonify({"error": "Некорректные данные"}), 400

    import sqlite3
    conn = sqlite3.connect(REVIEWS_DB)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO reviews (text, rating, product, date, user_id) VALUES (?, ?, ?, ?, ?)",
        (text, rating, product, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_id),
    )
    conn.commit()
    conn.close()

    log_action(user_id, "add_review", f"{product} | {rating}")
    return redirect(url_for("index"))


@app.route("/product_stats")
def product_stats():
    user_id = get_user_id()
    df = load_reviews_from_db()
    log_action(user_id, "view_product_stats", "")

    stats = df.groupby("product").agg(
        avg_rating=("rating", "mean"),
        n=("rating", "size"),
        pos_share=("rating", lambda s: (s >= 4).mean()),
    ).reset_index().sort_values("avg_rating", ascending=False)

    top_pos = top_positive_products(df, min_reviews=3, top_n=10).reset_index().to_dict(orient="records")
    top_neg_words = top_negative_words(df, top_n=10)

    return render_template(
        "product_stats.html",
        stats=stats.head(50).to_dict(orient="records"),
        top_pos=top_pos,
        top_neg_words=top_neg_words,
    )


@app.route("/models_compare")
def models_compare():
    user_id = get_user_id()
    log_action(user_id, "view_models_compare", "")

    metrics_path = MODELS_DIR / "metrics.csv"
    roc_path = MODELS_DIR / "roc.json"
    if not metrics_path.exists():
        return render_template("models_compare.html", metrics=[], roc=None,
                               message="Метрики ещё не посчитаны. Запустите scripts/train_models.py")

    metrics = pd.read_csv(metrics_path).to_dict(orient="records")

    import json
    roc = None
    if roc_path.exists():
        with open(roc_path, encoding="utf-8") as f:
            roc = json.load(f)

    return render_template("models_compare.html", metrics=metrics, roc=roc, message=None)


@app.route("/clusters")
def clusters():
    user_id = get_user_id()
    log_action(user_id, "view_clusters", "")

    clusters_csv = "data/clusters.csv"
    if not os.path.exists(clusters_csv):
        return render_template("clusters.html", plot=None,
                               message="Кластеры ещё не посчитаны. Запустите scripts/build_clusters.py")

    import plotly.express as px
    import json
    df = pd.read_csv(clusters_csv)

    fig = px.scatter(df, x="tsne_x", y="tsne_y", color=df["kmeans"].astype(str),
                     hover_data=["text", "rating", "product"],
                     title="Кластеры отзывов (t-SNE + KMeans)")
    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    report_path = MODELS_DIR / "cluster_report.json"
    report = {}
    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)

    return render_template("clusters.html", plot=plot_html, report=report, message=None)


@app.route("/logs")
@admin_required
def logs():
    log_action("admin", "view_logs", "")
    from config import LOG_FILE
    lines = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, encoding="utf-8") as f:
            lines = f.readlines()[-50:]
    return render_template("logs.html", lines=[l.rstrip() for l in lines])


@app.errorhandler(403)
def forbidden(e):
    return render_template("base.html", error=str(e)), 403


if __name__ == "__main__":
    seed_db_if_empty()
    app.run(debug=True, host="127.0.0.1", port=5000)