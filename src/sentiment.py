import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sentence_transformers import SentenceTransformer
from config import MODELS_DIR, RANDOM_STATE
from src.preprocessing import preprocess
from src.logger import logger

SENT_MODEL_PATH = MODELS_DIR / "sentiment_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf.joblib"
ST_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def make_label(rating):
    return 1 if rating >= 4 else 0  # 1 — позитив, 0 — негатив/нейтрал


def prepare_data(df):
    df = df.copy()
    df["clean"] = df["text"].astype(str).map(preprocess)
    df["label"] = df["rating"].map(make_label)
    return df


def train_all_models(df, use_sentence_transformer=False):
    df = prepare_data(df)
    X_text = df["clean"].values
    y = df["label"].values

    X_train_t, X_test_t, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    if use_sentence_transformer:
        st = SentenceTransformer(ST_MODEL_NAME)
        X_train = st.encode(X_train_t.tolist(), show_progress_bar=False)
        X_test = st.encode(X_test_t.tolist(), show_progress_bar=False)
        vectorizer = None
    else:
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train = vectorizer.fit_transform(X_train_t)
        X_test = vectorizer.transform(X_test_t)

    models = {
        "LogisticRegression": (
            LogisticRegression(max_iter=1000),
            {"C": [0.1, 1, 10], "solver": ["liblinear", "lbfgs"], "penalty": ["l2"]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {"n_estimators": [100, 200], "max_depth": [None, 10, 20], "min_samples_split": [2, 5]},
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, use_label_encoder=False),
            {"n_estimators": [100, 200], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]},
        ),
        "MLPClassifier": (
            MLPClassifier(max_iter=300, random_state=RANDOM_STATE),
            {"hidden_layer_sizes": [(64,), (128,), (64, 32)], "alpha": [0.0001, 0.001], "learning_rate_init": [0.001, 0.01]},
        ),
        "CatBoost": (
            CatBoostClassifier(verbose=0, random_state=RANDOM_STATE),
            {"iterations": [200, 400], "depth": [4, 6], "learning_rate": [0.05, 0.1]},
        ),
    }

    results = []
    best_models = {}
    roc_data = {}

    for name, (estimator, params) in models.items():
        logger.info(f"Обучение {name} (GridSearchCV)...")
        try:
            gs = GridSearchCV(estimator, params, cv=3, scoring="f1", n_jobs=-1)
            gs.fit(X_train, y_train)
            best = gs.best_estimator_
        except Exception as e:
            logger.warning(f"GridSearch не удался для {name}: {e}. Обучаю без подбора.")
            best = estimator.fit(X_train, y_train)
            gs = None

        y_pred = best.predict(X_test)
        if hasattr(best, "predict_proba"):
            y_proba = best.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred

        # метрики до подбора (для отчёта можно сравнить)
        base = estimator
        base.fit(X_train, y_train)
        y_pred_base = base.predict(X_test)
        base_metrics = {
            "acc": accuracy_score(y_test, y_pred_base),
            "prec": precision_score(y_test, y_pred_base, zero_division=0),
            "rec": recall_score(y_test, y_pred_base, zero_division=0),
            "f1": f1_score(y_test, y_pred_base, zero_division=0),
        }

        results.append({
            "model": name,
            "acc_base": base_metrics["acc"],
            "f1_base": base_metrics["f1"],
            "acc": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "best_params": str(gs.best_params_) if gs else "—",
        })

        best_models[name] = best
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr.tolist(), tpr.tolist(), auc(fpr, tpr))

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    best_name = results_df.iloc[0]["model"]
    joblib.dump(best_models[best_name], SENT_MODEL_PATH)
    if vectorizer is not None:
        joblib.dump(vectorizer, VECTORIZER_PATH)
    else:
        joblib.dump({"type": "sentence_transformer", "name": ST_MODEL_NAME}, VECTORIZER_PATH)

    return results_df, roc_data, best_name


def load_sentiment_model():
    model = joblib.load(SENT_MODEL_PATH)
    vec = joblib.load(VECTORIZER_PATH)
    return model, vec


def predict_sentiment(text: str):
    """Возвращает (label, confidence)."""
    model, vec = load_sentiment_model()
    cleaned = preprocess(text)
    if isinstance(vec, dict) and vec.get("type") == "sentence_transformer":
        st = SentenceTransformer(vec["name"])
        X = st.encode([cleaned])
    else:
        X = vec.transform([cleaned])
    proba = model.predict_proba(X)[0]
    idx = int(np.argmax(proba))
    label = "позитивный" if idx == 1 else "негативный"
    return label, float(proba[idx])