"""Funcoes de avaliacao, estilo grafico e selecao do campeao."""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_validate

MAIN_METRIC = "roc_auc"
SCORING = {
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "pr_auc": "average_precision",
}


def set_plot_style() -> None:
    """Padrao grafico do enunciado: whitegrid, fonte serifada, sem bordas sup./dir., viridis."""
    sns.set_theme(style="whitegrid", palette="viridis")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Georgia"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.dpi": 110,
        "figure.figsize": (8, 4.5),
    })


def evaluate_cv(model_name: str, strategy: str, pipe, X, y, cv, search_time: float | None = None) -> dict:
    """Avalia um pipeline nos folds ``cv``: treino, CV (media/desvio) e tempo de treino."""
    res = cross_validate(clone(pipe), X, y, cv=cv, scoring=SCORING,
                         return_train_score=True, n_jobs=1)
    t0 = time.perf_counter()
    clone(pipe).fit(X, y)
    fit_time = time.perf_counter() - t0
    row = {
        "Modelo": model_name,
        "Estratégia": strategy,
        "Score Treino": res["train_roc_auc"].mean(),
        "Score Validação (CV)": res["test_roc_auc"].mean(),
        "Std CV": res["test_roc_auc"].std(),
        "Tempo treino (s)": fit_time,
        "Tempo busca (s)": search_time if search_time is not None else 0.0,
        "_folds": res["test_roc_auc"],  # scores por fold (usado nos graficos com barra de erro)
    }
    row["Gap (Δ)"] = row["Score Treino"] - row["Score Validação (CV)"]
    for m in ["accuracy", "precision", "recall", "f1", "pr_auc"]:
        row[f"CV {m}"] = res[f"test_{m}"].mean()
    return row


def classification_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Metricas principais e secundarias para um limiar de decisao."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def get_feature_names(pipe) -> list[str]:
    """Nomes das colunas apos o ColumnTransformer (para feature importance)."""
    names = pipe.named_steps["preprocessor"].get_feature_names_out()
    return [n.split("__", 1)[1] for n in names]


def select_champion(table: pd.DataFrame) -> pd.Series:
    """Regra quantitativa de promocao a campeao.

    1. Mantem as configuracoes com CV >= (melhor CV - desvio-padrao do melhor), isto e,
       estatisticamente indistinguiveis do topo (regra de 1 desvio-padrao).
    2. Entre elas escolhe o menor gap treino-CV (melhor generalizacao); desempate por
       menor Std CV e, por fim, menor tempo de treino.
    """
    best = table.loc[table["Score Validação (CV)"].idxmax()]
    thr = best["Score Validação (CV)"] - best["Std CV"]
    cands = table[table["Score Validação (CV)"] >= thr]
    return cands.sort_values(["Gap (Δ)", "Std CV", "Tempo treino (s)"]).iloc[0]
