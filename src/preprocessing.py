"""Classes de pre-processamento e pipelines (sem data leakage)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_loader import CATEGORICAL_FEATURES, NUMERIC_FEATURES


class TelcoCleaner(BaseEstimator, TransformerMixin):
    """Correcao de tipos *sem estado* (nao aprende nada dos dados => sem leakage).

    * ``TotalCharges``: texto -> numerico; brancos de clientes com tenure == 0 (ainda nao
      faturados) viram 0.0 por regra de negocio; demais invalidos viram NaN (tratados
      depois pelo imputer do pipeline).
    * ``SeniorCitizen``: inteiro 0/1 -> categoria "No"/"Yes".
    * Operacao idempotente: pode ser reaplicada a dados ja limpos (usado no app).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if "TotalCharges" in X.columns:
            tc = pd.to_numeric(X["TotalCharges"].astype(str).str.strip(), errors="coerce")
            if "tenure" in X.columns:
                zero_tenure = pd.to_numeric(X["tenure"], errors="coerce") == 0
                tc = tc.mask(tc.isna() & zero_tenure, 0.0)
            X["TotalCharges"] = tc.astype(float)
        if "SeniorCitizen" in X.columns:
            X["SeniorCitizen"] = (
                X["SeniorCitizen"].astype(str).map({"0": "No", "1": "Yes", "No": "No", "Yes": "Yes"})
            )
        return X


def build_preprocessor(numeric=NUMERIC_FEATURES, categorical=CATEGORICAL_FEATURES) -> ColumnTransformer:
    """ColumnTransformer: imputer+scaler (numericas) e imputer+OneHot (categoricas)."""
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", num_pipe, list(numeric)), ("cat", cat_pipe, list(categorical))],
        remainder="drop",
    )


def build_pipeline(model) -> Pipeline:
    """Pipeline completo: limpeza -> pre-processamento -> modelo."""
    return Pipeline([
        ("cleaner", TelcoCleaner()),
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])
