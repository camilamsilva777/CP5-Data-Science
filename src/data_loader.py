"""Carga, download e particionamento do dataset Telco Customer Churn."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

RANDOM_STATE = 42
TARGET = "Churn"
ID_COL = "customerID"

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw_dataset.csv"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def download_dataset(path: str | Path = DEFAULT_PATH, url: str = DATA_URL) -> Path:
    """Baixa o CSV bruto caso ainda nao exista em ``data/``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Baixando dataset de {url} ...")
        urllib.request.urlretrieve(url, path)
    return path


def load_raw(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """Le o CSV bruto (baixando-o automaticamente se necessario)."""
    path = download_dataset(path)
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X (apenas as features modeladas) e y binario (Churn: Yes=1)."""
    X = df[FEATURES].copy()
    y = df[TARGET]
    if y.dtype == object or str(y.dtype) == "str":
        y = y.map({"Yes": 1, "No": 0})
    return X, y.astype(int)


def make_train_test(X, y, test_size: float = 0.2):
    """Hold-out estratificado 80/20 com random_state=42 (D_treino / D_teste)."""
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)


def make_cv(n_splits: int = 5) -> StratifiedKFold:
    """Objeto de CV unico, reutilizado por TODOS os modelos e estrategias."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
