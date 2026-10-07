"""App Streamlit — Previsão de Churn (campeão do Checkpoint 5) + teste de paridade com o notebook.

Execução:  streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.preprocessing import TelcoCleaner  # noqa: F401  (necessário para desserializar o pipeline)

ROOT = Path(__file__).resolve().parent
PIPELINE_PATH = ROOT / "artifacts" / "best_pipeline.joblib"
METADATA_PATH = ROOT / "artifacts" / "app_metadata.json"

st.set_page_config(page_title="Previsão de Churn", page_icon="📉", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    .result-card {border-radius: 16px; padding: 1.4rem 1.6rem; color: #fff; margin-bottom: .8rem;}
    .result-churn {background: linear-gradient(135deg, #b5179e, #7209b7);}
    .result-stay  {background: linear-gradient(135deg, #2a9d8f, #1b6f66);}
    .result-card h2 {margin: 0; font-size: 1.7rem;}
    .result-card p  {margin: .2rem 0 0 0; opacity: .92;}
    .parity-ok  {border-left: 6px solid #2a9d8f; background: rgba(42,157,143,.12); padding: .8rem 1rem; border-radius: 8px;}
    .parity-bad {border-left: 6px solid #d62828; background: rgba(214,40,40,.12); padding: .8rem 1rem; border-radius: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Carregando pipeline...")
def load_artifacts():
    pipeline = joblib.load(PIPELINE_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return pipeline, metadata


if not (PIPELINE_PATH.exists() and METADATA_PATH.exists()):
    st.error("Artefatos não encontrados em `artifacts/`. Execute o notebook "
             "`notebooks/Checkpoint05_RF_XGBoost_LightGBM.ipynb` (Run All) para gerá-los.")
    st.stop()

pipeline, meta = load_artifacts()
FEATURES = meta["feature_order"]
NUM, CAT = meta["numeric"], meta["categorical"]
SAMPLES = meta["consistency_samples"]
THRESHOLD = meta["threshold"]

LABELS = {
    "gender": "Gênero", "SeniorCitizen": "Idoso (≥ 65 anos)", "Partner": "Possui parceiro(a)",
    "Dependents": "Possui dependentes", "tenure": "Tempo de casa (meses)", "PhoneService": "Telefonia",
    "MultipleLines": "Múltiplas linhas", "InternetService": "Tipo de internet",
    "OnlineSecurity": "Segurança online", "OnlineBackup": "Backup online",
    "DeviceProtection": "Proteção de dispositivo", "TechSupport": "Suporte técnico",
    "StreamingTV": "Streaming de TV", "StreamingMovies": "Streaming de filmes", "Contract": "Contrato",
    "PaperlessBilling": "Fatura digital", "PaymentMethod": "Forma de pagamento",
    "MonthlyCharges": "Mensalidade (US$)", "TotalCharges": "Total cobrado (US$)",
}
GROUPS = {
    "👤 Perfil do cliente": ["gender", "SeniorCitizen", "Partner", "Dependents"],
    "📄 Contrato e cobrança": ["tenure", "Contract", "PaperlessBilling", "PaymentMethod",
                               "MonthlyCharges", "TotalCharges"],
    "📡 Serviços contratados": ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                                "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                                "StreamingMovies"],
}

# ---- estado inicial: mediana (numéricas) / moda (categóricas) de D_treino --------------------------
for col in FEATURES:
    key = f"in_{col}"
    if key not in st.session_state:
        st.session_state[key] = NUM[col]["default"] if col in NUM else CAT[col]["default"]


def widget_bounds(col: str):
    """Faixas do slider/number_input, ampliadas para sempre conter as amostras de consistência."""
    vals = [s["inputs"][col] for s in SAMPLES]
    lo, hi = min([NUM[col]["min"], *vals]), max([NUM[col]["max"], *vals])
    return lo, hi


def current_inputs() -> dict:
    row = {}
    for col in FEATURES:
        v = st.session_state[f"in_{col}"]
        row[col] = int(v) if col == "tenure" else float(v) if col in NUM else str(v)
    return row


def predict(row: dict) -> tuple[int, float]:
    df = pd.DataFrame([row], columns=FEATURES)
    proba = float(pipeline.predict_proba(df)[0, 1])
    return int(pipeline.predict(df)[0]), proba


def fill_sample(idx: int) -> None:
    """Callback: preenche os inputs com a amostra escolhida e marca a validação."""
    for col, value in SAMPLES[idx]["inputs"].items():
        st.session_state[f"in_{col}"] = value
    st.session_state["validated_idx"] = idx


# ---- sidebar: inputs dinâmicos -----------------------------------------------------------------------
st.sidebar.title("🎛️ Dados do cliente")
st.sidebar.caption("Valores iniciais = mediana/moda do conjunto de treino.")
for title, cols in GROUPS.items():
    with st.sidebar.expander(title, expanded=title.startswith(("👤", "📄"))):
        for col in cols:
            if col == "tenure":
                lo, hi = widget_bounds(col)
                st.slider(LABELS[col], int(lo), int(hi), key=f"in_{col}")
            elif col in NUM:
                lo, hi = widget_bounds(col)
                st.number_input(LABELS[col], min_value=0.0, max_value=float(round(hi * 1.5)), step=0.05,
                                format="%.2f", key=f"in_{col}")
            else:
                st.selectbox(LABELS[col], CAT[col]["options"], key=f"in_{col}")

# ---- cabeçalho -----------------------------------------------------------------------------------------
st.title("📉 Previsão de Churn de Clientes")
st.caption(f"Modelo em produção: **{meta['model']}** · CV ROC-AUC {meta['cv_score']:.4f} ± {meta['cv_std']:.4f} · "
           f"Teste ROC-AUC {meta['test_metrics']['roc_auc']:.4f}")

tab_pred, tab_val, tab_about = st.tabs(["🔮 Prever", "✅ Validar Amostra do Teste de Consistência", "ℹ️ Sobre o modelo"])

# ---- aba 1: previsão -------------------------------------------------------------------------------------
with tab_pred:
    st.subheader("Ajuste os dados na barra lateral e clique em **Prever**")
    if st.button("Prever", type="primary", width="stretch"):
        pred, proba = predict(current_inputs())
        left, right = st.columns([3, 2])
        with left:
            cls = "result-churn" if pred == 1 else "result-stay"
            msg = "Risco alto de cancelamento" if pred == 1 else "Cliente tende a permanecer"
            st.markdown(
                f'<div class="result-card {cls}"><h2>{msg}</h2>'
                f"<p>Classe prevista: <b>{'Churn (1)' if pred else 'Permanece (0)'}</b> · limiar = {THRESHOLD:.2f}</p></div>",
                unsafe_allow_html=True,
            )
            st.progress(min(max(proba, 0.0), 1.0), text=f"Probabilidade de churn: {proba:.1%}")
        with right:
            c1, c2 = st.columns(2)
            c1.metric("P(churn)", f"{proba:.1%}")
            c2.metric("P(permanece)", f"{1 - proba:.1%}")
            st.caption("Dica: para priorizar retenção, considere um limiar menor que 0,5 (maior recall).")
        with st.expander("Ver entradas enviadas ao modelo"):
            st.dataframe(pd.DataFrame([current_inputs()]).T.astype(str).rename(columns={0: "valor"}), width="stretch")

# ---- aba 2: paridade com o notebook ----------------------------------------------------------------------
with tab_val:
    st.subheader("Teste de paridade app × notebook")
    st.write("Escolha uma das 5 amostras reais de `D_teste` usadas no notebook. O botão **preenche os inputs** da "
             "barra lateral com a amostra, roda o pipeline do app e compara classe e probabilidade com as do notebook.")
    names = [f"{i + 1}. {s['caso']}  (índice {s['indice_teste']})" for i, s in enumerate(SAMPLES)]
    choice = st.selectbox("Amostra", range(len(SAMPLES)), format_func=lambda i: names[i], key="sample_choice")
    st.button("Validar Amostra do Teste de Consistência", type="primary", on_click=fill_sample,
              args=(choice,), width="stretch")

    idx = st.session_state.get("validated_idx")
    if idx is not None:
        sample = SAMPLES[idx]
        ui_row = current_inputs()                                 # exatamente o que está nos widgets
        inputs_match = all(ui_row[c] == sample["inputs"][c] for c in FEATURES)
        pred, proba = predict(ui_row)
        same_class = pred == sample["y_pred"]
        same_proba = proba == sample["proba"]
        ok = inputs_match and same_class and same_proba

        st.markdown(f"**Amostra validada:** {names[idx]}")
        st.dataframe(pd.DataFrame({
            "": ["Classe prevista (ŷ)", "P(churn)", "y real"],
            "Notebook": [str(sample["y_pred"]), repr(sample["proba"]), str(sample["y_true"])],
            "App": [str(pred), repr(proba), str(sample["y_true"])],
            "Idêntico?": ["✅" if same_class else "❌", "✅" if same_proba else "❌", "—"],
        }).set_index(""), width="stretch")
        if ok:
            st.markdown('<div class="parity-ok">✅ <b>PARIDADE 100%</b>: inputs preenchidos corretamente, mesma classe e '
                        'mesma probabilidade (diferença = 0) entre o app e o notebook.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="parity-bad">❌ <b>DIVERGÊNCIA</b>: inputs iguais = {inputs_match}, classe igual = '
                        f'{same_class}, |Δ prob| = {abs(proba - sample["proba"]):.3e}</div>', unsafe_allow_html=True)
        if st.session_state.get("sample_choice") != idx:
            st.info("Você mudou a seleção — clique no botão novamente para validar a nova amostra.")

# ---- aba 3: sobre -----------------------------------------------------------------------------------------
with tab_about:
    st.subheader("Resumo do modelo")
    m = meta["test_metrics"]
    cols = st.columns(6)
    for c, (k, label) in zip(cols, [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("accuracy", "Acurácia"),
                                    ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]):
        c.metric(f"{label} (teste)", f"{m[k]:.3f}")
    st.markdown(
        "* **Dados:** Telco Customer Churn (IBM sample data), 7.043 clientes.\n"
        "* **Protocolo:** split 80/20 estratificado, CV 5-fold nos mesmos folds para RF, XGBoost e LightGBM, "
        "tuning com Grid Search e Optuna, seleção pela regra 1-SE (menor gap).\n"
        "* **Pipeline único:** limpeza → imputação/padronização/One-Hot → modelo (sem data leakage)."
    )
