# Checkpoint 5 — Random Forest × XGBoost × LightGBM (Previsão de Churn)

Data Science e Statistical Computing · FIAP 2026 · Prof. Jones Egydio

**Integrantes**

| Nome | RM |
|---|---|
| [Nome do Aluno 1] | [RM1] |
| [Nome do Aluno 2] | [RM2] |
| [Nome do Aluno 3] | [RM3] |
| [Nome do Aluno 4] | [RM4] |
| [Nome do Aluno 5] | [RM5] |

## Resumo do problema

Prever se um cliente de telecomunicações vai **cancelar o serviço (churn)** a partir de dados cadastrais, contratuais e de serviços, para priorizar ações de retenção. É um problema de **classificação binária desbalanceada** (~26,5% de churn), então a métrica principal é **ROC-AUC** (secundárias: acurácia, precision, recall, F1, PR-AUC).

* **Dados:** *Telco Customer Churn* (IBM sample data), 7.043 clientes × 21 colunas. Fonte: <https://github.com/IBM/telco-customer-churn-on-icp4d> (dados de amostra públicos da IBM, uso educacional; baixados automaticamente se `data/raw_dataset.csv` não existir).
* **Wrangling:** `TotalCharges` texto → numérico (11 brancos de clientes com `tenure=0` → 0), `SeniorCitizen` 0/1 → categoria, remoção de `customerID` e de 22 duplicatas → **7.021 × 20**, sem nulos e sem duplicatas.
* **Protocolo:** split 80/20 estratificado (`random_state=42`) → `D_treino` (5.616) e `D_teste` (1.405, usado **uma única vez**); `StratifiedKFold(5, shuffle=True, random_state=42)` **idêntico** para os 3 algoritmos e para Grid Search/Optuna; todo pré-processamento aprendido (imputer, scaler, One-Hot) dentro de `Pipeline`/`ColumnTransformer` → **sem data leakage**.
* **Tuning:** Grid Search **e** Optuna (30 trials, TPE) para RF, XGBoost e LightGBM → 6 buscas, 9 configurações.

## Resultados (ROC-AUC, 5-fold CV em `D_treino`)

| # | Modelo | Estratégia | Treino | CV (ROC-AUC) | Gap Δ | Std CV | Tempo treino (s) | Tempo busca (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | M_RF | Baseline | 1.0000 | 0.8235 | 0.1765 | 0.0149 | 0.37 | 0 |
| 2 | M_RF | Grid Search | 0.8758 | 0.8491 | 0.0267 | 0.0172 | 1.03 | 142 |
| 3 | M_RF | Optuna | 0.8791 | 0.8493 | 0.0298 | 0.0171 | 0.87 | 165 |
| 4 | M_XGB | Baseline | 0.9898 | 0.8216 | 0.1682 | 0.0182 | 0.37 | 0 |
| **5** | **M_XGB** | **Grid Search** | **0.8650** | **0.8511** | **0.0139** | **0.0176** | **0.72** | **90** |
| 6 | M_XGB | Optuna | 0.8721 | 0.8517 | 0.0204 | 0.0174 | 0.55 | 117 |
| 7 | M_LGBM | Baseline | 0.9628 | 0.8332 | 0.1296 | 0.0160 | 0.25 | 0 |
| 8 | M_LGBM | Grid Search | 0.8784 | 0.8493 | 0.0291 | 0.0173 | 0.25 | 62 |
| 9 | M_LGBM | Optuna | 0.8713 | 0.8517 | 0.0197 | 0.0170 | 0.19 | 46 |

**Campeão: `M_XGB / Grid Search`** (`max_depth=2`, `learning_rate=0.03`, `n_estimators=300`, `subsample=0.8`). Regra de promoção: entre as configurações a no máximo 1 desvio-padrão do melhor CV (6 de 9), escolhe-se o **menor gap treino−CV** (0,014). A diferença para o melhor CV absoluto (0,0006) é muito menor que o desvio entre folds (~0,017).

**Principais conclusões**

* Os três baselines sofrem **overfitting severo** (gap 0,13–0,18); o tuning o reduziu para 0,014–0,030 e elevou o CV de ~0,82–0,83 para ~0,85.
* Grid Search e Optuna chegam a soluções estatisticamente equivalentes; o Optuna tem custo fixo (30 trials) e explora um espaço maior, e foi estável entre sementes (desvio ≈ 3–6·10⁻⁴).
* **Teste final (`D_teste`, uma única avaliação):** ROC-AUC **0,8436** (IC95% bootstrap 0,821–0,865), PR-AUC 0,657, acurácia 0,806, precision 0,671, recall 0,522, F1 0,587 — consistente com o CV (0,8511 ± 0,0176), sem sinal de vazamento.

## Estrutura do projeto

```
.
├── data/raw_dataset.csv                      # Telco Customer Churn (bruto)
├── artifacts/
│   ├── best_pipeline.joblib                  # pipeline completo treinado (limpeza → pré-proc. → XGBoost)
│   ├── app_metadata.json                     # defaults (mediana/moda), opções e 5 amostras do teste de consistência
│   └── results_9_configs.csv                 # tabela de decisão
├── notebooks/Checkpoint05_RF_XGBoost_LightGBM.ipynb   # Exercícios 1–7 (executa de ponta a ponta)
├── src/
│   ├── __init__.py
│   ├── data_loader.py                        # download/carga, split 80/20, objeto de CV
│   ├── preprocessing.py                      # TelcoCleaner, ColumnTransformer, build_pipeline
│   └── evaluate.py                           # métricas, estilo gráfico, avaliação CV, regra do campeão
├── app.py                                    # interface Streamlit + teste de paridade
├── requirements.txt
├── README.md
└── presentation_script.md                    # roteiro cronometrado (10 min)
```

## Como executar localmente

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

**1. Notebook (gera `artifacts/`)** — Run All (~20 min, dominado por Grid Search e Optuna):

```bash
cd notebooks
python -m nbconvert --to notebook --execute --inplace Checkpoint05_RF_XGBoost_LightGBM.ipynb
# ou abra no Jupyter/VS Code e use "Run All"
```

**2. App Streamlit** (a partir da raiz do projeto):

```bash
streamlit run app.py
```

* Barra lateral: inputs dinâmicos (sliders, selects, numéricos) com **mediana/moda** do treino como padrão; botão **Prever** mostra classe e probabilidade.
* Aba **“Validar Amostra do Teste de Consistência”**: preenche os inputs com uma das 5 amostras reais de `D_teste` do notebook e verifica que classe e probabilidade são **idênticas** às do notebook.

> **Nota (Windows):** se o Controle de Aplicativo/Smart App Control bloquear DLLs do scikit-learn/scipy (`DLL load failed ... política de Controle de Aplicativo`), é preciso liberar a política ou executar em outro ambiente (WSL, Conda, Colab). `scipy==1.16.3` foi o usado nos testes.

## Reprodutibilidade

`random_state=42` em split, CV, modelos e `TPESampler(seed=42)`. Versões pinadas em `requirements.txt` (o pipeline serializado depende de `scikit-learn==1.9.1` e `xgboost==3.4.1`).
