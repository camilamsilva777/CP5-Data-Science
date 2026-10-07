# Roteiro de Apresentação — 10 minutos

**Checkpoint 5 · Random Forest × XGBoost × LightGBM · Previsão de Churn**
Grupo: [Nome do Aluno 1] (RM1), [Nome do Aluno 2] (RM2), [Nome do Aluno 3] (RM3), [Nome do Aluno 4] (RM4), [Nome do Aluno 5] (RM5)

> Sugestão de divisão de falas: Slides 1–2 (aluno 1), 3–4 (aluno 2), 5–6 (aluno 3), 7–8 (aluno 4), 9–10 (aluno 5). Tempo total: **10:00**.

| Slide | Tempo | Acumulado |
|---|---|---|
| 1. Abertura | 0:30 | 0:30 |
| 2. Problema e dados | 1:00 | 1:30 |
| 3. EDA e wrangling | 1:00 | 2:30 |
| 4. Metodologia (anti-leakage) | 1:00 | 3:30 |
| 5. Baselines | 1:00 | 4:30 |
| 6. Tuning: Grid × Optuna | 1:30 | 6:00 |
| 7. Seleção do campeão | 1:00 | 7:00 |
| 8. Generalização e teste final | 1:00 | 8:00 |
| 9. Demo Streamlit + paridade | 1:45 | 9:45 |
| 10. Conclusões | 0:15 | 10:00 |

---

## Slide 1 — Abertura (0:00–0:30)
* Título, integrantes.
* **Fala:** "Comparamos Random Forest, XGBoost e LightGBM para prever cancelamento de clientes de telecom, usando exatamente o mesmo protocolo experimental para os três, e entregamos o campeão em uma aplicação Streamlit."

## Slide 2 — Problema e dados (0:30–1:30)
* Churn = cliente cancelou (1) ou não (0); **7.043 clientes × 21 colunas**, Telco Customer Churn (IBM sample data); ~26,5% de churn.
* Uso prático: lista priorizada de clientes em risco para o time de retenção (probabilidade, não só classe).
* **Métrica principal: ROC-AUC** — classes desbalanceadas (acurácia enganosa: "ninguém cancela" já dá 74%), independe de limiar, mede qualidade do ranking de risco. Secundárias: F1, precision, recall, PR-AUC.

## Slide 3 — EDA e wrangling (1:30–2:30)
* Problemas encontrados: `TotalCharges` como texto (11 brancos, todos `tenure=0` → 0), `SeniorCitizen` 0/1 tratado como categoria, `customerID` removido, 22 duplicatas de perfil removidas → **7.021 × 20**, 0 nulos, 0 duplicatas.
* Achados (gráficos do notebook): contrato mensal tem **42,6%** de churn vs **2,8%** no de 2 anos; fibra óptica 41,8%; cheque eletrônico 45,1%; mediana de tenure de quem cancela = 10 meses vs 38.

## Slide 4 — Metodologia sem vazamento (2:30–3:30)
* Split 80/20 estratificado, `random_state=42`; **D_teste guardado** até o final.
* **StratifiedKFold(5, shuffle, 42)** único, usado por baselines, Grid Search, Optuna, curva de aprendizado → os 3 algoritmos nos **mesmos folds**.
* Imputer, scaler e One-Hot dentro de `Pipeline`/`ColumnTransformer` → nada aprendido fora do treino de cada fold. Limpeza de tipos em um transformador sem estado (`TelcoCleaner`).

## Slide 5 — Baselines (3:30–4:30)
* CV ROC-AUC: RF 0,8235 · XGB 0,8216 · **LGBM 0,8332**.
* **Todos com overfitting severo:** treino 1,00 / 0,99 / 0,96 → gap 0,18 / 0,17 / 0,13. Nenhum underfitting.
* **Mensagem:** o problema não é falta de capacidade, é falta de regularização → tuning.

## Slide 6 — Tuning: Grid Search × Optuna (4:30–6:00)
* 6 buscas: Grid (32/24/24 combinações) e Optuna (30 trials TPE) para cada modelo, mesma métrica e mesmos folds.
* Resultados CV: RF 0,8491 / 0,8493 · XGB 0,8511 / 0,8517 · LGBM 0,8493 / 0,8517 (Grid / Optuna).
* **Comparação:** qualidade equivalente (diferenças ≪ desvio entre folds ≈ 0,017); Optuna tem custo fixo e explora espaço contínuo maior; estável entre sementes (desvio ≈ 3–6·10⁻⁴); Grid é determinístico mas explode com mais hiperparâmetros.
* Gap caiu de 0,13–0,18 para **0,014–0,030**.

## Slide 7 — Seleção do campeão (6:00–7:00)
* Tabela das **9 configurações** (treino, CV, gap, std, tempo).
* Regra: dentro de 1 desvio do melhor CV (6 de 9) → **menor gap**. Campeão: **M_XGB / Grid Search** — CV 0,8511 ± 0,0176, **gap 0,014** (menor de todos), diferença para o melhor CV absoluto de apenas 0,0006.
* Curva de aprendizado: treino e validação próximos, validação quase em platô → mais dados trariam pouco; não há under nem overfitting relevante. Importância: `Contract`, `OnlineSecurity`, `InternetService`, `TechSupport`, `PaymentMethod`.

## Slide 8 — Generalização e teste final (7:00–8:00)
* Retreino com 100% de D_treino; **uma única avaliação** em D_teste: **ROC-AUC 0,8436** (IC95% 0,821–0,865) vs CV 0,8511 ± 0,0176 → generalização consistente, sem vazamento.
* F1 0,587, recall 0,52, precision 0,67 no limiar 0,5 — em produção, reduzir o limiar para ganhar recall.
* **Teste de consistência (5 amostras reais):** acerto expressivo de churn (p=0,93), acerto expressivo de permanência (p=0,006), fronteira (p=0,500), erro mais confiante (p=0,013 para um churner real) e um acerto típico.

## Slide 9 — Demo Streamlit + paridade (8:00–9:45)
1. `streamlit run app.py` → inputs na barra lateral com **mediana/moda** como padrão (0:20).
2. Alterar tenure, contrato mensal, fibra óptica → **Prever** → classe + probabilidade (0:40).
3. Aba **"Validar Amostra do Teste de Consistência"** → escolher a amostra "Fronteira" → botão preenche os inputs e mostra a tabela **Notebook × App**: mesma classe e **mesma probabilidade (diferença = 0)** → "PARIDADE 100%" (0:45).
4. Repetir rapidamente com o "Erro mais confiante" para mostrar que a paridade vale também quando o modelo erra (0:20).
* **Fala:** "A paridade prova que o pipeline serializado — limpeza, pré-processamento e modelo — é o mesmo do notebook."

## Slide 10 — Conclusões (9:45–10:00)
* Protocolo único e sem vazamento; tuning reduziu overfitting de ~0,15 para ~0,015–0,03 de gap; M_XGB / Grid Search promovido; teste confirma 0,84 de AUC.
* Próximos passos: limiar por custo de negócio, calibração, novas variáveis (uso, suporte), monitoramento de drift.
* "Obrigado — perguntas?"

---

### Possíveis perguntas e respostas curtas
* **Por que não escolheram o maior CV?** Diferença de 0,0006 ≪ desvio entre folds; preferimos o menor gap (menos risco em produção).
* **Por que ROC-AUC?** Desbalanceamento + independência de limiar.
* **Como garantem ausência de leakage?** Tudo que aprende parâmetros está no Pipeline; D_teste só usado no Exercício 7, uma vez.
* **Por que `tenure` tem pouca importância no XGBoost?** Árvores rasas capturam o efeito primeiro via `Contract`, fortemente correlacionado com `tenure`; importância por ganho é dividida entre variáveis correlacionadas.
