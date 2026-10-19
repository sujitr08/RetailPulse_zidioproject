# RetailPulse Project Tracker

Last reviewed: 2026-10-18
Source of truth: Zidio Data Science & Analytics project brief plus execution evidence recorded in the project conversation.

## Status definitions
- PASS: executed successfully and evidence exists.
- IN_PROGRESS: implementation exists, but end-to-end acceptance is not yet verified.
- BLOCKED: a dependency or missing source data prevents a valid result.
- NOT_STARTED: no verified implementation evidence yet.
- NOT_MET: the implementation was evaluated, but the acceptance target was not reached.

## Week 1 — Data Exploration & Preparation

| Task | Status | Evidence / notes |
|---|---|---|
| Dataset selection and initial EDA | PASS | Online Retail workbook, 1,067,371 source rows; EDA report and seven figures exist. |
| Data cleaning and feature engineering | PASS | `cleanData.py`; `cleanedTransactions.csv` has 1,067,371 rows and 24 columns; quality flags retained. |
| Great Expectations validation | PASS | 36 checks passed, 0 failed. |
| K-Means and DBSCAN segmentation | PASS | Evaluation CSVs, customer segment output, PCA visualizations and report exist. |
| Time-series preparation and stationarity/decomposition | PASS | 739 daily observations, ADF report, decomposition and ACF figures exist. |
| Prophet baseline | PASS | Holdout metrics and forecast artifacts exist; MAPE 27.50%, WAPE 37.02%. |
| LSTM baseline | PASS | Model and holdout artifacts exist; MAPE 51.83%, WAPE 65.43%. |
| Week 1 checkpoint: EDA, cleaned data, baseline models logged in MLflow | IN_PROGRESS | Artifacts exist; confirm MLflow runs/artifacts and final reproducibility before closing checkpoint. |

## Week 2 — Advanced Modeling & Churn Prediction

| Task | Status | Evidence / notes |
|---|---|---|
| Prophet + LSTM ensemble | PASS | Evaluation completed; ensemble MAPE 39.61%, worse than Prophet. Keep Prophet as the best current baseline. |
| XGBoost churn model with SHAP | PASS | Model, SHAP plots, metrics and risk predictions exist. |
| Temporal churn validation | PASS | ROC-AUC 0.7633; precision@top 20% 0.7381. These do not meet the 0.88 and 0.75 targets, respectively. |
| Inventory optimization baseline | PASS | Recommendations for 4,916 products. Stock-on-hand and unit cost unavailable, so actual order quantities/savings cannot be calculated. |
| Demand confidence/spike audit | PASS | 378 of 4,916 products flagged for review; no recommendation files overwritten. |
| Transaction-level spike inspection | PASS | PROD02919 is a single 80,995-unit transaction; flags show candidate sale, not cancellation/duplicate. Commercial validity remains unverified. |
| Feature importance and model tuning with Optuna | IN_PROGRESS | Feature importance and SHAP exist; Optuna tuning not verified. |
| Evidently drift detection | IN_PROGRESS | Offline reports exist. Sales drift is historical distribution shift, not proof of model degradation; churn score split is not chronological. |
| Automated retraining with Airflow | NOT_STARTED | No verified DAG execution. |
| Week 2 checkpoint | IN_PROGRESS | Several modeling components exist, but tuning and retraining remain outstanding and acceptance targets are unmet. |

## Week 3 — Dashboard & Analytics

| Task | Status | Evidence / notes |
|---|---|---|
| Streamlit multi-page dashboard | IN_PROGRESS | `app.py` created; successful runtime and functional tests not yet confirmed. |
| Forecast visualization and what-if analysis | IN_PROGRESS | Forecast plots exist; dashboard controls and output behavior need verification. |
| Segmentation and churn dashboard | IN_PROGRESS | Models/data exist; dashboard behavior needs verification. |
| Inventory recommendations UI | IN_PROGRESS | Recommendations exist; outlier/confidence review should be surfaced. |
| Real-time metrics and alerts | NOT_STARTED | Offline monitoring reports are not equivalent to live alerting. |
| CSV/PDF export functionality | IN_PROGRESS | Must test from the running dashboard. |
| Week 3 checkpoint | IN_PROGRESS | Do not close until dashboard functions are tested end-to-end. |

## Week 4 — Deployment & Production Polish

| Task | Status | Evidence / notes |
|---|---|---|
| Docker multi-stage build | NOT_STARTED | No verified build. |
| Kubernetes manifests and deployment | NOT_STARTED | No verified deployment. |
| GitHub Actions CI/CD | NOT_STARTED | No verified workflow. |
| AWS/GCP or other public HTTPS deployment | NOT_STARTED | Public demo URL not yet verified. |
| Prometheus/Grafana monitoring | NOT_STARTED | Evidently reports exist; Prometheus/Grafana not verified. |
| Load testing and final accuracy validation | NOT_STARTED | Not verified. |
| Final QA, README, demo video, PDF | NOT_STARTED | Submission artifacts and links need to be produced and checked. |

## Acceptance targets

| Metric | Target | Current measured result | Status |
|---|---:|---:|---|
| Forecast MAPE, 30-day horizon | <= 12% | Prophet 27.50%; LSTM 51.83%; equal-weight ensemble 39.61% | NOT_MET |
| Churn ROC-AUC | >= 0.88 | Temporal validation 0.7633 | NOT_MET |
| Churn precision at top 20% | >= 0.75 | Temporal validation 0.7381 | NOT_MET |
| Inventory reduction / stockout reduction / revenue improvement | Demonstrated measured improvement | No on-hand stock, verified costs, or outcome data | BLOCKED |
| 10M+ transactions/month; daily batch under 5 min | Demonstrated performance | Not load-tested | NOT_STARTED |
| MLflow tracking and drift observability | Runs/artifacts and meaningful monitoring | MLflow DB and Evidently reports exist; full tracking/reproducibility audit pending | IN_PROGRESS |

## Important data and interpretation limitations
- Dataset is historical and ends 2011-12-09; it is not live retail data.
- The source has 243,007 missing customer IDs and lacks populated unit cost and stock-on-hand values.
- Cross-sheet duplicate copies are flagged and retained in the source; analysis policies are documented.
- Forecast MAPE is computed on non-zero holdout days; always state this alongside the metric.
- The sales drift report compares historical time periods; it does not prove model degradation.
- Churn score-distribution comparison used a row-order split, not a chronological split; do not label it true temporal drift.
- Inventory recommendations are planning estimates, not purchase orders or demonstrated business savings.
- Do not claim any unmet metric or business-impact target as achieved.

## Next action
1. Run and test the Streamlit dashboard locally.
2. Fix runtime errors and validate each page, interactive control, and export.
3. Only then close Week 3 and continue with Optuna/Airflow and deployment work.
