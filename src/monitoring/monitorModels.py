"""RetailPulse historical model/data monitoring. Run from project root:
python src/monitoring/monitorModels.py
"""
from __future__ import annotations
import json, logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / 'data' / 'processedData'
REPORTS = ROOT / 'reports' / 'modelReports'
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger('retailPulse.monitoring')


def safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)): return value.isoformat()
    if value is None or (isinstance(value, float) and np.isnan(value)): return None
    return value


def read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists(): raise FileNotFoundError(f'Required file not found: {path}')
    df = pd.read_csv(path)
    missing = sorted(set(columns) - set(df.columns))
    if missing: raise ValueError(f'{path.name} missing columns: {missing}')
    return df


def regression_metrics(df: pd.DataFrame, actual_col: str, pred_col: str) -> dict:
    pair = df[[actual_col, pred_col]].apply(pd.to_numeric, errors='coerce').dropna()
    if pair.empty: return {'observations': 0, 'status': 'NO_VALID_PAIRED_ROWS'}
    actual, pred = pair[actual_col].to_numpy(float), pair[pred_col].to_numpy(float)
    err = pred - actual
    nonzero = np.abs(actual) > 1e-9
    return {'observations': int(len(pair)), 'mae': float(np.mean(np.abs(err))),
            'rmse': float(np.sqrt(np.mean(err ** 2))),
            'mape_nonzero_actual_percent': float(np.mean(np.abs(err[nonzero] / actual[nonzero])) * 100) if nonzero.any() else None,
            'wape_percent': float(np.sum(np.abs(err)) / np.sum(np.abs(actual)) * 100) if np.sum(np.abs(actual)) > 1e-9 else None,
            'mean_error_forecast_minus_actual': float(np.mean(err)), 'status': 'PASS'}


def forecast_monitoring() -> tuple[list[dict], dict]:
    specs = [('Prophet', 'prophetHoldoutForecast.csv', 'actualSales', 'forecastSales', 'ds'),
             ('LSTM', 'lstmHoldoutForecast.csv', 'actualSales', 'lstmForecast', 'date'),
             ('Equal-weight ensemble', 'ensembleHoldoutForecast.csv', 'actualSales', 'ensembleForecast', 'date')]
    results = []
    for model, filename, actual, pred, datecol in specs:
        path = PROCESSED / filename
        if not path.exists():
            results.append({'model': model, 'status': 'MISSING_FILE', 'file': filename}); continue
        df = read_csv(path, [actual, pred, datecol])
        item = regression_metrics(df, actual, pred)
        item.update({'model': model, 'source_file': str(path.relative_to(ROOT)),
                     'date_start': str(df[datecol].min()), 'date_end': str(df[datecol].max())})
        results.append(item)
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(REPORTS / 'forecastMonitoringMetrics.csv', index=False)
    valid = [r for r in results if isinstance(r.get('wape_percent'), (int, float))]
    return results, {'models_evaluated': len(valid), 'best_by_wape_on_existing_holdout': min(valid, key=lambda r:r['wape_percent'])['model'] if valid else None}


def churn_monitoring() -> dict:
    path = REPORTS / 'churnTemporalRiskPredictions.csv'
    if not path.exists(): return {'status':'MISSING_FILE', 'source_file':str(path.relative_to(ROOT))}
    df = read_csv(path, ['churned','churnProbability','predictedChurnAt0_5'])
    for col in ['churned','churnProbability','predictedChurnAt0_5']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['churned','churnProbability','predictedChurnAt0_5'])
    y = df.churned.astype(int).to_numpy(); scores = df.churnProbability.to_numpy(float); pred = df.predictedChurnAt0_5.astype(int).to_numpy()
    auc = float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else None
    ap = float(average_precision_score(y, scores)) if len(np.unique(y)) > 1 else None
    result = {'status':'HISTORICAL_TEMPORAL_HOLDOUT_REASSESSMENT', 'source_file':str(path.relative_to(ROOT)),
              'observations':int(len(df)), 'observed_churn_rate_percent':float(y.mean()*100) if len(y) else None,
              'mean_predicted_churn_probability_percent':float(scores.mean()*100) if len(scores) else None,
              'roc_auc':auc, 'average_precision':ap,
              'precision_at_0_5':float(precision_score(y,pred,zero_division=0)),
              'recall_at_0_5':float(recall_score(y,pred,zero_division=0)),
              'f1_at_0_5':float(f1_score(y,pred,zero_division=0)),
              'note':'Reassessment of the existing labeled temporal holdout, not a new independent test.'}
    (REPORTS/'churnMonitoringMetrics.json').write_text(json.dumps(safe(result),indent=2),encoding='utf-8')
    return result



def distribution_drift_metrics(reference: pd.Series, current: pd.Series) -> dict:
    """Quantify numeric distribution shift; thresholds are investigation heuristics, not proof of model failure."""
    ref = pd.to_numeric(reference, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    cur = pd.to_numeric(current, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if len(ref) < 10 or len(cur) < 10:
        return {"status": "INSUFFICIENT_DATA", "reference_valid_rows": int(len(ref)),
                "current_valid_rows": int(len(cur)),
                "note": "At least 10 valid observations per window are required for this diagnostic."}

    ks = ks_2samp(ref, cur, alternative="two-sided", method="auto")
    pooled_std = float(np.sqrt((np.var(ref, ddof=1) + np.var(cur, ddof=1)) / 2))
    standardized_mean_shift = float((np.mean(cur) - np.mean(ref)) / pooled_std) if pooled_std > 1e-12 else (
        0.0 if np.isclose(np.mean(ref), np.mean(cur)) else None)

    # PSI bins are anchored to reference quantiles; clipping avoids log(0).
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        edges = np.unique(np.concatenate(([-np.inf], np.quantile(ref, [0.25, 0.5, 0.75]), [np.inf])))
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    p_value = float(ks.pvalue)

    if psi >= 0.20 or (p_value < 0.05 and standardized_mean_shift is not None and abs(standardized_mean_shift) >= 0.20):
        status = "DRIFT_SIGNAL"
    elif psi >= 0.10 or p_value < 0.05:
        status = "POSSIBLE_SHIFT"
    else:
        status = "NO_STRONG_SIGNAL"

    return {
        "status": status, "reference_valid_rows": int(len(ref)), "current_valid_rows": int(len(cur)),
        "reference_mean": float(np.mean(ref)), "current_mean": float(np.mean(cur)),
        "reference_median": float(np.median(ref)), "current_median": float(np.median(cur)),
        "reference_std": float(np.std(ref, ddof=1)), "current_std": float(np.std(cur, ddof=1)),
        "ks_statistic": float(ks.statistic), "ks_p_value": p_value,
        "standardized_mean_shift": standardized_mean_shift, "population_stability_index": psi,
        "thresholds": {"psi_possible_shift": 0.10, "psi_drift_signal": 0.20,
                       "ks_p_value": 0.05, "standardized_mean_shift_for_drift_signal": 0.20},
        "note": "Heuristic distribution diagnostics only. Statistical significance is sample-size sensitive and does not establish model degradation."
    }


def evidently_drift(reference: pd.DataFrame, current: pd.DataFrame, html_path: Path) -> dict:
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
        report = Report([DataDriftPreset()])
        result = report.run(reference_data=reference, current_data=current)
        result.save_html(str(html_path))
        payload = {'status':'REPORT_CREATED','reference_rows':len(reference),'current_rows':len(current),
                   'html_report':str(html_path.relative_to(ROOT))}
        try:
            raw = result.json(); payload['evidently_result'] = json.loads(raw) if isinstance(raw,str) else raw
        except Exception as exc: payload['result_json_note'] = str(exc)
        return payload
    except Exception as exc:
        log.exception('Evidently report failed: %s', html_path.name)
        return {'status':'FAILED','error':f'{type(exc).__name__}: {exc}', 'reference_rows':len(reference), 'current_rows':len(current)}


def sales_drift() -> dict:
    path = PROCESSED/'dailySales.csv'
    if not path.exists(): return {'status':'MISSING_FILE'}
    df = read_csv(path,['date','dailySales']); df['date'] = pd.to_datetime(df.date,errors='coerce'); df['dailySales'] = pd.to_numeric(df.dailySales,errors='coerce')
    df = df.dropna(subset=['date','dailySales']).sort_values('date').reset_index(drop=True)
    if len(df)<20: return {'status':'INSUFFICIENT_DATA','rows':len(df)}
    cut = int(len(df)*.7); ref=df.iloc[:cut][['dailySales']].copy(); cur=df.iloc[cut:][['dailySales']].copy()
    out = evidently_drift(ref,cur,REPORTS/'salesDataDriftReport.html')
    out['distribution_metrics'] = distribution_drift_metrics(ref['dailySales'], cur['dailySales'])
    out.update({'comparison':'Earlier 70% of daily sales vs later 30%', 'date_reference_start':str(df.date.iloc[0].date()),
                'date_reference_end':str(df.date.iloc[cut-1].date()),'date_current_start':str(df.date.iloc[cut].date()),
                'date_current_end':str(df.date.iloc[-1].date()),'feature':'dailySales',
                'caveat':'Historical temporal drift analysis, not live production monitoring.'})
    return out


def churn_score_drift() -> dict:
    path=REPORTS/'churnTemporalRiskPredictions.csv'
    if not path.exists(): return {'status':'MISSING_FILE'}
    df=read_csv(path,['churnProbability']); df['churnProbability']=pd.to_numeric(df.churnProbability,errors='coerce'); df=df.dropna(subset=['churnProbability'])
    if len(df)<20: return {'status':'INSUFFICIENT_DATA','rows':len(df)}
    cut=int(len(df)*.7); ref=df.iloc[:cut][['churnProbability']].copy(); cur=df.iloc[cut:][['churnProbability']].copy()
    out=evidently_drift(ref,cur,REPORTS/'churnScoreDriftReport.html')
    out['distribution_metrics'] = distribution_drift_metrics(ref['churnProbability'], cur['churnProbability'])
    out.update({'comparison':'First 70% vs last 30% of rows in saved prediction file', 'feature':'churnProbability',
                'caveat':'Rows are not a time-ordered production stream; this is a score-distribution diagnostic only.'})
    return out


def main() -> int:
    REPORTS.mkdir(parents=True,exist_ok=True)
    errors=[]
    try: forecasts, forecast_summary=forecast_monitoring()
    except Exception as exc: log.exception('Forecast monitoring failed'); forecasts=[]; forecast_summary={'status':'FAILED'}; errors.append(f'Forecast: {type(exc).__name__}: {exc}')
    try: churn=churn_monitoring()
    except Exception as exc: log.exception('Churn metrics failed'); churn={'status':'FAILED','error':str(exc)}; errors.append(f'Churn: {type(exc).__name__}: {exc}')
    try: sales=sales_drift()
    except Exception as exc: log.exception('Sales drift failed'); sales={'status':'FAILED','error':str(exc)}; errors.append(f'Sales drift: {type(exc).__name__}: {exc}')
    try: score=churn_score_drift()
    except Exception as exc: log.exception('Score drift failed'); score={'status':'FAILED','error':str(exc)}; errors.append(f'Score drift: {type(exc).__name__}: {exc}')
    statuses={'forecast_metrics':'PASS' if any(x.get('observations',0) for x in forecasts) else 'REVIEW',
              'churn_metrics':'PASS' if churn.get('observations',0) else 'REVIEW',
              'sales_data_drift':sales.get('status','UNKNOWN'),'churn_score_distribution':score.get('status','UNKNOWN')}
    failed=bool(errors) or any(v in {'FAILED','MISSING_FILE','INSUFFICIENT_DATA'} for v in statuses.values())
    summary={'project':'RetailPulse','generated_at_utc':datetime.now(timezone.utc).isoformat(),
             'assessment_type':'Historical offline monitoring; not live production monitoring',
             'overall_status':'REVIEW_REQUIRED' if failed else 'COMPLETED_WITH_CAVEATS','statuses':statuses,
             'forecast_summary':forecast_summary,'forecast_metrics':forecasts,'churn_metrics':churn,
             'sales_data_drift':sales,'churn_score_distribution':score,'errors':errors,
             'limitations':['Source transaction data ends on 2011-12-09.','Historical comparisons are not live monitoring.',
             'Churn score distribution comparison is not time-ordered and is diagnostic only.',
             'Forecast metrics use the existing 30-day holdout, not a new independent test.',
             'The source data lacks reliable on-hand stock and unit-cost fields.']}
    (REPORTS/'modelMonitoringSummary.json').write_text(json.dumps(safe(summary),indent=2),encoding='utf-8')
    lines=['RetailPulse Model Monitoring Report','='*36,f"Generated (UTC): {summary['generated_at_utc']}",f"Assessment: {summary['assessment_type']}",f"Overall status: {summary['overall_status']}",'','Status summary:']
    lines += [f'- {k}: {v}' for k,v in statuses.items()]
    lines += ['', 'Forecast holdout metrics:']
    for m in forecasts: lines.append(f"- {m.get('model','Unknown')}: MAE={m.get('mae','N/A')}, RMSE={m.get('rmse','N/A')}, MAPE={m.get('mape_nonzero_actual_percent','N/A')}%, WAPE={m.get('wape_percent','N/A')}%")
    sales_dm = sales.get('distribution_metrics', {}) if isinstance(sales, dict) else {}
    score_dm = score.get('distribution_metrics', {}) if isinstance(score, dict) else {}
    lines += ['', 'Churn temporal holdout reassessment:',
              f"- ROC-AUC: {churn.get('roc_auc','N/A')}",
              f"- Average precision: {churn.get('average_precision','N/A')}",
              f"- Precision at 0.5: {churn.get('precision_at_0_5','N/A')}",
              '', 'Sales drift:', f"- Status: {sales.get('status','UNKNOWN')}",
              f"- Report: {sales.get('html_report','Not created')}"]
    if sales_dm:
        lines += [f"- Quantitative drift status: {sales_dm.get('status','N/A')}",
                  f"- KS statistic: {sales_dm.get('ks_statistic','N/A')}",
                  f"- KS p-value: {sales_dm.get('ks_p_value','N/A')}",
                  f"- Population Stability Index (PSI): {sales_dm.get('population_stability_index','N/A')}",
                  f"- Standardized mean shift: {sales_dm.get('standardized_mean_shift','N/A')}"]
    lines += ['', 'Churn score distribution diagnostic:',
              f"- Status: {score.get('status','UNKNOWN')}",
              f"- Report: {score.get('html_report','Not created')}"]
    if score_dm:
        lines += [f"- Diagnostic status: {score_dm.get('status','N/A')}",
                  f"- KS statistic: {score_dm.get('ks_statistic','N/A')}",
                  f"- KS p-value: {score_dm.get('ks_p_value','N/A')}",
                  f"- Population Stability Index (PSI): {score_dm.get('population_stability_index','N/A')}",
                  f"- Standardized mean shift: {score_dm.get('standardized_mean_shift','N/A')}"]
    lines += ['', 'Limitations:']
    lines += [f'- {x}' for x in summary['limitations']]
    if errors: lines += ['', 'Errors:']+[f'- {e}' for e in errors]
    (REPORTS/'modelMonitoringReport.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    log.info('Summary: %s',REPORTS/'modelMonitoringSummary.json'); log.info('Report: %s',REPORTS/'modelMonitoringReport.txt'); log.info('Overall status: %s',summary['overall_status'])
    return 1 if errors else 0

if __name__=='__main__': raise SystemExit(main())
