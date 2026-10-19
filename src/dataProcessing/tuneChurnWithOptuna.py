import json
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/processedData/cleanedTransactions.csv'
REPORTS = ROOT / 'reports/modelReports'
MODELS = ROOT / 'models'
FEATURE_DAYS, LABEL_DAYS, TRIALS, SEED = 365, 90, 30, 42
FEATURES = ['recencyDays','frequency','uniquePurchaseDays','monetaryTotal','averageOrderValue','quantityTotal','transactionRows']

def load_sales():
    needed = {'customerId','invoiceDate','quantity','SalesAmount','isCandidateSale','isCrossSheetDuplicate'}
    df = pd.read_csv(DATA, usecols=lambda c: c in needed, low_memory=False)
    missing = needed - set(df.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    df['invoiceDate'] = pd.to_datetime(df['invoiceDate'], errors='coerce')
    for c in ['customerId','quantity','SalesAmount']: df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in ['isCandidateSale','isCrossSheetDuplicate']:
        df[c] = df[c].astype(str).str.lower().isin(['true','1','1.0'])
    df = df[df.customerId.notna() & df.invoiceDate.notna() & df.isCandidateSale & ~df.isCrossSheetDuplicate & (df.quantity > 0) & (df.SalesAmount > 0)].copy()
    df['customerId'] = df.customerId.astype('int64')
    df['day'] = df.invoiceDate.dt.normalize()
    return df.sort_values('day')

def snapshot(sales, cutoff):
    start, end = cutoff-pd.Timedelta(days=FEATURE_DAYS), cutoff+pd.Timedelta(days=LABEL_DAYS)
    hist = sales[(sales.day > start) & (sales.day <= cutoff)]
    future = sales[(sales.day > cutoff) & (sales.day <= end)]
    if hist.empty: return pd.DataFrame()
    g = hist.groupby('customerId').agg(lastPurchase=('day','max'), frequency=('day','size'), uniquePurchaseDays=('day','nunique'), monetaryTotal=('SalesAmount','sum'), averageOrderValue=('SalesAmount','mean'), quantityTotal=('quantity','sum'), transactionRows=('day','size'))
    g['recencyDays'] = (cutoff-g.lastPurchase).dt.days
    g['featureCutoff'] = cutoff
    g['churned'] = (~g.index.isin(set(future.customerId))).astype(int)
    g[FEATURES] = g[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0)
    return g.reset_index()[['customerId','featureCutoff',*FEATURES,'churned']]

def metrics(y, prob):
    pred = (prob >= .5).astype(int); top = max(1,int(np.ceil(len(prob)*.2))); idx = np.argsort(prob)[::-1][:top]
    return {'rocAuc':float(roc_auc_score(y,prob)), 'averagePrecision':float(average_precision_score(y,prob)), 'precisionAtTop20Percent':float(np.asarray(y)[idx].mean()), 'precisionAtThreshold05':float(precision_score(y,pred,zero_division=0)), 'recallAtThreshold05':float(recall_score(y,pred,zero_division=0)), 'f1AtThreshold05':float(f1_score(y,pred,zero_division=0))}

def main():
    REPORTS.mkdir(parents=True,exist_ok=True); MODELS.mkdir(parents=True,exist_ok=True)
    if not DATA.exists(): raise FileNotFoundError(f'Dataset not found: {DATA}')
    sales=load_sales(); last=sales.day.max()-pd.Timedelta(days=LABEL_DAYS)
    snaps=[snapshot(sales,c) for c in [last-pd.Timedelta(days=180),last-pd.Timedelta(days=90),last]]
    snaps=[s for s in snaps if not s.empty]
    if len(snaps)<3: raise ValueError('Could not create three temporal snapshots; inspect date range.')
    train=pd.concat(snaps[:-1],ignore_index=True); valid=snaps[-1]
    Xtr,ytr=train[FEATURES],train.churned.astype(int); Xv,yv=valid[FEATURES],valid.churned.astype(int)
    if ytr.nunique()<2 or yv.nunique()<2: raise ValueError('Train and validation must each contain both classes.')
    rows=[]
    def objective(trial):
        params={'n_estimators':trial.suggest_int('n_estimators',150,600,step=50),'max_depth':trial.suggest_int('max_depth',2,8),'learning_rate':trial.suggest_float('learning_rate',.01,.2,log=True),'min_child_weight':trial.suggest_float('min_child_weight',1,12),'subsample':trial.suggest_float('subsample',.65,1),'colsample_bytree':trial.suggest_float('colsample_bytree',.65,1),'gamma':trial.suggest_float('gamma',0,5),'reg_alpha':trial.suggest_float('reg_alpha',1e-5,5,log=True),'reg_lambda':trial.suggest_float('reg_lambda',1e-3,10,log=True)}
        model=XGBClassifier(**params,objective='binary:logistic',eval_metric='logloss',random_state=SEED,n_jobs=-1)
        model.fit(Xtr,ytr); m=metrics(yv.to_numpy(),model.predict_proba(Xv)[:,1]); rows.append({'trial':trial.number,**params,**m}); return m['rocAuc']
    print(f'Rows: {len(sales):,}; training snapshot rows: {len(train):,}; validation customers: {len(valid):,}')
    print(f'Train churn: {ytr.mean():.3%}; validation churn: {yv.mean():.3%}; trials: {TRIALS}')
    study=optuna.create_study(direction='maximize',study_name='retailPulseChurnTemporalTuning',sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective,n_trials=TRIALS,show_progress_bar=True)
    pd.DataFrame(rows).sort_values('rocAuc',ascending=False).to_csv(REPORTS/'churnOptunaTuningResults.csv',index=False)
    model=XGBClassifier(**study.best_params,objective='binary:logistic',eval_metric='logloss',random_state=SEED,n_jobs=-1); model.fit(Xtr,ytr)
    best=metrics(yv.to_numpy(),model.predict_proba(Xv)[:,1]); model_path=MODELS/'churnXgboostOptuna.json'; model.save_model(model_path)
    payload={'studyName':study.study_name,'trials':len(study.trials),'objective':'maximize temporal validation ROC-AUC','bestTrial':study.best_trial.number,'bestParameters':study.best_params,'validationMetrics':best,'features':FEATURES,'trainingSnapshotCutoffs':sorted(pd.to_datetime(train.featureCutoff).dt.strftime('%Y-%m-%d').unique().tolist()),'validationCutoff':str(pd.Timestamp(valid.featureCutoff.iloc[0]).date()),'trainingRows':len(train),'validationCustomers':len(valid),'trainingChurnRate':float(ytr.mean()),'validationChurnRate':float(yv.mean()),'limitations':['Only one latest temporal validation snapshot is used.','Customers may appear in multiple earlier training snapshots.','Compare measured ROC-AUC with the >=0.88 target; do not assume it is met.']}
    (REPORTS/'churnOptunaBestParameters.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    lines=['RetailPulse Optuna Churn Tuning Report','='*38,f'Trials: {len(study.trials)}',f'Best trial: {study.best_trial.number}',f'Training rows: {len(train):,}',f'Validation customers: {len(valid):,}',f'Training churn: {ytr.mean():.3%}',f'Validation churn: {yv.mean():.3%}','','Validation metrics:']+[f'- {k}: {v:.4f}' for k,v in best.items()]+['','Best parameters:']+[f'- {k}: {v}' for k,v in study.best_params.items()]+['','Caveats:']+[f'- {x}' for x in payload['limitations']]
    (REPORTS/'churnOptunaTuningReport.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\nCompleted. Best temporal ROC-AUC:',round(best['rocAuc'],4)); print('Precision@top20%:',round(best['precisionAtTop20Percent'],4)); print('Report:',REPORTS/'churnOptunaTuningReport.txt'); print('Model:',model_path)

if __name__=='__main__': main()
