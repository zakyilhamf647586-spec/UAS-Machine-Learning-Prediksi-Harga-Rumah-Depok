"""Reproducible, scoped house-price regression training.

This dataset has extreme anomalies. The explicit fixed ranges define the
in-scope population; metrics DO NOT apply to homes outside those ranges.
GroupShuffleSplit prevents identical feature vectors being shared across
train/test, but it cannot establish property-level identity.
"""
from __future__ import annotations
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
import sklearn
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
TARGET = 'Harga'
NUMERIC = ['Kamar Tidur', 'Kamar Mandi', 'Garasi', 'Luas Tanah', 'Luas Bangunan']
CATEGORICAL = ['Lokasi']
FEATURES = NUMERIC + CATEGORICAL
DATA = ROOT / 'dataset/Cleaned_Final_Rumah.csv'


def in_scope(df: pd.DataFrame) -> pd.Series:
    """Explicit *analytical scope*, not a claim about all properties in Depok.

    Designed to exclude clearly anomalous extremes; fixed ahead of splitting.
    Limits must be revisited if project scope or data collection changes.
    """
    return (
        df['Harga'].between(100_000_000, 10_000_000_000)
        & df['Luas Tanah'].between(20, 1500)
        & df['Luas Bangunan'].between(20, 1500)
        & df['Kamar Tidur'].between(1, 15)
        & df['Kamar Mandi'].between(1, 15)
        & df['Garasi'].between(0, 10)
        & df['Lokasi'].notna()
    )


def preprocess():
    return ColumnTransformer([
        ('numeric', 'passthrough', NUMERIC),
        ('location', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL)
    ], remainder='drop')


def score_model(model, X, y):
    yp = model.predict(X)
    return {
        'MAE_Rp': round(float(mean_absolute_error(y, yp)), 2),
        'RMSE_Rp': round(float(np.sqrt(mean_squared_error(y, yp))), 2),
        'R2': round(float(r2_score(y, yp)), 6),
    }


def main():
    df = pd.read_csv(DATA)
    assert set(FEATURES+[TARGET]).issubset(df.columns), 'Dataset has missing columns.'
    df = df[FEATURES+[TARGET]].copy()
    df[NUMERIC+[TARGET]] = df[NUMERIC+[TARGET]].apply(pd.to_numeric, errors='raise')
    df[CATEGORICAL] = df[CATEGORICAL].astype('string')
    mask = in_scope(df)
    clean = df.loc[mask].reset_index(drop=True)
    # Treat feature-identical rows as same group to prevent exact input overlap.
    groups = pd.util.hash_pandas_object(clean[FEATURES], index=False)
    splitter = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=42)
    train_idx, test_idx = next(splitter.split(clean, groups=groups))
    assert not set(groups.iloc[train_idx]).intersection(set(groups.iloc[test_idx]))
    X_train, X_test = clean.loc[train_idx, FEATURES], clean.loc[test_idx, FEATURES]
    y_train, y_test = clean.loc[train_idx, TARGET], clean.loc[test_idx, TARGET]
    baseline = float(y_train.median())
    baseline_metrics = {
        'MAE_Rp': round(float(mean_absolute_error(y_test, np.full(len(y_test),baseline))),2),
        'RMSE_Rp': round(float(np.sqrt(mean_squared_error(y_test, np.full(len(y_test),baseline)))),2),
        'R2': round(float(r2_score(y_test, np.full(len(y_test),baseline))),6),
    }
    models = {
        'Linear Regression': Pipeline([('prep', preprocess()),('model', LinearRegression())]),
        'Random Forest': Pipeline([('prep', preprocess()),('model', RandomForestRegressor(
            n_estimators=150, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1
        ))]),
    }
    metrics = {}
    for name,pipe in models.items():
        pipe.fit(X_train, y_train)
        metrics[name] = score_model(pipe, X_test, y_test)
        metrics[name]['train'] = score_model(pipe, X_train, y_train)
        print(name, metrics[name],flush=True)
    selected = 'Random Forest' if metrics['Random Forest']['MAE_Rp'] < metrics['Linear Regression']['MAE_Rp'] else 'Linear Regression'
    final = models[selected]  # trained on train only, test remains untouched
    ROOT.joinpath('models').mkdir(exist_ok=True)
    joblib.dump(final, ROOT/'models/house_price_pipeline.joblib', compress=3)
    # Test persisted pipeline and one-hot handling on a few real observed locations
    loaded = joblib.load(ROOT/'models/house_price_pipeline.joblib')
    first = X_test.iloc[[0]].copy()
    sample_preds = {}
    for location in ['Cinere','Beji','Sawangan']:
        sample = first.copy(); sample['Lokasi'] = location
        price = float(loaded.predict(sample)[0]); assert np.isfinite(price)
        sample_preds[location]=round(price)
    test_predictions = final.predict(X_test)
    bounds = pd.DataFrame({'actual':y_test.values, 'predicted':test_predictions})
    bounds['price_band'] = pd.cut(bounds['actual'],bins=[0,1e9,3e9,np.inf],labels=['<1 miliar','1–3 miliar','>3 miliar'])
    per_band = {str(k):{'n':len(g),'MAE_Rp':round(float(mean_absolute_error(g.actual,g.predicted)),2)} for k,g in bounds.groupby('price_band',observed=True)}
    result = {
        'scope':'Fixed analytical bounds: Harga Rp100 juta–10 miliar; tanah & bangunan 20–1500 m²; kamar tidur & mandi 1–15; garasi 0–10; not market-wide.',
        'dataset_rows':int(len(df)), 'in_scope_rows':int(len(clean)), 'excluded_rows':int((~mask).sum()),
        'train_rows':len(train_idx),'test_rows':len(test_idx), 'locations':int(clean['Lokasi'].nunique()),
        'split':'GroupShuffleSplit(test_size=0.20,random_state=42); identical feature vector kept within one split',
        'duplicate_feature_vectors_all':int(df.duplicated(subset=FEATURES).sum()),
        'duplicate_feature_vectors_scope':int(clean.duplicated(subset=FEATURES).sum()),
        'baseline_median_Rp':round(baseline),'baseline':baseline_metrics, 'models':metrics,
        'selected':selected,'test_MAE_by_actual_price_band':per_band,'sample_location_predictions_Rp':sample_preds,
        'versions':{'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,'scikit-learn':sklearn.__version__},
        'note':'One held-out group split; no cross-validation. Not a fair market valuation. Dataset provenance has not been supplied.'
    }
    (ROOT/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SUMMARY',json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
