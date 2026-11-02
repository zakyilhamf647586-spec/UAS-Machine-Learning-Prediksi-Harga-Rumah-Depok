"""Run from repository root: python -m pytest tests/test_model.py -q"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score
from train_model import FEATURES, TARGET, in_scope

ROOT=Path(__file__).resolve().parents[1]
model=joblib.load(ROOT/'models/house_price_pipeline.joblib') # trusted project artifact only

def test_all_locations_can_be_encoded_and_predicted():
    d=pd.read_csv(ROOT/'dataset/Cleaned_Final_Rumah.csv')
    d=d.loc[in_scope(d)].reset_index(drop=True)
    sample=d.loc[[0],FEATURES].copy()
    for name in sorted(d.Lokasi.unique()):
        sample['Lokasi']=name
        pred=model.predict(sample)
        assert pred.shape==(1,)
        assert np.isfinite(pred).all()

def test_no_identical_feature_groups_cross_split():
    d=pd.read_csv(ROOT/'dataset/Cleaned_Final_Rumah.csv')
    d=d.loc[in_scope(d)].reset_index(drop=True)
    g=pd.util.hash_pandas_object(d[FEATURES],index=False)
    train, test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(d,groups=g))
    assert not set(g.iloc[train]) & set(g.iloc[test])

def test_persisted_model_reproduces_reported_holdout_metrics():
    import json
    metrics=json.loads((ROOT/'metrics.json').read_text(encoding='utf-8'))
    d=pd.read_csv(ROOT/'dataset/Cleaned_Final_Rumah.csv')
    d=d.loc[in_scope(d)].reset_index(drop=True)
    g=pd.util.hash_pandas_object(d[FEATURES],index=False)
    _,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(d,groups=g))
    prediction=model.predict(d.loc[test,FEATURES])
    actual=d.loc[test,TARGET]
    assert abs(mean_absolute_error(actual,prediction)-metrics['models'][metrics['selected']]['MAE_Rp'])<1
    assert abs(r2_score(actual,prediction)-metrics['models'][metrics['selected']]['R2'])<1e-5
