"""Model smoke tests run without CSV; data-dependent regression tests run when data is local.

Execute: python -m pytest tests/test_model.py -q
The public CI should show three passes and three skips without the Kaggle CSV.
Full holdout validation runs locally when dataset/Cleaned_Final_Rumah.csv is available.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_model import FEATURES, TARGET, in_scope  # noqa: E402

MODEL_PATH = ROOT / "models" / "house_price_pipeline.joblib"
METRICS_PATH = ROOT / "metrics.json"
DATA_PATH = ROOT / "dataset" / "Cleaned_Final_Rumah.csv"
DATA_REQUIRED = pytest.mark.skipif(
    not DATA_PATH.exists(), reason="Dataset external: download Kaggle CSV for full local evaluation"
)

@pytest.fixture(scope="module")
def model():
    # Only load the model produced by your own trusted training run.
    return joblib.load(MODEL_PATH)

@pytest.fixture(scope="module")
def metrics():
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def _input(location: str, area: int = 100) -> pd.DataFrame:
    return pd.DataFrame([{
        "Kamar Tidur": 3, "Kamar Mandi": 2, "Garasi": 1,
        "Luas Tanah": 120, "Luas Bangunan": area, "Lokasi": location,
    }], columns=FEATURES)


def test_all_known_locations_can_be_predicted_without_csv(model, metrics):
    encoder = model.named_steps["prep"].named_transformers_["location"]
    locations = list(map(str, encoder.categories_[0]))
    assert len(locations) == metrics["locations"]
    for loc in locations:
        y = model.predict(_input(loc))
        assert y.shape == (1,)
        assert np.isfinite(y).all()
        assert (y > 0).all()


def test_single_vs_batch_prediction_without_csv(model):
    cases = pd.concat([_input("Beji", 100), _input("Margonda", 100), _input("Beji", 200)], ignore_index=True)
    batch = model.predict(cases)
    one_by_one = np.array([model.predict(cases.iloc[[i]])[0] for i in range(len(cases))])
    assert np.allclose(batch, one_by_one, rtol=0, atol=1e-6)


def test_reported_metadata_is_well_formed_without_csv(metrics):
    selected = metrics["selected"]
    assert selected in metrics["models"]
    assert metrics["train_rows"] > 0 and metrics["test_rows"] > 0
    assert metrics["in_scope_rows"] == metrics["train_rows"] + metrics["test_rows"]
    for info in metrics["models"].values():
        assert info["MAE_Rp"] > 0
        assert np.isfinite(info["R2"])
    # This checks metadata structure, NOT re-computation of holdout scores.


@DATA_REQUIRED
def test_split_does_not_share_identical_feature_vectors():
    d = pd.read_csv(DATA_PATH)
    d = d.loc[in_scope(d)].reset_index(drop=True)
    g = pd.util.hash_pandas_object(d[FEATURES], index=False)
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(d, groups=g))
    assert not set(g.iloc[train]) & set(g.iloc[test])


@DATA_REQUIRED
def test_persisted_model_reproduces_holdout_scores(model, metrics):
    d = pd.read_csv(DATA_PATH)
    d = d.loc[in_scope(d)].reset_index(drop=True)
    g = pd.util.hash_pandas_object(d[FEATURES], index=False)
    _, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(d, groups=g))
    p = model.predict(d.loc[test, FEATURES])
    y = d.loc[test, TARGET]
    expected = metrics["models"][metrics["selected"]]
    assert abs(mean_absolute_error(y, p) - expected["MAE_Rp"]) < 1
    assert abs(r2_score(y, p) - expected["R2"]) < 1e-5


@DATA_REQUIRED
def test_all_dataset_locations_can_be_predicted(model):
    d = pd.read_csv(DATA_PATH)
    d = d.loc[in_scope(d)]
    sample = d.iloc[[0]][FEATURES].copy()
    for loc in sorted(d["Lokasi"].unique()):
        sample["Lokasi"] = loc
        pred = model.predict(sample)
        assert pred.shape == (1,)
        assert np.isfinite(pred).all()
