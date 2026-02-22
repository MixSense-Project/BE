from __future__ import annotations
import os
import numpy as np
from dataclasses import dataclass
from joblib import dump, load

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, SGDRegressor

from .ml_teacher import teacher_strategy_label, teacher_transition_score, STRATS

@dataclass
class StrategyDecision:
    strategy: str
    proba: dict

def _vec_from_feat(feat: dict, keys: list[str]) -> np.ndarray:
    return np.array([float(feat.get(k, 0.0)) for k in keys], dtype=np.float32)

# feature keys (고정)
STRATEGY_KEYS = [
    "bpm_a","bpm_b","bpm_diff",
    "key_conf_min","key_diff",
    "rms_mean_a","rms_mean_b","energy_sim",
    "onset_mean_a","onset_mean_b",
    "centroid_a","centroid_b",
    "has_drums_a","has_drums_b","has_harmony_a","has_harmony_b","has_bass_a","has_bass_b",
]

TRANS_KEYS = ["rms_slope_local","onset_mean_local","centroid_local"]

def train_strategy_model(feats: list[dict], out_path: str) -> str:
    X = np.stack([_vec_from_feat(f, STRATEGY_KEYS) for f in feats])
    y_str = [teacher_strategy_label(f) for f in feats]
    y = np.array([STRATS.index(s) for s in y_str], dtype=int)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("ovr", OneVsRestClassifier(
            LogisticRegression(
                max_iter=2000,
                solver="lbfgs",   # scaler가 있으면 잘 수렴
            )
        ))
    ])
    clf.fit(X, y)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dump({"model": clf, "keys": STRATEGY_KEYS}, out_path)
    return out_path

def train_transition_model(cands: list[dict], out_path: str) -> str:
    X = np.stack([_vec_from_feat(c, TRANS_KEYS) for c in cands])
    y = np.array([teacher_transition_score(c) for c in cands], dtype=np.float32)

    # Ridge 대신 SGD 회귀: ill-conditioned 경고 제거 + 스케일링 내장
    reg = Pipeline([
        ("scaler", StandardScaler()),
        ("sgd", SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=1e-3,
            max_iter=2000,
            tol=1e-4,
            random_state=42,
        ))
    ])
    reg.fit(X, y)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dump({"model": reg, "keys": TRANS_KEYS}, out_path)
    return out_path

def load_strategy_model(path: str):
    return load(path)

def load_transition_model(path: str):
    return load(path)

def predict_strategy(model_pack, feat: dict) -> StrategyDecision:
    clf = model_pack["model"]
    keys = model_pack["keys"]
    x = _vec_from_feat(feat, keys).reshape(1, -1)

    # sklearn은 학습된 클래스만 classes_로 들고 있음
    # Pipeline(OneVsRestClassifier)인 경우: clf.named_steps["ovr"].classes_
    classes = None
    try:
        # Pipeline
        if hasattr(clf, "named_steps") and "ovr" in clf.named_steps:
            classes = clf.named_steps["ovr"].classes_
        elif hasattr(clf, "classes_"):
            classes = clf.classes_
    except Exception:
        classes = None

    proba = clf.predict_proba(x)[0]

    # 기본값 0으로 깔고, 학습된 클래스만 채운다
    proba_map = {s: 0.0 for s in STRATS}

    if classes is None:
        # 최후의 fallback: 길이 맞는 만큼만 채움
        for i in range(min(len(proba), len(STRATS))):
            proba_map[STRATS[i]] = float(proba[i])
    else:
        # classes_는 [0,1,2] 같은 인덱스 라벨일 가능성이 큼
        for i, c in enumerate(classes):
            idx = int(c)
            if 0 <= idx < len(STRATS):
                proba_map[STRATS[idx]] = float(proba[i])

    best = max(proba_map.items(), key=lambda kv: kv[1])[0]
    return StrategyDecision(strategy=best, proba=proba_map)


def score_transitions(model_pack, cands: list[dict]) -> list[tuple[float, dict]]:
    reg = model_pack["model"]
    keys = model_pack["keys"]
    X = np.stack([_vec_from_feat(c, keys) for c in cands])
    s = reg.predict(X)
    out = [(float(s[i]), cands[i]) for i in range(len(cands))]
    out.sort(key=lambda x: x[0], reverse=True)
    return out
