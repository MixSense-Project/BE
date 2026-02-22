from __future__ import annotations
import os, time, json
import numpy as np
import librosa

from .ml_features import (
    parse_bpm_from_name, estimate_bpm, normalize_bpm,
    estimate_key, key_distance,
    rms_stats, onset_stats, spectral_stats,
)
from .bus_builder import build_or_load_bus
from .ml_models import load_strategy_model, load_transition_model, predict_strategy, score_transitions

def _bool(v) -> int:
    return 1 if v else 0

def analyze_track(track_id: str, bus: dict, sr: int) -> dict:
    y = bus.get("mix")
    bpm = parse_bpm_from_name(track_id)
    if bpm <= 0:
        bpm = estimate_bpm(y, sr)
    bpm = normalize_bpm(bpm)

    key_str, key_conf, key_idx = estimate_key(y, sr)

    rs = rms_stats(y, sr)
    os_ = onset_stats(y, sr)
    sp = spectral_stats(y, sr)

    meta = {
        "track_id": track_id,
        "bpm": bpm,
        "key": key_str,
        "key_conf": float(key_conf),
        "key_idx": int(key_idx),
        "rms_mean": rs["rms_mean"],
        "rms_std": rs["rms_std"],
        "rms_slope": rs["rms_slope"],
        "onset_mean": os_["onset_mean"],
        "onset_std": os_["onset_std"],
        "centroid": sp["centroid"],
        "rolloff": sp["rolloff"],
        "has_drums": _bool(bus.get("drums") is not None),
        "has_bass": _bool(bus.get("bass") is not None),
        "has_harmony": _bool(bus.get("harmony") is not None),
    }
    return meta

def build_pair_features(a: dict, b: dict) -> dict:
    bpm_diff = abs(round(a["bpm"], 1) - round(b["bpm"], 1)) if a["bpm"] > 0 and b["bpm"] > 0 else 999.0
    key_conf_min = min(a["key_conf"], b["key_conf"])
    kd = key_distance(a["key_idx"], b["key_idx"])

    # energy similarity: rms_mean 가까울수록 1
    da = a["rms_mean"]; db = b["rms_mean"]
    energy_sim = float(1.0 / (1.0 + abs(da - db) * 15.0))

    feat = {
        "bpm_a": a["bpm"], "bpm_b": b["bpm"], "bpm_diff": bpm_diff,
        "key_conf_min": key_conf_min, "key_diff": kd,
        "rms_mean_a": a["rms_mean"], "rms_mean_b": b["rms_mean"],
        "energy_sim": energy_sim,
        "onset_mean_a": a["onset_mean"], "onset_mean_b": b["onset_mean"],
        "centroid_a": a["centroid"], "centroid_b": b["centroid"],
        "has_drums_a": a["has_drums"], "has_drums_b": b["has_drums"],
        "has_harmony_a": a["has_harmony"], "has_harmony_b": b["has_harmony"],
        "has_bass_a": a["has_bass"], "has_bass_b": b["has_bass"],
    }
    return feat

def make_explanation(a: dict, b: dict, pair_feat: dict, strategy: str, proba: dict) -> dict:
    bpm_diff = pair_feat["bpm_diff"]
    kd = pair_feat["key_diff"]
    kconf = pair_feat["key_conf_min"]

    reasons = []
    reasons.append(f"BPM: {a['bpm']:.1f} vs {b['bpm']:.1f} (diff {bpm_diff:.1f})")
    if a["key"] != "UNKNOWN" and b["key"] != "UNKNOWN":
        reasons.append(f"Key: {a['key']} vs {b['key']} (distance {kd})")
    else:
        reasons.append(f"Key confidence low (min_conf={kconf:.2f}) → harmonic mixing restricted")

    if bpm_diff <= 4:
        reasons.append("Tempo compatible → allow phrase-based blending")
    else:
        reasons.append("Tempo mismatch → avoid heavy stretching; prefer swap/takeover")

    if kconf >= 0.55 and kd <= 2:
        reasons.append("Harmonic compatibility OK → harmony blend allowed")
    else:
        reasons.append("Harmonic compatibility risky → bass/harmony are limited to avoid dissonance")

    strat_desc = {
        "DRUM_SWAP": "Keep groove stable, swap drums near the end; harmony/bass minimized.",
        "HARMONY_BLEND": "Introduce B harmony first, then gradual takeover; preserve groove continuity.",
        "FULL_TAKEOVER": "Short blend windows + clear takeover in the last section.",
        "NO_MIX": "Detected high risk (tempo/key) → recommend not mixing or only FX/percussion.",
    }.get(strategy, "")

    return {
        "strategy": strategy,
        "strategy_desc": strat_desc,
        "strategy_proba": proba,
        "reasons": reasons,
    }

def build_transition_candidates(y: np.ndarray, sr: int, duration_sec: float, bpm: float) -> list[dict]:
    """
    bar 후보마다 local 특징을 만들고, transition model로 스코어링
    """
    if y is None or y.size == 0 or bpm <= 0:
        return []
    # bar grid
    spb = 60.0 / bpm
    spbar = spb * 4.0
    # candidates from 24s to duration-20s
    t_list = np.arange(24.0, max(24.0, duration_sec - 20.0), spbar).tolist()

    # local windows (±2s)로 slope/온셋 특징
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr)

    # spectral centroid for local
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    def _idx(t):
        return int((t * sr) / hop)

    out = []
    for t in t_list:
        i = _idx(t)
        w = 10  # about ~10 frames window
        a = max(0, i - w); b = min(len(rms), i + w)
        if b - a < 6:
            continue
        seg = rms[a:b]
        x = np.linspace(0, 1, seg.size)
        slope = float(np.polyfit(x, seg, 1)[0])
        o_m = float(np.mean(onset[a:b])) if b <= len(onset) else float(np.mean(onset))
        c_m = float(np.mean(cent[a:b])) if b <= len(cent) else float(np.mean(cent))
        out.append({"t_sec": float(t), "rms_slope_local": slope, "onset_mean_local": o_m, "centroid_local": c_m})
    return out

def decide_plan(
    zip_a: str,
    zip_b: str,
    data_dir: str,
    sr: int,
    duration_sec: float,
    target_k: int,
    model_dir: str = "models",
) -> dict:
    # load buses
    id_a, bus_a, meta_a = build_or_load_bus(zip_a, data_dir=data_dir, sr=sr)
    id_b, bus_b, meta_b = build_or_load_bus(zip_b, data_dir=data_dir, sr=sr)

    # track analyses
    A = analyze_track(id_a, bus_a, sr)
    B = analyze_track(id_b, bus_b, sr)
    pair_feat = build_pair_features(A, B)

    # models
    strat_pack = load_strategy_model(os.path.join(model_dir, "strategy_clf.joblib"))
    trans_pack = load_transition_model(os.path.join(model_dir, "transition_reg.joblib"))

    dec = predict_strategy(strat_pack, pair_feat)
    expl = make_explanation(A, B, pair_feat, dec.strategy, dec.proba)

    # transition ranking (use A mix for grid)
    y_ref = bus_a.get("mix")
    cands = build_transition_candidates(y_ref, sr, duration_sec, A["bpm"] if A["bpm"] > 0 else 120.0)
    scored = score_transitions(trans_pack, cands) if cands else []

    # pick top-k with spacing constraint
    picks = []
    min_gap = 18.0
    for s, c in scored:
        t = c["t_sec"]
        if all(abs(t - p["t_sec"]) >= min_gap for p in picks):
            picks.append({"t_sec": t, "score": float(s)})
        if len(picks) >= target_k:
            break

    plan = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {"A": zip_a, "B": zip_b},
        "track_A": A,
        "track_B": B,
        "pair_features": pair_feat,
        "explanation": expl,          # ✅ 프론트에 바로 보여줄 텍스트/근거
        "transitions": picks,         # ✅ 전이 시점 + score
    }
    return plan
