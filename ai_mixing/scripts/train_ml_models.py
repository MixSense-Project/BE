from __future__ import annotations
import os, glob, json
import numpy as np

# src import
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from mixsense_mixing.bus_builder import build_or_load_bus
from mixsense_mixing.ml_decider import analyze_track, build_pair_features, build_transition_candidates
from mixsense_mixing.ml_models import train_strategy_model, train_transition_model

def main():
    data_dir = "data"
    sr = 48000
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    zips = sorted(glob.glob(os.path.join(data_dir, "inputs", "zips", "*.zip")))
    if len(zips) < 2:
        raise SystemExit("data/inputs/zips/*.zip 에 최소 2개 zip 필요")

    # 1) 각 트랙 분석 저장
    tracks = []
    for z in zips:
        tid, bus, _ = build_or_load_bus(z, data_dir=data_dir, sr=sr)
        t = analyze_track(tid, bus, sr)
        t["zip"] = z
        tracks.append(t)

    # 2) pair feats 대량 생성 (조합)
    pair_feats = []
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            A = tracks[i]; B = tracks[j]
            f = build_pair_features(A, B)
            # stem availability도 반영하기 위해 A/B를 포함
            f.update({
                "has_drums_a": A["has_drums"], "has_drums_b": B["has_drums"],
                "has_harmony_a": A["has_harmony"], "has_harmony_b": B["has_harmony"],
                "has_bass_a": A["has_bass"], "has_bass_b": B["has_bass"],
            })
            pair_feats.append(f)

    # 3) transition 후보 샘플 생성 (각 트랙에서 bar 후보 특징)
    cands = []
    for t in tracks:
        # mix를 다시 로드하려면 bus 필요 → build_or_load_bus 다시 호출
        tid, bus, _ = build_or_load_bus(t["zip"], data_dir=data_dir, sr=sr)
        y = bus.get("mix")
        dur = min(150.0, len(y)/sr) if y is not None else 0.0
        bpm = t["bpm"] if t["bpm"] > 0 else 120.0
        cands.extend(build_transition_candidates(y, sr, dur, bpm))

    # 4) train
    strat_path = train_strategy_model(pair_feats, os.path.join(model_dir, "strategy_clf.joblib"))
    trans_path = train_transition_model(cands, os.path.join(model_dir, "transition_reg.joblib"))

    with open(os.path.join(model_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "num_zips": len(zips),
            "num_pairs": len(pair_feats),
            "num_transition_samples": len(cands),
            "strategy_model": strat_path,
            "transition_model": trans_path,
        }, f, ensure_ascii=False, indent=2)

    print("OK:", strat_path, trans_path)

if __name__ == "__main__":
    main()
