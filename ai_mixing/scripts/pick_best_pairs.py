from __future__ import annotations

import os
import glob
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from mixsense_mixing.bus_builder import build_or_load_bus
from mixsense_mixing.ml_decider import analyze_track, build_pair_features
from mixsense_mixing.ml_models import load_strategy_model, predict_strategy


@dataclass
class PairRow:
    a_zip: str
    b_zip: str
    a_id: str
    b_id: str
    bpm_a: float
    bpm_b: float
    bpm_diff: float
    key_a: str
    key_b: str
    key_conf_min: float
    key_diff: int
    energy_sim: float
    strategy: str
    proba: Dict[str, float]
    score: float
    reasons: List[str]


def _short(p: str) -> str:
    return os.path.basename(p)


def _pair_score(row: PairRow) -> float:
    """
    랭킹 점수:
    - harmony_blend 확률 최우선
    - bpm_diff 작을수록 좋음
    - energy_sim 높을수록 좋음
    - key_conf_min 낮으면 패널티
    """
    pb = float(row.proba.get("HARMONY_BLEND", 0.0))
    pfull = float(row.proba.get("FULL_TAKEOVER", 0.0))
    pdr = float(row.proba.get("DRUM_SWAP", 0.0))

    # bpm_diff는 0~5 정도가 좋은 범위라고 보고 역가중
    bpm_term = max(0.0, 1.0 - (row.bpm_diff / 5.0))

    # key confidence 패널티
    conf_term = min(1.0, max(0.0, (row.key_conf_min - 0.45) / 0.35))  # 0.45 이하 거의 0, 0.8 근처 1

    # 최종 점수 (harmony_blend 우선)
    return (
        2.2 * pb
        + 0.7 * pfull
        + 0.2 * pdr
        + 0.6 * row.energy_sim
        + 0.5 * bpm_term
        + 0.4 * conf_term
        - 0.2 * (row.key_diff / 2.0)  # key_diff가 0일수록 좋게
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips_dir", default="data/inputs/zips", help="directory containing *.zip stems")
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--model_dir", default="models")
    ap.add_argument("--max_pairs", type=int, default=20, help="print top N pairs")
    ap.add_argument("--min_key_conf", type=float, default=0.55)
    ap.add_argument("--max_key_diff", type=int, default=2)
    ap.add_argument("--max_bpm_diff", type=float, default=4.5)
    ap.add_argument("--allow_low_conf", action="store_true", help="if set, do not filter by key_conf_min")
    ap.add_argument("--save_json", default="", help="optional output json path")
    args = ap.parse_args()

    zips = sorted(glob.glob(os.path.join(args.zips_dir, "*.zip")))
    if len(zips) < 2:
        raise SystemExit(f"Need >=2 zip files in {args.zips_dir}")

    # Load strategy model (이미 train_ml_models.py로 생성돼 있어야 함)
    strat_path = os.path.join(args.model_dir, "strategy_clf.joblib")
    if not os.path.exists(strat_path):
        raise SystemExit(f"Missing {strat_path}. Run: python scripts/train_ml_models.py")

    strat_pack = load_strategy_model(strat_path)

    # 1) Track analyze (한 번씩만)
    tracks = []
    for z in zips:
        tid, bus, _ = build_or_load_bus(z, data_dir="data", sr=args.sr)
        meta = analyze_track(tid, bus, args.sr)
        meta["zip"] = z
        meta["_id"] = tid
        tracks.append(meta)

    # 2) Pair evaluate
    rows: List[PairRow] = []
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            A = tracks[i]
            B = tracks[j]
            feat = build_pair_features(A, B)
            dec = predict_strategy(strat_pack, feat)

            reasons = [
                f"BPM {A['bpm']:.1f} vs {B['bpm']:.1f} (diff {feat['bpm_diff']:.1f})",
                f"Key {A['key']}({A['key_conf']:.2f}) vs {B['key']}({B['key_conf']:.2f}) (diff {feat['key_diff']})",
                f"energy_sim={feat['energy_sim']:.3f}",
            ]

            row = PairRow(
                a_zip=A["zip"],
                b_zip=B["zip"],
                a_id=A["_id"],
                b_id=B["_id"],
                bpm_a=float(A["bpm"]),
                bpm_b=float(B["bpm"]),
                bpm_diff=float(feat["bpm_diff"]),
                key_a=str(A["key"]),
                key_b=str(B["key"]),
                key_conf_min=float(feat["key_conf_min"]),
                key_diff=int(feat["key_diff"]),
                energy_sim=float(feat["energy_sim"]),
                strategy=str(dec.strategy),
                proba=dict(dec.proba),
                score=0.0,
                reasons=reasons,
            )
            row.score = _pair_score(row)
            rows.append(row)

    # 3) Filter: "하모니 블렌드가 가능한 페어"만 먼저 뽑기
    def ok(r: PairRow) -> bool:
        if r.bpm_diff > args.max_bpm_diff:
            return False
        if r.key_diff > args.max_key_diff:
            return False
        if (not args.allow_low_conf) and (r.key_conf_min < args.min_key_conf):
            return False
        return True

    good = [r for r in rows if ok(r)]
    good.sort(key=lambda r: r.score, reverse=True)

    # 4) Print
    print(f"\nFound zips: {len(zips)}")
    print(f"Candidate pairs total: {len(rows)}")
    print(f"Filtered (key_diff<={args.max_key_diff}, bpm_diff<={args.max_bpm_diff}, key_conf_min>={args.min_key_conf if not args.allow_low_conf else 'ANY'}): {len(good)}\n")

    top = good[: args.max_pairs] if good else []
    if not top:
        print("No good pairs found under current constraints.")
        print("Try: --allow_low_conf OR relax --max_key_diff to 3")
        return

    for idx, r in enumerate(top, 1):
        pb = r.proba.get("HARMONY_BLEND", 0.0)
        print(
            f"[{idx:02d}] {_short(r.a_zip)}  <->  {_short(r.b_zip)}\n"
            f"     bpm_diff={r.bpm_diff:.1f} | key={r.key_a} vs {r.key_b} (diff {r.key_diff}, conf_min {r.key_conf_min:.2f}) | energy_sim={r.energy_sim:.3f}\n"
            f"     strategy={r.strategy} | P(HARMONY_BLEND)={pb:.3f} | score={r.score:.3f}\n"
        )

    # 5) Save json (optional)
    if args.save_json:
        payload = []
        for r in top:
            payload.append({
                "a_zip": r.a_zip,
                "b_zip": r.b_zip,
                "bpm_diff": r.bpm_diff,
                "key_a": r.key_a,
                "key_b": r.key_b,
                "key_conf_min": r.key_conf_min,
                "key_diff": r.key_diff,
                "energy_sim": r.energy_sim,
                "strategy": r.strategy,
                "strategy_proba": r.proba,
                "score": r.score,
                "reasons": r.reasons,
            })
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("Saved:", args.save_json)


if __name__ == "__main__":
    main()
