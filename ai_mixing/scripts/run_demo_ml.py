from __future__ import annotations

import os
import json
import time
import argparse
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from mixsense_mixing.ml_decider import decide_plan
from mixsense_mixing.bus_builder import build_or_load_bus
from mixsense_mixing.io import write_wav_48k24


def _ensure_len(y: np.ndarray, L: int) -> np.ndarray:
    if y is None:
        return np.zeros((L,), dtype=np.float32)
    if len(y) >= L:
        return y[:L].astype(np.float32)
    out = np.zeros((L,), dtype=np.float32)
    out[: len(y)] = y
    return out.astype(np.float32)


def _soft_clip(y: np.ndarray, lim: float = 0.98) -> np.ndarray:
    y = y.astype(np.float32)
    return np.tanh(y / max(lim, 1e-6)) * lim


def _s_curve(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones((n,), dtype=np.float32)
    x = np.linspace(0, np.pi, n, dtype=np.float32)
    return (1.0 - np.cos(x)) * 0.5


def apply_fade_out(y: np.ndarray, sr: int, fade_out_sec: float) -> np.ndarray:
    if fade_out_sec <= 0:
        return y
    L = len(y)
    n = int(fade_out_sec * sr)
    if n <= 1 or n >= L:
        return y
    g = 1.0 - _s_curve(n)  # 1 -> 0
    y2 = y.copy().astype(np.float32)
    y2[L - n : L] *= g.astype(np.float32)
    return y2


def maybe_override_strategy(plan: dict, hb_threshold: float = 0.30) -> dict:
    exp = plan.get("explanation", {})
    pair = plan.get("pair_features", {})
    proba = exp.get("strategy_proba", {})

    key_diff = int(pair.get("key_diff", 99))
    p_hb = float(proba.get("HARMONY_BLEND", 0.0))
    strat = exp.get("strategy", "FULL_TAKEOVER")

    if key_diff <= 1 and p_hb >= hb_threshold and strat != "HARMONY_BLEND":
        exp["strategy_prev"] = strat
        exp["strategy"] = "HARMONY_BLEND"
        exp.setdefault("reasons", []).append(
            f"Override: key_diff={key_diff} and P(HARMONY_BLEND)={p_hb:.3f} >= {hb_threshold:.2f} → choose HARMONY_BLEND for audible mixing."
        )
        plan["explanation"] = exp
    return plan


def highpass_1pole(x: np.ndarray, sr: int, f: float = 180.0) -> np.ndarray:
    """저역(둥둥)만 줄이고 중고역은 살리는 간단 하이패스."""
    if x is None or x.size == 0:
        return np.zeros((0,), dtype=np.float32)
    y = x.astype(np.float32)

    rc = 1.0 / (2 * np.pi * f)
    dt = 1.0 / sr
    a = rc / (rc + dt)

    hp = np.zeros_like(y)
    hp[0] = y[0]
    for i in range(1, len(y)):
        hp[i] = a * (hp[i - 1] + y[i] - y[i - 1])
    return hp.astype(np.float32)


def enforce_even_transition_times(plan: dict, duration: float, k_target: int, pad_head: float, pad_tail: float) -> dict:
    """
    ✅ 전이 시간을 무조건 균등 배치해서
    매 전이가 'hold 10초'로 들리게 강제한다.
    (decide_plan이 준 전이 score는 유지하고, t_sec만 바꿈)
    """
    trans = plan.get("transitions", [])
    if not trans:
        return plan

    trans_sorted = sorted(trans, key=lambda d: float(d.get("score", 0.0)), reverse=True)
    k_eff = min(k_target, len(trans_sorted))

    span = max(1.0, duration - pad_head - pad_tail)
    step = span / (k_eff + 1)
    slots = [pad_head + step * (i + 1) for i in range(k_eff)]

    new_trans = []
    for i in range(k_eff):
        d = dict(trans_sorted[i])
        d["t_sec"] = float(slots[i])
        new_trans.append(d)

    new_trans = sorted(new_trans, key=lambda d: float(d["t_sec"]))

    plan["transitions_original"] = trans
    plan["transitions"] = new_trans
    plan.setdefault("render_notes", [])
    plan["render_notes"].append(
        f"Transitions re-timed evenly (k={k_eff}, pad_head={pad_head}, pad_tail={pad_tail}) to guarantee audible 10s blend windows."
    )
    return plan


def build_weight_curve(L, sr, transitions, peak, ramp_in_sec, hold_sec, ramp_out_sec):
    w = np.zeros((L,), dtype=np.float32)
    transitions = sorted(transitions, key=lambda d: float(d.get("t_sec", 0.0)))

    for p in transitions:
        t0 = float(p.get("t_sec", 0.0))
        s = int(max(0.0, t0 - ramp_in_sec) * sr)
        m = int(max(0.0, t0) * sr)
        h = int(min(L / sr, t0 + hold_sec) * sr)
        e = int(min(L / sr, t0 + hold_sec + ramp_out_sec) * sr)

        if m > s:
            w[s:m] = np.maximum(w[s:m], _s_curve(m - s) * peak)
        if h > m:
            w[m:h] = np.maximum(w[m:h], peak)
        if e > h:
            w[h:e] = np.maximum(w[h:e], (1.0 - _s_curve(e - h)) * peak)

    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", default="outputs")

    ap.add_argument("--dur", type=float, default=150.0)
    ap.add_argument("--k", type=int, default=6)

    ap.add_argument("--ramp_in", type=float, default=2.0)
    ap.add_argument("--hold", type=float, default=10.0)
    ap.add_argument("--ramp_out", type=float, default=2.0)

    # ✅ 전이 슬롯을 곡 전체에 고르게 퍼뜨리기 위한 패딩
    ap.add_argument("--pad_head", type=float, default=25.0)
    ap.add_argument("--pad_tail", type=float, default=25.0)

    # 체감 강도
    ap.add_argument("--duck", type=float, default=0.75)          # A를 더 눌러서 섞임 티 강화
    ap.add_argument("--peak", type=float, default=0.80)          # B 가중 상한
    ap.add_argument("--hb_threshold", type=float, default=0.30)

    # 끝 takeover / fadeout
    ap.add_argument("--end_takeover", type=float, default=14.0)
    ap.add_argument("--takeover_mix", type=float, default=0.65)
    ap.add_argument("--fade_out", type=float, default=10.0)

    # Sleepless 둥둥 억제(블렌드용 하이패스)
    ap.add_argument("--hp", type=float, default=180.0)

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    plan = decide_plan(
        args.a, args.b,
        data_dir="data",
        sr=48000,
        duration_sec=args.dur,
        target_k=args.k,
        model_dir="models",
    )
    plan = maybe_override_strategy(plan, hb_threshold=args.hb_threshold)

    # ✅ 여기서 무조건 균등 배치 강제(실수 여지 제거)
    plan = enforce_even_transition_times(plan, duration=args.dur, k_target=args.k, pad_head=args.pad_head, pad_tail=args.pad_tail)

    idA, busA, _ = build_or_load_bus(args.a, data_dir="data", sr=48000)
    idB, busB, _ = build_or_load_bus(args.b, data_dir="data", sr=48000)

    L = int(args.dur * 48000)
    A = _ensure_len(busA.get("mix"), L)
    Bmix = _ensure_len(busB.get("mix"), L)

    # ✅ 블렌드용 B: Bmix를 하이패스해서 '둥둥' 줄이고 섞임 티는 살림
    B_blend = _ensure_len(highpass_1pole(Bmix, 48000, f=float(args.hp)), L)

    # weight curve
    w = build_weight_curve(L, 48000, plan.get("transitions", []), float(args.peak), args.ramp_in, args.hold, args.ramp_out)

    # ducking + blend
    A_gain = (1.0 - w * float(args.duck)).astype(np.float32)
    out = (A * A_gain + B_blend * w).astype(np.float32)

    # end takeover (너무 둥둥이면 takeover_mix 더 낮춰)
    end_len = int(args.end_takeover * 48000)
    if 0 < end_len < L:
        s = L - end_len
        g = _s_curve(end_len)
        Bt = (Bmix * float(args.takeover_mix)).astype(np.float32)
        out[s:L] = (out[s:L] * (1.0 - g) + Bt[s:L] * g).astype(np.float32)

    out = _soft_clip(out, lim=0.98)
    out = apply_fade_out(out, 48000, args.fade_out)

    # 기록
    plan.setdefault("render_cfg", {})
    plan["render_cfg"].update({
        "duration_sec": float(args.dur),
        "k_transitions": int(args.k),
        "ramp_in_sec": float(args.ramp_in),
        "hold_sec": float(args.hold),
        "ramp_out_sec": float(args.ramp_out),
        "pad_head": float(args.pad_head),
        "pad_tail": float(args.pad_tail),
        "duck_strength": float(args.duck),
        "peak": float(args.peak),
        "hb_threshold": float(args.hb_threshold),
        "hp_hz": float(args.hp),
        "end_takeover_sec": float(args.end_takeover),
        "takeover_mix": float(args.takeover_mix),
        "fade_out_sec": float(args.fade_out),
        "forced_even_transitions": True,
    })

    stamp = time.strftime("%Y%m%d_%H%M%S")
    wav_path = os.path.join(args.out, f"mixML_{idA}__{idB}__{int(args.dur)}s__k{args.k}__{stamp}.wav")
    json_path = wav_path.replace(".wav", "_plan.json")

    write_wav_48k24(wav_path, out, sr=48000)
    plan["outputs"] = {"wav": wav_path, "plan_json": json_path}

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    ts = [round(float(t["t_sec"]), 2) for t in plan.get("transitions", [])]
    print("OK")
    print("WAV:", wav_path)
    print("PLAN:", json_path)
    print("FORCED TRANSITION SLOTS:", ts)


if __name__ == "__main__":
    main()
