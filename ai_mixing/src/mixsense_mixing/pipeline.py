from __future__ import annotations

import os, json, time, re
import numpy as np
import librosa

from .types import MixResult
from .bus_builder import build_or_load_bus
from .probe import estimate_bpm_and_beats, build_bar_candidates, compute_track_features, choose_backbone
from .scheduler import schedule_transitions
from .render import render_styleB_from_bus
from .io import write_wav_48k24


def _parse_bpm_from_name(name: str) -> float:
    """
    파일명/track_id에서 bpm 힌트를 추출.
    예: 'Sauce_142bpm' / 'Newleaf 140bpm' / 'song-128BPM' 등
    """
    if not name:
        return 0.0
    m = re.search(r"(\d{2,3})\s*bpm", name.lower())
    if not m:
        m = re.search(r"bpm\s*(\d{2,3})", name.lower())
    if m:
        v = float(m.group(1))
        if 40 <= v <= 250:
            return v
    return 0.0


def _normalize_bpm(bpm: float, target_range=(80.0, 180.0)) -> float:
    """
    beat_track이 half/double/quarter-time으로 튀는 걸 보정.
    - bpm이 너무 낮으면 2배/4배 후보로 올리고
    - 너무 높으면 1/2, 1/4 후보로 내린다.
    """
    lo, hi = target_range
    if bpm <= 0:
        return 0.0

    candidates = [bpm, bpm * 2, bpm * 4, bpm / 2, bpm / 4]
    # 범위 안 후보 우선
    in_range = [c for c in candidates if lo <= c <= hi]
    if in_range:
        # 원래 값과의 변화가 가장 작은 후보
        in_range.sort(key=lambda c: abs(c - bpm))
        return float(in_range[0])

    # 그래도 없으면 범위 중심에 가까운 걸 선택
    mid = (lo + hi) / 2
    candidates.sort(key=lambda c: abs(c - mid))
    return float(candidates[0])


def _choose_master_bpm(bpm1: float, bpm2: float, fallback: float = 120.0) -> float:
    """
    master_bpm 선택: 둘 다 정상 범위면 둘 중 하나를 고르고,
    하나만 있으면 그걸, 둘 다 없으면 fallback.
    """
    if bpm1 <= 0 and bpm2 <= 0:
        return float(fallback)
    if bpm1 <= 0:
        return float(bpm2)
    if bpm2 <= 0:
        return float(bpm1)

    # 둘 다 있으면 중앙값 느낌으로 더 안정적인 걸 선택(극단 ratio 회피)
    r12 = bpm1 / bpm2
    r21 = bpm2 / bpm1
    return float(bpm1 if abs(1 - r21) <= abs(1 - r12) else bpm2)


def _clamp_ratio(r: float, lo: float = 0.90, hi: float = 1.10) -> float:
    if r <= 0:
        return 1.0
    return float(max(lo, min(hi, r)))


def run_mix(
    zip1: str,
    zip2: str,
    out_dir: str = "outputs",
    data_dir: str = "data",
    target_duration: float = 150.0,
    target_k: int = 5,
    min_gap_sec: float = 24.0,
    end_takeover_sec: float = 26.0,
    sr: int = 48000,
    inject_ratio: float = 0.9,
) -> MixResult:
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    # 1) zip -> bus cache -> bus load
    id1, bus1, meta1 = build_or_load_bus(zip1, data_dir=data_dir, sr=sr)
    id2, bus2, meta2 = build_or_load_bus(zip2, data_dir=data_dir, sr=sr)

    if "mix" not in bus1 or "mix" not in bus2:
        raise RuntimeError("mix_bus 생성 실패: 입력 zip에 유효한 오디오가 부족합니다.")

    # 2) bpm/beat 추정 (mix_bus 기반)
    bpm1_est, beats1 = estimate_bpm_and_beats(bus1["mix"], sr)
    bpm2_est, beats2 = estimate_bpm_and_beats(bus2["mix"], sr)

    # 3) 파일명 bpm 힌트 우선 적용 + 보정
    bpm1_hint = _parse_bpm_from_name(id1) or _parse_bpm_from_name(os.path.basename(zip1))
    bpm2_hint = _parse_bpm_from_name(id2) or _parse_bpm_from_name(os.path.basename(zip2))

    # 기본은 estimate, 단 힌트가 있으면 힌트로 override
    bpm1 = float(bpm1_hint if bpm1_hint > 0 else bpm1_est)
    bpm2 = float(bpm2_hint if bpm2_hint > 0 else bpm2_est)

    # half/double-time 보정 (힌트 사용 시에도 안전하게 한번)
    bpm1 = _normalize_bpm(bpm1)
    bpm2 = _normalize_bpm(bpm2)

    # master bpm 결정
    master_bpm = _choose_master_bpm(bpm1, bpm2, fallback=120.0)

    # ratio (master로 맞추기)
    r1_raw = master_bpm / bpm1 if bpm1 > 0 else 1.0
    r2_raw = master_bpm / bpm2 if bpm2 > 0 else 1.0

    # ✅ 핵심: 비정상 ratio는 time-stretch 금지(Clamp)
    r1_apply = _clamp_ratio(r1_raw)
    r2_apply = _clamp_ratio(r2_raw)

    dur1 = len(bus1["mix"]) / sr
    dur2 = len(bus2["mix"]) / sr

    # bar 후보 생성 (150초 내)
    bar1 = build_bar_candidates(beats1, master_bpm, min(target_duration, dur1))
    bar2 = build_bar_candidates(beats2, master_bpm, min(target_duration, dur2))
    bar_times = sorted(set([t for t in (bar1 + bar2) if 0 < t < target_duration]))

    # 4) A/B 자동 선택(features는 bus 기준)
    tf1 = compute_track_features(id1, sr, dur1, bus1, bpm1, r1_raw, target_duration)
    tf2 = compute_track_features(id2, sr, dur2, bus2, bpm2, r2_raw, target_duration)
    choice = choose_backbone(tf1, tf2)

    def pick(which: str):
        if which == "T1":
            return id1, bus1, r1_raw, r1_apply, meta1, tf1
        return id2, bus2, r2_raw, r2_apply, meta2, tf2

    bb_id, bb_bus, bb_ratio_raw, bb_ratio_apply, bb_meta, _ = pick(choice["backbone"])
    inj_id, inj_bus, inj_ratio_raw, inj_ratio_apply, inj_meta, _ = pick(choice["inject"])

    # 5) time-stretch (apply ratio만 적용)
    def stretch_bus(bus: dict, ratio_apply: float) -> dict:
        out = {}
        for k, y in bus.items():
            if y is None:
                continue
            if abs(1.0 - ratio_apply) < 1e-3:
                out[k] = y
            else:
                out[k] = librosa.effects.time_stretch(y, rate=float(ratio_apply)).astype(np.float32)
        out["_master_bpm"] = float(master_bpm)
        return out

    bb_bus_s = stretch_bus(bb_bus, bb_ratio_apply)
    inj_bus_s = stretch_bus(inj_bus, inj_ratio_apply)

    # 6) schedule (k 자동 감소는 "raw ratio" 기준으로 판단하게 유지)
    # -> raw ratio가 크면 schedule_transitions 내부에서 k를 줄인다.
    events = schedule_transitions(
        duration_sec=float(target_duration),
        target_k=int(target_k),
        bar_times=bar_times,
        stretch_ratio=float(inj_ratio_raw),  # ✅ raw를 넘겨서 위험하면 k 감소
        min_gap_sec=float(min_gap_sec),
        end_takeover_sec=float(end_takeover_sec),
    )

    y_out = render_styleB_from_bus(
        backbone_bus=bb_bus_s,
        inject_bus=inj_bus_s,
        sr=sr,
        duration_sec=float(target_duration),
        events=events,
        inject_ratio=float(inject_ratio),
        end_takeover_sec=float(end_takeover_sec),
    )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    used_k = len(events)
    out_wav = os.path.join(
        out_dir,
        f"mix_{bb_id}_asA__{inj_id}_asB__{int(target_duration)}s__k{used_k}__{stamp}.wav",
    )
    out_log = out_wav.replace(".wav", "_log.json")
    write_wav_48k24(out_wav, y_out, sr=sr)

    log = {
        "spec": {
            "target_duration": target_duration,
            "target_k": target_k,
            "used_k": used_k,
            "min_gap_sec": min_gap_sec,
            "end_takeover_sec": end_takeover_sec,
            "sr": sr,
            "output": "WAV PCM_24 48kHz",
        },
        "inputs": {"zip1": zip1, "zip2": zip2},
        "track_ids": {"zip1": id1, "zip2": id2},
        "bus_meta": {id1: meta1, id2: meta2},
        "choice": choice,
        "bpm": {
            id1: {"est": bpm1_est, "hint": bpm1_hint, "final": bpm1},
            id2: {"est": bpm2_est, "hint": bpm2_hint, "final": bpm2},
        },
        "master_bpm": master_bpm,
        "ratios": {
            id1: {"raw": r1_raw, "apply": r1_apply},
            id2: {"raw": r2_raw, "apply": r2_apply},
            "inject_used": {"raw": inj_ratio_raw, "apply": inj_ratio_apply},
            "backbone_used": {"raw": bb_ratio_raw, "apply": bb_ratio_apply},
        },
        "track_features": {id1: tf1.__dict__, id2: tf2.__dict__},
        "events": [e.__dict__ for e in events],
        "runtime_sec": time.time() - t0,
    }
    with open(out_log, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    return MixResult(
        out_wav_path=out_wav,
        log_json_path=out_log,
        used_k=used_k,
        events=events,
        backbone=bb_id,
        inject=inj_id,
    )
