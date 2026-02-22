from __future__ import annotations
from typing import Dict, Tuple, List
import numpy as np
import librosa
from .silence import activity_ratio
from .types import TrackFeatures


def sanitize_beat_times(beat_times):
    if beat_times is None:
        return np.asarray([], dtype=float)
    beat_times = np.asarray(beat_times, dtype=float)
    if len(beat_times) > 0:
        beat_times = np.unique(np.sort(beat_times))
    return beat_times


def estimate_bpm_and_beats(y: np.ndarray, sr: int, probe_windows=((10, 55), (60, 105))) -> Tuple[float, np.ndarray]:
    beat_times_all = []
    bpms = []
    for a, b in probe_windows:
        s = int(a * sr)
        e = int(b * sr)
        if s >= len(y):
            continue
        e = min(e, len(y))
        seg = y[s:e]
        if len(seg) < sr * 10:
            continue
        tempo, beats = librosa.beat.beat_track(y=seg, sr=sr, units="time")
        bpms.append(float(tempo))
        if beats is not None and len(beats) > 0:
            beat_times_all.append(beats + a)
    bpm = float(np.median(bpms)) if bpms else 0.0
    bt = sanitize_beat_times(np.concatenate(beat_times_all) if beat_times_all else None)
    return bpm, bt


def build_bar_candidates(beat_times: np.ndarray, bpm: float, duration_sec: float) -> List[float]:
    if bpm <= 0:
        return []
    spb = 60.0 / bpm
    spbar = spb * 4.0
    if beat_times is None or len(beat_times) == 0:
        n = int(duration_sec // spbar)
        return [i * spbar for i in range(1, n)]
    t0 = float(beat_times[0])
    bars = []
    t = t0 + spbar
    while t < duration_sec:
        bars.append(t)
        t += spbar
    return bars


def stability_score(y: np.ndarray, sr: int, probe_windows=((10, 25), (40, 55), (70, 85))) -> float:
    rms = []
    for a, b in probe_windows:
        s = int(a * sr)
        e = int(b * sr)
        if s >= len(y):
            continue
        e = min(e, len(y))
        seg = y[s:e]
        if len(seg) < sr:
            continue
        rms.append(float(np.sqrt(np.mean(seg * seg))))
    if len(rms) < 2:
        return 0.5
    r = np.asarray(rms, dtype=float)
    cv = float(np.std(r) / (np.mean(r) + 1e-8))
    return float(1.0 / (1.0 + 5.0 * cv))


def stretch_cost_score(ratio: float) -> float:
    d = abs(1.0 - ratio)
    if ratio < 0.90 or ratio > 1.10:
        return 0.05
    if d <= 0.05:
        return float(1.0 - (d / 0.05) * 0.2)
    return float(max(0.2, 0.8 - (d - 0.05) * 6.0))


def compute_track_features(
    name: str,
    sr: int,
    duration_sec: float,
    bus_audio: Dict[str, np.ndarray],
    bpm: float,
    stretch_ratio: float,
    target_duration_sec: float = 150.0,
) -> TrackFeatures:
    notes = []
    cont = {
        "drums": activity_ratio(bus_audio.get("drums"), sr, target_duration_sec),
        "bass": activity_ratio(bus_audio.get("bass"), sr, target_duration_sec),
        "harmony": activity_ratio(bus_audio.get("harmony"), sr, target_duration_sec),
    }

    # ✅ numpy array에서 or 금지 -> None 체크로 선택
    if bus_audio.get("mix") is not None:
        ref = bus_audio.get("mix")
    elif bus_audio.get("harmony") is not None:
        ref = bus_audio.get("harmony")
    else:
        ref = bus_audio.get("drums")

    stab = stability_score(ref, sr) if ref is not None else 0.5
    sc = stretch_cost_score(stretch_ratio)
    if bpm <= 0:
        notes.append("bpm_estimate_failed")

    return TrackFeatures(
        name=name,
        bpm=float(bpm),
        sr=int(sr),
        duration_sec=float(duration_sec),
        continuity=cont,
        stability=float(stab),
        stretch_cost=float(sc),
        notes=notes,
    )


def choose_backbone(t1: TrackFeatures, t2: TrackFeatures) -> Dict:
    w_cont, w_stab, w_stretch = 0.55, 0.25, 0.20

    def total(tf: TrackFeatures) -> float:
        c = 0.30 * tf.continuity.get("drums", 0) + 0.15 * tf.continuity.get("bass", 0) + 0.10 * tf.continuity.get("harmony", 0)
        return w_cont * c + w_stab * tf.stability + w_stretch * tf.stretch_cost

    s1 = total(t1)
    s2 = total(t2)
    bb = "T1" if s1 >= s2 else "T2"
    inj = "T2" if bb == "T1" else "T1"
    return {
        "backbone": bb,
        "inject": inj,
        "scores": {
            "T1": {"continuity": t1.continuity, "stability": t1.stability, "stretch": t1.stretch_cost, "total": s1},
            "T2": {"continuity": t2.continuity, "stability": t2.stability, "stretch": t2.stretch_cost, "total": s2},
        },
        "track_names": {"T1": t1.name, "T2": t2.name},
    }
