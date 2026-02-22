from __future__ import annotations
import numpy as np

def is_effectively_silent(y: np.ndarray, sr: int,
                          windows=((10, 25), (40, 55), (70, 85)),
                          peak_th=1e-4, rms_th=1e-5) -> bool:
    if y is None or y.size == 0:
        return True
    L = len(y)
    for a, b in windows:
        s = int(a*sr); e = int(b*sr)
        if s >= L:
            continue
        e = min(e, L)
        seg = y[s:e]
        if seg.size < sr:
            continue
        peak = float(np.max(np.abs(seg)))
        rms = float(np.sqrt(np.mean(seg*seg)))
        if peak > peak_th and rms > rms_th:
            return False
    return True

def activity_ratio(y: np.ndarray, sr: int, target_duration_sec: float,
                   frame_sec: float = 0.5, rms_th: float = 2e-5) -> float:
    if y is None or y.size == 0:
        return 0.0
    L = min(len(y), int(target_duration_sec*sr))
    y = y[:L]
    hop = max(1, int(frame_sec*sr))
    n = max(1, L // hop)
    active = 0
    for i in range(n):
        seg = y[i*hop:(i+1)*hop]
        if seg.size == 0:
            continue
        rms = float(np.sqrt(np.mean(seg*seg)))
        if rms > rms_th:
            active += 1
    return active / n
