from __future__ import annotations
import re
import numpy as np
import librosa

# ---------- BPM ----------
def parse_bpm_from_name(name: str) -> float:
    if not name:
        return 0.0
    s = name.lower()
    m = re.search(r"(\d{2,3})\s*bpm", s) or re.search(r"bpm\s*(\d{2,3})", s)
    if not m:
        return 0.0
    v = float(m.group(1))
    return v if 40 <= v <= 250 else 0.0

def estimate_bpm(y: np.ndarray, sr: int) -> float:
    if y is None or y.size < sr * 10:
        return 0.0
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo)

def normalize_bpm(bpm: float, lo: float = 80.0, hi: float = 180.0) -> float:
    if bpm <= 0:
        return 0.0
    cands = [bpm, bpm*2, bpm*4, bpm/2, bpm/4]
    inr = [c for c in cands if lo <= c <= hi]
    if inr:
        inr.sort(key=lambda c: abs(c - bpm))
        return float(inr[0])
    mid = (lo + hi) / 2
    cands.sort(key=lambda c: abs(c - mid))
    return float(cands[0])

# ---------- KEY (Krumhansl-like) ----------
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=float)
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=float)

_NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def estimate_key(y: np.ndarray, sr: int) -> tuple[str, float, int]:
    """
    returns (key_str, confidence, key_index 0-11 where 0=C)
    confidence는 0~1 느낌. 낮으면 UNKNOWN 처리 권장.
    """
    if y is None or y.size < sr * 10:
        return ("UNKNOWN", 0.0, -1)

    # chroma (CQT)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    pcp = chroma.mean(axis=1)
    if np.allclose(pcp.sum(), 0):
        return ("UNKNOWN", 0.0, -1)

    pcp = pcp / (pcp.sum() + 1e-9)

    best = None
    scores = []

    for mode_name, tmpl in [("major", _MAJOR), ("minor", _MINOR)]:
        tmpl = tmpl / tmpl.sum()
        for k in range(12):
            rot = np.roll(tmpl, k)
            s = float(np.corrcoef(pcp, rot)[0, 1])
            scores.append(s)
            if best is None or s > best[0]:
                best = (s, k, mode_name)

    if best is None:
        return ("UNKNOWN", 0.0, -1)

    best_score = best[0]
    # confidence: best와 2등 차이 기반
    s_sorted = sorted(scores, reverse=True)
    gap = float(s_sorted[0] - s_sorted[1]) if len(s_sorted) > 1 else 0.0
    conf = float(np.clip((best_score + 1) / 2 * 0.7 + np.clip(gap, 0, 0.2) * 1.5, 0, 1))

    key_name = _NOTE_NAMES[best[1]] + ("" if best[2] == "major" else "m")
    return (key_name, conf, int(best[1]))

def key_distance(k1: int, k2: int) -> int:
    if k1 < 0 or k2 < 0:
        return 99
    d = abs(k1 - k2) % 12
    return int(min(d, 12 - d))

# ---------- ENERGY / RHYTHM ----------
def rms_stats(y: np.ndarray, sr: int) -> dict:
    if y is None or y.size == 0:
        return {"rms_mean": 0.0, "rms_std": 0.0, "rms_slope": 0.0}
    hop = 512
    frame = 2048
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    if rms.size < 4:
        return {"rms_mean": float(rms.mean() if rms.size else 0.0), "rms_std": float(rms.std() if rms.size else 0.0), "rms_slope": 0.0}
    x = np.linspace(0, 1, rms.size)
    slope = float(np.polyfit(x, rms, 1)[0])
    return {"rms_mean": float(rms.mean()), "rms_std": float(rms.std()), "rms_slope": slope}

def onset_stats(y: np.ndarray, sr: int) -> dict:
    if y is None or y.size == 0:
        return {"onset_mean": 0.0, "onset_std": 0.0}
    o = librosa.onset.onset_strength(y=y, sr=sr)
    return {"onset_mean": float(o.mean() if o.size else 0.0), "onset_std": float(o.std() if o.size else 0.0)}

def spectral_stats(y: np.ndarray, sr: int) -> dict:
    if y is None or y.size == 0:
        return {"centroid": 0.0, "rolloff": 0.0}
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    return {"centroid": float(np.mean(cent)), "rolloff": float(np.mean(roll))}
