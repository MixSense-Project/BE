from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfilt

def s_curve(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones((n,), dtype=np.float32)
    x = np.linspace(0, np.pi, n, dtype=np.float32)
    return (1.0 - np.cos(x)) * 0.5

def fade_in(y: np.ndarray, n: int) -> np.ndarray:
    n = min(n, len(y))
    if n <= 0:
        return y
    w = s_curve(n)
    out = y.copy()
    out[:n] *= w
    return out

def fade_out(y: np.ndarray, n: int) -> np.ndarray:
    n = min(n, len(y))
    if n <= 0:
        return y
    w = s_curve(n)[::-1]
    out = y.copy()
    out[-n:] *= w
    return out

def lowpass(y: np.ndarray, sr: int, cutoff_hz: float = 11000.0, order: int = 4) -> np.ndarray:
    sos = butter(order, cutoff_hz, btype="lowpass", fs=sr, output="sos")
    return sosfilt(sos, y).astype(np.float32)

def highpass(y: np.ndarray, sr: int, cutoff_hz: float = 120.0, order: int = 4) -> np.ndarray:
    sos = butter(order, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return sosfilt(sos, y).astype(np.float32)

def rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y*y)) + 1e-8)

def loudness_match(y: np.ndarray, ref: np.ndarray) -> np.ndarray:
    g = rms(ref) / rms(y)
    return (y * g).astype(np.float32)

def soft_normalize(y: np.ndarray, peak: float = 0.98) -> np.ndarray:
    m = float(np.max(np.abs(y)) + 1e-8)
    if m > peak:
        y = y * (peak / m)
    return y.astype(np.float32)
