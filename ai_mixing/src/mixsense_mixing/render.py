from __future__ import annotations
from typing import Dict
import numpy as np
from .dsp import soft_normalize, lowpass, highpass

def _ensure_len(y: np.ndarray, L: int) -> np.ndarray:
    if y is None:
        return np.zeros((L,), dtype=np.float32)
    if len(y) >= L:
        return y[:L].astype(np.float32)
    out = np.zeros((L,), dtype=np.float32)
    out[:len(y)] = y
    return out.astype(np.float32)

def _s_curve(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones((n,), dtype=np.float32)
    x = np.linspace(0, np.pi, n, dtype=np.float32)
    return (1.0 - np.cos(x)) * 0.5

def _ramp(L: int, sr: int, start_s: float, end_s: float, peak: float) -> np.ndarray:
    w = np.zeros((L,), dtype=np.float32)
    s = int(max(0.0, start_s) * sr)
    e = int(min(L / sr, end_s) * sr)
    if e <= s:
        return w
    seg = _s_curve(e - s) * float(peak)
    w[s:e] = seg
    return w

def render_styleB_from_bus(
    backbone_bus: Dict[str, np.ndarray],
    inject_bus: Dict[str, np.ndarray],
    sr: int,
    duration_sec: float,
    events,
    inject_ratio: float = 0.95,
    end_takeover_sec: float = 14.0,
) -> np.ndarray:
    """
    ✅ '믹싱처럼' 들리게 만드는 Stem-aware DJ render (최소버전)
    - Backbone: drums+bass를 계속 유지해 그루브 유지
    - Inject: harmony를 먼저 올리고, 그 다음 drums를 올림
    - 마지막엔 드럼 스왑(Backbone drums down, Inject drums up)
    """
    L = int(duration_sec * sr)

    # buses (없으면 mix에서 파생)
    A_mix = _ensure_len(backbone_bus.get("mix"), L)
    B_mix = _ensure_len(inject_bus.get("mix"), L)

    A_dr = _ensure_len(backbone_bus.get("drums"), L)
    A_bs = _ensure_len(backbone_bus.get("bass"), L)
    A_hy = _ensure_len(backbone_bus.get("harmony"), L)

    B_dr = _ensure_len(inject_bus.get("drums"), L)
    B_bs = _ensure_len(inject_bus.get("bass"), L)
    B_hy = _ensure_len(inject_bus.get("harmony"), L)

    master_bpm = float(backbone_bus.get("_master_bpm", 120.0))
    sec_per_bar = (60.0 / master_bpm) * 4.0 if master_bpm > 0 else 2.0

    # ---- 1) 이벤트 기반 타임라인 만들기 ----
    # 지금 로그처럼 events가 2개만 나와도, "누적"되게 만들어서 왔다갔다 없애기
    w_hy = np.zeros((L,), dtype=np.float32)  # harmony weight
    w_dr = np.zeros((L,), dtype=np.float32)  # drums weight
    w_bs = np.zeros((L,), dtype=np.float32)  # bass weight (늦게, 약하게)

    # 기본 정책:
    # - 각 이벤트 t에서: ramp_up(4bar) + hold(4bar) + release(4bar)인데
    #   release를 "완전 0"으로 내리지 않고 baseline을 남김(누적)
    baseline = 0.10  # B가 완전히 사라지지 않게 최소 존재감

    for ev in events:
        t = float(ev["t_sec"]) if isinstance(ev, dict) else float(ev.t_sec)
        ramp = 4 * sec_per_bar
        hold = 4 * sec_per_bar
        rel  = 4 * sec_per_bar

        # harmony: 제일 먼저 크게
        w_hy = np.maximum(w_hy, baseline + _ramp(L, sr, t - ramp, t, peak=inject_ratio))
        # hold 구간은 plateau
        s2 = int(t * sr); e2 = int(min(duration_sec, t + hold) * sr)
        if e2 > s2:
            w_hy[s2:e2] = np.maximum(w_hy[s2:e2], float(inject_ratio))
        # release는 baseline까지만 내려감
        s3 = e2; e3 = int(min(duration_sec, t + hold + rel) * sr)
        if e3 > s3:
            seg = (_s_curve(e3 - s3)[::-1] * float(inject_ratio))
            w_hy[s3:e3] = np.maximum(w_hy[s3:e3], baseline + seg)

        # drums: harmony보다 늦게/약하게
        w_dr = np.maximum(w_dr, _ramp(L, sr, t, t + ramp, peak=0.70 * inject_ratio))

        # bass: 더 늦게, 더 약하게 (키 안 맞으면 베이스가 제일 거슬림)
        w_bs = np.maximum(w_bs, _ramp(L, sr, t + hold, t + hold + ramp, peak=0.35 * inject_ratio))

    # ---- 2) 엔딩 드럼 스왑 ----
    # 마지막 end_takeover_sec 동안: A_dr down, B_dr up
    end_len = int(end_takeover_sec * sr)
    if 0 < end_len < L:
        s = L - end_len
        g = _s_curve(end_len)
        # 드럼은 확실히 넘어가게
        w_dr[s:L] = np.maximum(w_dr[s:L], 0.85 * g)
        # 하모니도 어느정도 따라가게
        w_hy[s:L] = np.maximum(w_hy[s:L], 0.75 * g)
        # 베이스는 너무 튀면 망하니까 제한
        w_bs[s:L] = np.maximum(w_bs[s:L], 0.45 * g)

    # ---- 3) 실제 합성 ----
    # backbone 그루브 유지: A_dr + A_bs를 중심으로
    # inject는 stem별 weight로 섞어준다
    # (필터는 과하게 안 씀. 과하면 또 “효과음” 됨)
    A_body = 0.55 * A_dr + 0.65 * A_bs + 0.45 * A_hy
    B_body = 0.55 * B_dr + 0.60 * B_bs + 0.50 * B_hy

    # stem별로 섞기
    out = A_body.copy()
    out += (B_hy * w_hy)
    out += (B_dr * w_dr)
    out += (B_bs * w_bs)

    # 너무 뭉개지면 하모니만 살짝 하이패스
    out = out + 0.15 * highpass(B_hy, sr, 150.0) * w_hy

    return soft_normalize(out.astype(np.float32), peak=0.98)
