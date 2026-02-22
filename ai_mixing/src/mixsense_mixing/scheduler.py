from __future__ import annotations
from typing import List
from .types import TransitionEvent

def schedule_transitions(
    duration_sec: float,
    target_k: int,
    bar_times: List[float],
    stretch_ratio: float,
    min_gap_sec: float = 28.0,
    end_takeover_sec: float = 18.0,
) -> List[TransitionEvent]:
    """
    ✅ '노래처럼' 들리게:
    - phrase(8bar) 경계에 가깝게 이벤트를 잡음
    - 이벤트는 길게(기본 4bar ramp) 잡고, 실제 렌더는 weight curve로 처리( render.py )
    """
    events: List[TransitionEvent] = []
    end_start = max(0.0, duration_sec - end_takeover_sec)

    # 후보: 너무 앞(인트로) / 너무 뒤(엔딩 takeover) 제외
    candidates = [t for t in bar_times if 24.0 < t < end_start - 24.0]
    if not candidates:
        return events

    # 목표 k는 4~6 사이 고정(체감)
    k = max(4, min(6, int(target_k)))

    # phrase(8bar) 경계에 가까운 t만 남기기: 바 간격이 대략 sec_per_bar일 때 8bar마다 찍히는 후보
    # bar_times 자체가 bar 후보라서, 여기서는 단순히 "띄엄띄엄" 고르되 min_gap을 크게 둔다.
    picked = []
    last = -1e9
    step = max(1, len(candidates) // (k + 1))
    for i in range(step, len(candidates), step):
        t = float(candidates[i])
        if t - last < min_gap_sec:
            continue
        picked.append(t)
        last = t
        if len(picked) >= k:
            break

    for t in picked:
        events.append(
            TransitionEvent(
                t_sec=float(t),
                bars=4,  # ✅ ramp 길이(바 단위). render에서 ramp/hold/release로 확장함
                from_track="A",
                to_track="B",
                stretch_ratio=float(stretch_ratio),
                mode="weight-curve",
                detail={"intent": "ramp_hold_release", "ramp_bars": 4, "hold_bars": 4, "release_bars": 4},
            )
        )

    return events
