from __future__ import annotations
import numpy as np

STRATS = ["DRUM_SWAP", "HARMONY_BLEND", "FULL_TAKEOVER", "NO_MIX"]

def teacher_strategy_label(feat: dict) -> str:
    """
    feat에는 bpm_diff, key_diff, key_conf_min, energy_sim 등 들어옴.
    """
    bpm_diff = feat.get("bpm_diff", 999.0)
    key_diff = feat.get("key_diff", 99)
    key_conf = feat.get("key_conf_min", 0.0)
    energy_sim = feat.get("energy_sim", 0.0)

    # key 신뢰도가 낮으면 하모니 금지 쪽으로
    harmonic_safe = (key_conf >= 0.55 and key_diff <= 2)

    # bpm 차이가 너무 크면 "완전 믹스" 금지
    tempo_safe = (bpm_diff <= 4.5)

    if not tempo_safe and not harmonic_safe:
        # 데모/서비스: 완전 금지 대신 가장 안전한 믹스 전략으로 폴백
        return "DRUM_SWAP"

    if tempo_safe and harmonic_safe:
        # 에너지 비슷하면 하모니 블렌드가 자연스러움
        return "HARMONY_BLEND" if energy_sim >= 0.5 else "FULL_TAKEOVER"

    if tempo_safe and not harmonic_safe:
        # 키 안 맞으면 드럼 중심으로만
        return "DRUM_SWAP"

    # tempo_safe가 아니지만 harmonic_safe면, 하모니는 제한적으로/후반 전환
    return "FULL_TAKEOVER"


def teacher_transition_score(cand: dict) -> float:
    """
    cand: 후보 시점 특징들
    """
    # 에너지가 내려가는 지점 + onset이 낮은 지점이 전이하기 좋다고 가정(약라벨)
    drop = np.clip(-cand.get("rms_slope_local", 0.0), 0, 1)
    calm = 1.0 / (1.0 + cand.get("onset_mean_local", 0.0) * 10)
    return float(0.6 * drop + 0.4 * calm)
