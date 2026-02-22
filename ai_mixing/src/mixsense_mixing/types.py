from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class TrackFeatures:
    name: str
    bpm: float
    sr: int
    duration_sec: float
    continuity: Dict[str, float]
    stability: float
    stretch_cost: float
    notes: List[str]

@dataclass
class TransitionEvent:
    t_sec: float
    bars: int
    from_track: str
    to_track: str
    stretch_ratio: float
    mode: str
    detail: Dict[str, Any]

@dataclass
class MixResult:
    out_wav_path: str
    log_json_path: str
    used_k: int
    events: List[TransitionEvent]
    backbone: str
    inject: str
