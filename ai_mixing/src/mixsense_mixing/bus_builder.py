from __future__ import annotations
import os, json, time, re
from typing import Dict, Tuple, List
import numpy as np

from .io import unpack_zip, list_audio_files, load_mono, write_wav_48k24, safe_track_id, guess_role
from .silence import is_effectively_silent
from .dsp import highpass, lowpass, loudness_match, soft_normalize


def _sum_tracks(tracks: List[np.ndarray]) -> np.ndarray:
    if not tracks:
        return np.zeros((0,), dtype=np.float32)
    L = max(len(y) for y in tracks)
    out = np.zeros((L,), dtype=np.float32)
    for y in tracks:
        out[: len(y)] += y
    return soft_normalize(out, peak=0.98)


def _pick_fullmix_file(files: List[str]) -> str | None:
    """
    zip 내부 오디오 파일들 중에서 '진짜 곡(full_mix/master/mix/track)'을 가장 먼저 선택.
    No Love / KeepGoing 같은 구조(mix+master/full_mix.wav)에 맞춤.
    """
    scored = []
    for f in files:
        bn = os.path.basename(f).lower()
        path = f.lower()
        s = 0

        # 최우선
        if re.search(r"full[_\s-]?mix", bn): s += 100
        if "master" in bn: s += 80

        # 그 다음
        if "mix" in bn: s += 50
        if "track" in bn: s += 40

        # 폴더 힌트
        if "mix+master" in path or "mixmaster" in path: s += 20
        if "stems" in path: s -= 10  # stems는 후보에서 밀어냄

        scored.append((s, f))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored or scored[0][0] <= 0:
        return None
    return scored[0][1]


def build_or_load_bus(zip_path: str, data_dir: str, sr: int = 48000) -> Tuple[str, Dict[str, np.ndarray], Dict]:
    track_id = safe_track_id(zip_path)
    bus_dir = os.path.join(data_dir, "processed", "bus", track_id)
    meta_path = os.path.join(bus_dir, "meta.json")

    if os.path.exists(meta_path):
        bus = {}
        for role, fn in [("drums", "drum_bus.wav"), ("bass", "bass_bus.wav"), ("harmony", "harmony_bus.wav"), ("mix", "mix_bus.wav")]:
            p = os.path.join(bus_dir, fn)
            if os.path.exists(p):
                y, _ = load_mono(p, sr=sr)
                bus[role] = y
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return track_id, bus, meta

    os.makedirs(bus_dir, exist_ok=True)
    tmp = os.path.join(data_dir, "cache", "tmp_unpack", track_id)
    os.makedirs(tmp, exist_ok=True)

    unpack_zip(zip_path, tmp)
    files = list_audio_files(tmp)
    if not files:
        raise RuntimeError(f"zip에 오디오 파일이 없음: {zip_path}")

    notes = []
    role_best: dict[str, tuple[int, np.ndarray, str]] = {}

    # ✅ 1) full_mix/master 먼저 찾기
    fullmix_path = _pick_fullmix_file(files)
    mix_y = None
    if fullmix_path:
        y, _ = load_mono(fullmix_path, sr=sr)
        if y is not None and y.size > 0:
            mix_y = soft_normalize(y, peak=0.98)
            notes.append(f"mix_from_fullmix:{os.path.basename(fullmix_path)}")

    # ✅ 2) stems는 drums/bass/harmony만 잡는다 (mix가 있더라도 stem 합으로 mix 만들지 말 것)
    for fpath in files:
        role = guess_role(fpath)
        y, _ = load_mono(fpath, sr=sr)

        # mix 후보는 위에서 이미 고정했으니 여기서는 스킵(중복 방지)
        if role == "mix":
            continue

        if is_effectively_silent(y, sr):
            notes.append(f"silence_excluded:{os.path.basename(fpath)}")
            continue

        if role not in role_best or len(y) > role_best[role][0]:
            role_best[role] = (len(y), y, fpath)

    drums = role_best.get("drums", (0, None, ""))[1]
    bass = role_best.get("bass", (0, None, ""))[1]
    harmony = role_best.get("harmony", (0, None, ""))[1]

    # ✅ 3) mix_y가 없을 때만 마지막 수단으로 stem 합
    if mix_y is None:
        non_vocal = [v[1] for k, v in role_best.items() if k != "vocal" and v[1] is not None]
        mix_y = _sum_tracks(non_vocal) if non_vocal else np.zeros((0,), dtype=np.float32)
        notes.append("mix_from_sum_nonvocal_fallback")

    # proxies if missing
    if drums is None and mix_y.size > 0:
        # 간단 proxy: HPSS 대신 안전한 highpass/lowpass 분리
        drums = highpass(mix_y, sr, cutoff_hz=200.0)
        notes.append("drums_proxy_from_highpass_200")

    if bass is None and mix_y.size > 0:
        bass = lowpass(mix_y, sr, cutoff_hz=180.0)
        notes.append("bass_proxy_from_lowpass_180")

    if harmony is None and mix_y.size > 0:
        harmony = highpass(mix_y, sr, cutoff_hz=180.0)
        notes.append("harmony_proxy_from_highpass_180")

    # loudness match to mix
    if mix_y.size > 0:
        if drums is not None:
            drums = loudness_match(drums, mix_y)
        if bass is not None:
            bass = loudness_match(bass, mix_y)
        if harmony is not None:
            harmony = loudness_match(harmony, mix_y)

    # save
    write_wav_48k24(os.path.join(bus_dir, "mix_bus.wav"), soft_normalize(mix_y), sr=sr)
    if drums is not None:
        write_wav_48k24(os.path.join(bus_dir, "drum_bus.wav"), soft_normalize(drums), sr=sr)
    if bass is not None:
        write_wav_48k24(os.path.join(bus_dir, "bass_bus.wav"), soft_normalize(bass), sr=sr)
    if harmony is not None:
        write_wav_48k24(os.path.join(bus_dir, "harmony_bus.wav"), soft_normalize(harmony), sr=sr)

    meta = {
        "track_id": track_id,
        "source_zip": os.path.basename(zip_path),
        "sr": sr,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": notes,
        "files": [fn for fn in ["mix_bus.wav", "drum_bus.wav", "bass_bus.wav", "harmony_bus.wav"] if os.path.exists(os.path.join(bus_dir, fn))],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # reload
    bus = {}
    for role, fn in [("drums", "drum_bus.wav"), ("bass", "bass_bus.wav"), ("harmony", "harmony_bus.wav"), ("mix", "mix_bus.wav")]:
        p = os.path.join(bus_dir, fn)
        if os.path.exists(p):
            y, _ = load_mono(p, sr=sr)
            bus[role] = y

    return track_id, bus, meta
