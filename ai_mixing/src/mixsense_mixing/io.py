from __future__ import annotations
import os, zipfile, re
from typing import List, Tuple
import numpy as np
import librosa
import soundfile as sf

def unpack_zip(zip_path: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(out_dir)
    return out_dir

def list_audio_files(root: str) -> List[str]:
    exts = {".wav",".flac",".mp3",".m4a",".aiff",".aif"}
    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.startswith("._") or fn == ".DS_Store":
                continue
            if os.path.splitext(fn)[1].lower() in exts:
                files.append(os.path.join(dp, fn))
    return sorted(files)

def load_mono(path: str, sr: int = 48000) -> Tuple[np.ndarray, int]:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32), sr

def write_wav_48k24(path: str, y: np.ndarray, sr: int = 48000) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y, sr, subtype="PCM_24")

def safe_track_id(zip_path: str) -> str:
    base = os.path.splitext(os.path.basename(zip_path))[0]
    base = re.sub(r"[^a-zA-Z0-9_-]+", "_", base).strip("_")
    return base or "track"

def guess_role(filename: str) -> str:
    fn = os.path.basename(filename).lower()
    if re.search(r"\b(vox|vocal|voice)\b", fn):
        return "vocal"
    if re.search(r"\b(drums?|kick|snare|hihat|hat|perc|percussion)\b", fn):
        return "drums"
    if re.search(r"\b(bass|sub)\b", fn):
        return "bass"
    if re.search(r"\b(mix|master|full|stereo)\b", fn):
        return "mix"
    return "harmony"
