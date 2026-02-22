from __future__ import annotations
import re
from datetime import datetime
from typing import Dict, Set
from .schemas import LLMIntent
from .taxonomy import canonicalize_list, build_canonical_map

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def clamp(x: float, lo: float, hi: float) -> float:
    try:
        v = float(x)
    except Exception:
        return lo
    return max(lo, min(hi, v))

def validate_date(s: str | None) -> str | None:
    if not s:
        return None
    s = str(s).strip()
    if not _DATE_RE.match(s):
        return None
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return s
    except Exception:
        return None

def validate_intent(
    intent: LLMIntent,
    allowed_values: Dict[str, list],
    known_artist_ids: Set[str],
    known_track_ids: Set[str],
) -> LLMIntent:
    genre_map = build_canonical_map(allowed_values.get("genre_in", []))
    sub_genre_map = build_canonical_map(allowed_values.get("sub_genre_in", []))
    mood_map = build_canonical_map(allowed_values.get("mood_tags_in", []))
    ctx_map = build_canonical_map(allowed_values.get("context_tags_in", []))

    f = intent.filters

    f.genre_in = canonicalize_list(f.genre_in, genre_map)
    f.sub_genre_in = canonicalize_list(f.sub_genre_in, sub_genre_map)
    f.mood_tags_in = canonicalize_list(f.mood_tags_in, mood_map)
    f.context_tags_in = canonicalize_list(f.context_tags_in, ctx_map)

    if f.popularity_min is not None:
        f.popularity_min = clamp(f.popularity_min, 0.0, 100.0)
    if f.popularity_max is not None:
        f.popularity_max = clamp(f.popularity_max, 0.0, 100.0)
    if f.popularity_min is not None and f.popularity_max is not None and f.popularity_min > f.popularity_max:
        f.popularity_min, f.popularity_max = f.popularity_max, f.popularity_min

    f.release_date_from = validate_date(f.release_date_from)
    f.release_date_to = validate_date(f.release_date_to)

    f.artist_id_in = [a for a in f.artist_id_in if a in known_artist_ids]
    f.exclude_artist_ids = [a for a in f.exclude_artist_ids if a in known_artist_ids]
    f.exclude_track_ids = [t for t in f.exclude_track_ids if t in known_track_ids]

    intent.confidence = clamp(intent.confidence, 0.0, 1.0)

    w = intent.ranking_weights
    w.w_artist_pref = clamp(w.w_artist_pref, 0.0, 1.0)
    w.w_popularity = clamp(w.w_popularity, 0.0, 1.0)
    w.w_recency = clamp(w.w_recency, 0.0, 1.0)
    w.w_mood_match = clamp(w.w_mood_match, 0.0, 1.0)
    w.w_context_match = clamp(w.w_context_match, 0.0, 1.0)

    d = intent.diversity_rules
    d.max_tracks_per_artist = max(1, int(d.max_tracks_per_artist or 1))

    return intent
