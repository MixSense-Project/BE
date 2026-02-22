from __future__ import annotations
from typing import Dict, List
import os
import pandas as pd
import numpy as np
from .schemas import LLMIntent, SearchResult, SearchResponse, RankingWeights

DEFAULT_WEIGHTS = RankingWeights(
    w_artist_pref=0.35,
    w_popularity=0.40,
    w_recency=0.25,
    w_mood_match=0.10,
    w_context_match=0.10,
)

def _normalize_weights(w: RankingWeights) -> RankingWeights:
    """
    v1 default: ignore LLM-provided weights for reproducibility.
    Enable LLM-controlled weights by setting:
      MIXSENSE_LLM_WEIGHTS=1
    """
    allow = str(os.environ.get("MIXSENSE_LLM_WEIGHTS", "0")).strip().lower() in {"1","true","yes","y"}
    if not allow:
        return DEFAULT_WEIGHTS

    vals = np.array([w.w_artist_pref, w.w_popularity, w.w_recency, w.w_mood_match, w.w_context_match], dtype=float)
    s = float(vals.sum())
    if s <= 1e-9:
        return DEFAULT_WEIGHTS
    vals = vals / s
    return RankingWeights(
        w_artist_pref=float(vals[0]),
        w_popularity=float(vals[1]),
        w_recency=float(vals[2]),
        w_mood_match=float(vals[3]),
        w_context_match=float(vals[4]),
    )

def _overlap_score(track_tags: List[str], query_tags: List[str]) -> float:
    if not query_tags or not track_tags:
        return 0.0
    s_track = set(track_tags)
    s_query = set(query_tags)
    return len(s_track & s_query) / max(1, len(s_query))

def filter_candidates(df: pd.DataFrame, intent: LLMIntent) -> pd.DataFrame:
    f = intent.filters
    out = df

    out = out[out["youtube_video_id"].notna()]

    if f.exclude_track_ids:
        out = out[~out["track_id"].isin(f.exclude_track_ids)]
    if f.exclude_artist_ids:
        out = out[~out["artist_id"].isin(f.exclude_artist_ids)]
    if f.artist_id_in:
        out = out[out["artist_id"].isin(f.artist_id_in)]

    if f.genre_in:
        out = out[out["genre"].isin(f.genre_in)]
    if f.sub_genre_in:
        out = out[out["sub_genre"].isin(f.sub_genre_in)]

    if f.popularity_min is not None:
        out = out[out["popularity"] >= f.popularity_min]
    if f.popularity_max is not None:
        out = out[out["popularity"] <= f.popularity_max]

    if f.release_date_from:
        out = out[out["release_date"] >= pd.to_datetime(f.release_date_from)]
    if f.release_date_to:
        out = out[out["release_date"] <= pd.to_datetime(f.release_date_to)]

    return out

def rank_candidates(df: pd.DataFrame, intent: LLMIntent) -> pd.DataFrame:
    w = _normalize_weights(intent.ranking_weights)
    mood_q = intent.filters.mood_tags_in
    ctx_q = intent.filters.context_tags_in

    out = df.copy()
    out["mood_match"] = out["mood_tags_list"].apply(lambda xs: _overlap_score(xs, mood_q))
    out["context_match"] = out["context_tags_list"].apply(lambda xs: _overlap_score(xs, ctx_q))
    out["score"] = (
        w.w_artist_pref * out["artist_pref_score"] +
        w.w_popularity  * out["popularity_norm"] +
        w.w_recency     * out["recency_score"] +
        w.w_mood_match  * out["mood_match"] +
        w.w_context_match * out["context_match"]
    )
    return out.sort_values("score", ascending=False)

def apply_diversity(df: pd.DataFrame, max_tracks_per_artist: int, k: int) -> pd.DataFrame:
    if max_tracks_per_artist <= 0:
        max_tracks_per_artist = 1
    kept = []
    counts: Dict[str, int] = {}
    for _, row in df.iterrows():
        aid = row["artist_id"]
        c = counts.get(aid, 0)
        if c >= max_tracks_per_artist:
            continue
        counts[aid] = c + 1
        kept.append(row)
        if len(kept) >= k:
            break
    return pd.DataFrame(kept) if kept else df.head(0)

def to_results(df: pd.DataFrame, k: int) -> List[SearchResult]:
    base_cols = ["track_id","title","artist","artist_id","youtube_video_id","popularity","release_date","genre","sub_genre","score"]
    optional_cols = [c for c in ["track_image_url","thumbnail_url"] if c in df.columns]
    cols = base_cols + optional_cols

    out: List[SearchResult] = []
    for _, r in df.head(k)[cols].iterrows():
        track_image_url = r.get("track_image_url", None)
        thumbnail_url = r.get("thumbnail_url", None)
        out.append(SearchResult(
            track_id=str(r["track_id"]),
            title=str(r["title"]),
            artist=str(r["artist"]),
            artist_id=str(r["artist_id"]),
            youtube_video_id=str(r["youtube_video_id"]),
            popularity=float(r["popularity"]),
            release_date=str(pd.to_datetime(r["release_date"]).date()),
            genre=str(r["genre"]),
            sub_genre=(None if pd.isna(r["sub_genre"]) else str(r["sub_genre"])),
            score=float(r["score"]),
            track_image_url=(None if pd.isna(track_image_url) else str(track_image_url)),
            thumbnail_url=(None if pd.isna(thumbnail_url) else str(thumbnail_url)),
        ))
    return out

def apply_fallback_step(intent: LLMIntent, step: str) -> LLMIntent:
    i2 = intent.model_copy(deep=True)
    f = i2.filters
    step = str(step).strip().upper()
    if step == "DROP_CONTEXT":
        f.context_tags_in = []
    elif step == "DROP_MOOD":
        f.mood_tags_in = []
    elif step == "WIDEN_POPULARITY":
        if f.popularity_min is not None:
            f.popularity_min = max(0.0, float(f.popularity_min) - 10.0)
        if f.popularity_max is not None:
            f.popularity_max = min(100.0, float(f.popularity_max) + 10.0)
    elif step == "WIDEN_DATE_RANGE":
        f.release_date_from = None
        f.release_date_to = None
    elif step == "DROP_GENRE":
        f.genre_in = []
        f.sub_genre_in = []
    return i2

def search_with_fallback(df: pd.DataFrame, intent: LLMIntent, k: int = 20) -> SearchResponse:
    debug = {}
    cand = filter_candidates(df, intent)
    debug["base_candidate_count"] = int(len(cand))

    if len(cand) > 0:
        ranked = rank_candidates(cand, intent)
        diverse = apply_diversity(ranked, intent.diversity_rules.max_tracks_per_artist, k)
        return SearchResponse(
            mode="base",
            used_intent=intent,
            n_candidates=int(len(cand)),
            results=to_results(diverse, k),
            debug=debug,
        )

    for step in intent.fallback_plan:
        i2 = apply_fallback_step(intent, step)
        cand2 = filter_candidates(df, i2)
        debug[f"candidate_after_{step}"] = int(len(cand2))
        if len(cand2) > 0:
            ranked2 = rank_candidates(cand2, i2)
            diverse2 = apply_diversity(ranked2, i2.diversity_rules.max_tracks_per_artist, k)
            return SearchResponse(
                mode=f"fallback:{step}",
                used_intent=i2,
                n_candidates=int(len(cand2)),
                results=to_results(diverse2, k),
                debug=debug,
            )

    return SearchResponse(
        mode="no_results",
        used_intent=intent,
        n_candidates=0,
        results=[],
        debug=debug,
    )
