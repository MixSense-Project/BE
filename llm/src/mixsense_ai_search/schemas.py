from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

IntentType = Literal["HOME", "SEARCH", "SIMILAR", "AUTOPLAY_HINT"]

class Filters(BaseModel):
    artist_id_in: List[str] = Field(default_factory=list)
    genre_in: List[str] = Field(default_factory=list)
    sub_genre_in: List[str] = Field(default_factory=list)

    # v1 recommendation: treat as SOFT ranking bonuses (not hard filters)
    mood_tags_in: List[str] = Field(default_factory=list)
    context_tags_in: List[str] = Field(default_factory=list)

    popularity_min: Optional[float] = None  # 0..100
    popularity_max: Optional[float] = None  # 0..100

    release_date_from: Optional[str] = None  # YYYY-MM-DD
    release_date_to: Optional[str] = None    # YYYY-MM-DD

    exclude_artist_ids: List[str] = Field(default_factory=list)
    exclude_track_ids: List[str] = Field(default_factory=list)

class RankingWeights(BaseModel):
    w_artist_pref: float = 0.0
    w_popularity: float = 0.0
    w_recency: float = 0.0
    w_mood_match: float = 0.0
    w_context_match: float = 0.0

class DiversityRules(BaseModel):
    max_tracks_per_artist: int = 1

class LLMIntent(BaseModel):
    intent_type: IntentType = "SEARCH"
    query_text: str

    filters: Filters = Field(default_factory=Filters)
    ranking_weights: RankingWeights = Field(default_factory=RankingWeights)
    diversity_rules: DiversityRules = Field(default_factory=DiversityRules)

    fallback_plan: List[str] = Field(default_factory=lambda: [
        "DROP_CONTEXT",
        "DROP_MOOD",
        "WIDEN_POPULARITY",
        "WIDEN_DATE_RANGE",
        "DROP_GENRE",
    ])

    confidence: float = 0.0  # 0..1

class SearchResult(BaseModel):
    track_id: str
    title: str
    artist: str
    artist_id: str
    youtube_video_id: str
    popularity: float
    release_date: str
    genre: str
    sub_genre: Optional[str] = None
    score: float
    # Optional artwork URLs (for UI). Prefer track_image_url if available; else thumbnail_url.
    track_image_url: Optional[str] = None
    thumbnail_url: Optional[str] = None

class ExternalResult(BaseModel):
    title: str
    artist: str
    youtube_video_id: str
    thumbnail_url: Optional[str] = None


class SearchResponse(BaseModel):
    mode: str  # base / fallback:<step> / no_results / clarify
    used_intent: LLMIntent
    n_candidates: int
    results: List[SearchResult] = Field(default_factory=list)
    # Optional: when we can't answer from our catalog, we can return YouTube search results.
    external_results: List[ExternalResult] = Field(default_factory=list)
    external_search_url: Optional[str] = None

    confidence_recalc: float = 0.0
    clarification_question: Optional[str] = None

    debug: dict = Field(default_factory=dict)
