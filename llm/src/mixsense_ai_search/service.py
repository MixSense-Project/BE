from __future__ import annotations
from typing import Dict, Optional
import os
from datetime import date, timedelta
import pandas as pd
import re
from .schemas import LLMIntent, SearchResponse, ExternalResult
from .validate import validate_intent
from .engine import search_with_fallback
from .intent_parser import LLMIntentParser, rule_based_parse
from .youtube_search import youtube_search, to_external_results
from .query_refiner import refine_youtube_query_with_gpt

def load_catalog(path: str) -> pd.DataFrame:
    if path.endswith(".pkl"):
        return pd.read_pickle(path)
    return pd.read_csv(path)

def build_artist_lookup(df: pd.DataFrame) -> Dict[str, str]:
    def norm_artist(s: str) -> str:
        s = str(s).strip().lower()
        return " ".join(s.split())
    tmp = df[["artist","artist_id"]].dropna().copy()
    tmp["artist_norm"] = tmp["artist"].map(norm_artist)
    counts = tmp.groupby(["artist_norm","artist_id"]).size().reset_index(name="n")
    best = counts.sort_values(["artist_norm","n"], ascending=[True,False]).drop_duplicates("artist_norm")
    return dict(zip(best["artist_norm"], best["artist_id"]))

def resolve_artist_id_from_query(query: str, artist_norm_to_id: Dict[str,str]) -> Optional[str]:
    q = " ".join(query.strip().lower().split())
    hits = []
    for a_norm, aid in artist_norm_to_id.items():
        if not a_norm:
            continue
        a_norm = str(a_norm).strip()
        # Short artist names (e.g., 'iu') are common; match them conservatively as a whole token.
        if len(a_norm) < 3:
            if re.search(rf"(^|\s){re.escape(a_norm)}(\s|$)", q):
                hits.append(aid)
            continue
        if a_norm in q:
            hits.append(aid)
        if len(hits) >= 2:
            break
    return hits[0] if len(hits) == 1 else None

def apply_artist_aliases(query: str, artist_aliases: Dict[str, str] | None) -> str:
    """Replace known artist aliases in the query (e.g., '아이유' -> 'IU').

    This is intentionally simple and deterministic (no LLM) so it's auditable.
    """
    if not artist_aliases:
        return query
    q = str(query)
    # Longer keys first to avoid partial overlaps
    for k in sorted(artist_aliases.keys(), key=lambda s: len(str(s)), reverse=True):
        v = artist_aliases[k]
        if k and k in q:
            q = q.replace(k, v)
    return q

def inject_keyword_filters(intent: LLMIntent, raw_query: str, allowed_values: Dict[str, list]) -> None:
    """Deterministic keyword → tag injection for high-precision cases.

    We ONLY inject tags if they exist in allowed_values (taxonomy).
    This prevents hallucinations while reducing needless 'clarify' prompts.
    """
    q = str(raw_query)

    allowed_moods = set(allowed_values.get("mood_tags_in", []))
    allowed_ctx = set(allowed_values.get("context_tags_in", []))
    allowed_genres = set(allowed_values.get("genre_in", []))

    def add_ctx(tag: str):
        if tag in allowed_ctx and tag not in intent.filters.context_tags_in:
            intent.filters.context_tags_in.append(tag)

    def add_mood(tag: str):
        if tag in allowed_moods and tag not in intent.filters.mood_tags_in:
            intent.filters.mood_tags_in.append(tag)

    def add_genre(tag: str):
        if tag in allowed_genres and tag not in intent.filters.genre_in:
            intent.filters.genre_in.append(tag)

    # Korean holidays / family days
    if any(x in q for x in ["설날", "추석", "명절", "연휴"]):
        add_ctx("Holiday")
        add_ctx("Family")

    # seasons / vibes (common natural language)
    if any(x in q for x in ["햇빛", "맑은 날", "화창", "산뜻", "기분 좋은"]):
        add_ctx("Feel-good")
        add_ctx("Morning")
        add_mood("Cheerful")

    # Christmas / winter
    if any(x in q for x in ["크리스마스", "캐롤", "성탄"]):
        add_ctx("Christmas")
        add_ctx("Holiday")
        add_genre("Holiday")
    if any(x in q for x in ["겨울", "추운", "눈 오는"]):
        add_ctx("Winter")

def _detect_language_hint(text: str) -> str | None:
    s = str(text or "")
    if re.search(r"[\uac00-\ud7a3]", s):
        return "ko"
    if re.search(r"[\u3040-\u30ff]", s):
        return "ja"
    return None

def _extract_quoted_phrases(text: str) -> list[str]:
    s = str(text or "")
    phrases: list[str] = []
    patterns = [
        r'"([^"]{2,200})"',
        r"'([^']{2,200})'",
        r"“([^”]{2,200})”",
        r"‘([^’]{2,200})’",
    ]
    for pat in patterns:
        for mm in re.finditer(pat, s):
            ph = (mm.group(1) or "").strip()
            if ph:
                phrases.append(ph)
    # keep unique, longer first
    phrases = sorted(set(phrases), key=lambda x: len(x), reverse=True)
    return phrases

def clean_identification_query(text: str) -> str:
    """Best-effort cleaning: keep distinctive phrases, remove filler like '이 노래 뭐야'."""
    s = str(text or "").strip()
    if not s:
        return s

    phrases = _extract_quoted_phrases(s)
    if phrases:
        # If user quoted lyrics, that's usually the best search key.
        base = " ".join(phrases[:2])
    else:
        base = s

    # Remove common filler
    fillers = [
        r"이\s*노래\s*뭐(야|지)\??",
        r"노래\s*뭐(야|지)\??",
        r"무슨\s*노래(야|지)\??",
        r"제목(\s*이)?\s*뭐(야|지)\??",
        r"찾아\s*줘",
        r"알려\s*줘",
        r"뭐\s*야\??",
        r"가사(가)?",
        r"후렴(이)?",
        r"도입(이)?",
        r"부분(이)?",
        r"같은\s*데",
        r"같아",
    ]
    for f in fillers:
        base = re.sub(f, " ", base, flags=re.IGNORECASE)

    base = re.sub(r"\s+", " ", base).strip()
    # If too short, fall back to original
    return base if len(base) >= 2 else s

def looks_like_song_identification_query(query: str) -> bool:
    """Heuristic: user is asking 'what song is this?' by lyrics/description/phrase."""
    q = str(query or "")
    q_nospace = re.sub(r"\s+", "", q)

    # Explicit patterns
    if any(x in q for x in ["무슨 노래", "뭔 노래", "노래 뭐", "이 노래 뭐", "제목 뭐", "제목이 뭐", "찾아줘", "what song", "song name"]):
        return True

    # Mentions lyrics explicitly
    if "가사" in q and any(x in q for x in ["뭐", "찾아", "제목", "무슨"]):
        return True

    # Quoted lyric snippet often implies identification
    if _extract_quoted_phrases(q):
        # avoid misrouting pure recommendation requests
        if not any(x in q for x in ["추천", "들을", "플레이리스트", "비슷한", "같은 분위기"]):
            return True

    # Onomatopoeia / repeated syllables (common in ID queries)
    if re.search(r"(아\s*){3,}|(나\s*){3,}|(다메\s*){2,}", q):
        return True

    # Very "snippet-like" query: short but not a normal recommendation
    if len(q_nospace) <= 25 and not any(x in q for x in ["추천", "듣기 좋은", "플리", "노래 추천"]):
        # contains unusual repetition/ellipsis
        if "..." in q or "…" in q:
            return True

    return False

def is_foreign_only_hint(query: str) -> bool:
    q = str(query)
    return any(x in q for x in ["외국", "해외", "foreign", "international"])

def filter_foreign_only(df: pd.DataFrame) -> pd.DataFrame:
    # Best-effort: treat artists with Hangul in the artist string as 'Korean'.
    # This is imperfect but works for prototyping without country metadata.
    hangul = re.compile(r"[\uac00-\ud7a3]")
    return df[~df["artist"].astype(str).apply(lambda s: bool(hangul.search(s)))]

def recalc_confidence(intent: LLMIntent) -> float:
    f = intent.filters
    signal = 0.0
    if f.artist_id_in:
        signal += 0.25
    if f.genre_in or f.sub_genre_in:
        signal += 0.25
    if f.mood_tags_in or f.context_tags_in:
        signal += 0.15
    if f.popularity_min is not None or f.popularity_max is not None:
        signal += 0.20
    if f.release_date_from or f.release_date_to:
        signal += 0.15

    llm_conf = max(0.0, min(1.0, float(intent.confidence or 0.0)))
    return max(0.0, min(1.0, 0.5 * llm_conf + 0.5 * signal))

def ai_search(
    user_query: str,
    catalog_df: pd.DataFrame,
    allowed_values: Dict[str, list],
    llm_parser: Optional[LLMIntentParser] = None,
    seed_track_id: Optional[str] = None,
    k: int = 5,
    artist_aliases: Optional[Dict[str, str]] = None,
    enable_external_youtube_search: bool = True,
) -> SearchResponse:
    known_artist_ids = set(catalog_df["artist_id"].dropna().astype(str).unique())
    known_track_ids = set(catalog_df["track_id"].dropna().astype(str).unique())

    # Apply simple artist alias replacements (e.g., '아이유' -> 'IU') for both parsing and artist-id resolution.
    query_for_parse = apply_artist_aliases(user_query, artist_aliases)
    query_for_resolve = query_for_parse

    # Optional: 'foreign only' hint (best-effort, no country metadata).
    foreign_only = is_foreign_only_hint(user_query)
    df_for_search = filter_foreign_only(catalog_df) if foreign_only else catalog_df

    # 1) Song identification / lyric snippet queries → go external first (YouTube search).
    #    This avoids hallucinations and works even without catalog metadata.
    if enable_external_youtube_search and looks_like_song_identification_query(user_query):
        # Use a minimal intent just for logging/debug (no recommendations here).
        base_intent = rule_based_parse(query_for_parse)
        base_intent.query_text = user_query

        # Clean query and optionally refine with GPT (query generator, not a recommender).
        cleaned = clean_identification_query(user_query)
        lang = _detect_language_hint(user_query)

        # UX rule: identification queries should show TOP3 candidates (cap),
        # even if user asks for more. If user explicitly asks for 1, allow 1.
        identify_k = min(3, max(1, int(k)))

        plan = refine_youtube_query_with_gpt(user_query, model=os.environ.get("MIXSENSE_OPENAI_MODEL", "gpt-4o-mini"))
        # plan is usually a RefinedYouTubeQuery, but be robust if it comes back as a string.
        if isinstance(plan, str):
            if plan.strip():
                cleaned = plan.strip()
        else:
            if plan and getattr(plan, "query", None):
                cleaned = plan.query
                lang = getattr(plan, "language", None) or lang

        items, url = youtube_search(
            cleaned,
            max_results=identify_k,
            language=lang,
            order="relevance",
            video_category_id="10",
        )
        ext = (to_external_results(items) if items else [])[:identify_k]
        return SearchResponse(
            mode="external",
            used_intent=base_intent,
            n_candidates=0,
            results=[],
            external_results=[ExternalResult(**x) for x in ext],
            external_search_url=url,
            confidence_recalc=0.5,  # not meaningful for identification
            clarification_question=None,
            debug={"reason": "song_identification", "cleaned_query": cleaned, "foreign_only": foreign_only, "k": identify_k},
        )


    if llm_parser is None:
        llm_parser = LLMIntentParser(allowed_values=allowed_values)

    intent = llm_parser.parse(query_for_parse)
    if not intent.query_text:
        intent.query_text = user_query

    # SIMILAR override by seed_track_id (UI-driven)
    if seed_track_id and seed_track_id in known_track_ids:
        row = catalog_df[catalog_df["track_id"] == seed_track_id].iloc[0]
        intent.intent_type = "SIMILAR"
        intent.filters.genre_in = [row["genre"]] if pd.notna(row["genre"]) else []
        if pd.notna(row.get("sub_genre", None)):
            intent.filters.sub_genre_in = [row["sub_genre"]]
        intent.filters.mood_tags_in = list(row.get("mood_tags_list", []))[:3]
        intent.filters.context_tags_in = list(row.get("context_tags_list", []))[:3]
        intent.filters.exclude_track_ids = [seed_track_id]

    # Artist resolve (conservative, optional)
    if not intent.filters.artist_id_in:
        lookup = build_artist_lookup(catalog_df)
        aid = resolve_artist_id_from_query(query_for_resolve, lookup)
        if aid:
            intent.filters.artist_id_in = [aid]

    inject_keyword_filters(intent, user_query, allowed_values)

    intent = validate_intent(intent, allowed_values, known_artist_ids, known_track_ids)

    conf_recalc = recalc_confidence(intent)
    if conf_recalc < 0.28 and not (
        intent.filters.genre_in or intent.filters.sub_genre_in or
        intent.filters.mood_tags_in or intent.filters.context_tags_in or
        intent.filters.artist_id_in
    ):
        # If user is asking to identify a song by description/phrase, try YouTube search.
        if enable_external_youtube_search and looks_like_song_identification_query(user_query):
            items, url = youtube_search(user_query, max_results=min(8, max(3, k)))
            ext = to_external_results(items) if items else []
            return SearchResponse(
                mode="external",
                used_intent=intent,
                n_candidates=0,
                results=[],
                external_results=[ExternalResult(**x) for x in ext],
                external_search_url=url,
                confidence_recalc=conf_recalc,
                clarification_question=None,
                debug={"reason": "song_identification", "foreign_only": foreign_only},
            )

        return SearchResponse(
            mode="clarify",
            used_intent=intent,
            n_candidates=0,
            results=[],
            confidence_recalc=conf_recalc,
            clarification_question="장르/분위기/상황 중 하나만 더 구체화해줘. 예: R&B/힙합, 잔잔/신나는, 새벽/드라이브",
            debug={"reason": "low_specificity", "foreign_only": foreign_only},
        )

# If query targets a single artist (e.g., '아이유 앨범 노래들'), don't cap by artist diversity.
    if intent.filters.artist_id_in and len(intent.filters.artist_id_in) == 1:
        intent.diversity_rules.max_tracks_per_artist = max(intent.diversity_rules.max_tracks_per_artist, k)

    resp = search_with_fallback(df_for_search, intent, k=k)
    resp.confidence_recalc = conf_recalc

    # 2) "Latest/new" style requests: if our catalog can't satisfy (too few results or too old),
    #    fall back to YouTube search ordered by date.
    if enable_external_youtube_search and any(x in user_query for x in ["최신", "신곡", "요즘", "최근", "new", "latest", "이번"]):
        need_external = False
        if resp.mode == "no_results" or len(resp.results) < max(3, min(10, k)):
            need_external = True
        else:
            try:
                newest = max(pd.to_datetime(r.release_date) for r in resp.results)
                # If the newest result is older than ~6 months, it's probably not what the user means by "latest".
                if newest < (pd.to_datetime(date.today()) - pd.to_timedelta(180, unit="D")):
                    need_external = True
            except Exception:
                pass

        if need_external:
            # Build a YouTube query: prefer resolved artist name if any.
            yt_q = str(user_query)
            if intent.filters.artist_id_in and len(intent.filters.artist_id_in) == 1:
                aid = intent.filters.artist_id_in[0]
                try:
                    name = catalog_df[catalog_df["artist_id"] == aid]["artist"].mode().iloc[0]
                    if "앨범" in user_query:
                        yt_q = f"{name} new album"
                    else:
                        yt_q = f"{name} new song"
                except Exception:
                    pass

            lang = _detect_language_hint(user_query)
            items, url = youtube_search(
                yt_q,
                max_results=min(10, max(3, k)),
                language=lang,
                order="date",
                video_category_id="10",
            )
            ext = to_external_results(items) if items else []
            if ext or url:
                return SearchResponse(
                    mode="external",
                    used_intent=intent,
                    n_candidates=resp.n_candidates,
                    results=resp.results,  # kept for debugging; UI will show external when mode=external
                    external_results=[ExternalResult(**x) for x in ext],
                    external_search_url=url,
                    confidence_recalc=conf_recalc,
                    clarification_question=None,
                    debug={**(resp.debug or {}), "reason": "latest_fallback", "yt_query": yt_q, "foreign_only": foreign_only},
                )

    return resp
