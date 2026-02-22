from __future__ import annotations

import os
import re
import urllib.parse
from typing import List, Optional, Tuple, Dict, Any

import requests

_YT_SEARCH_ENDPOINT = "https://www.googleapis.com/youtube/v3/search"

def _clean_title(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    # remove common bracket suffixes but keep main text
    s2 = re.sub(r"\s*[\(\[].*?[\)\]]\s*", " ", s).strip()
    return re.sub(r"\s+", " ", s2).strip() or s

def split_artist_title(title: str, channel_title: str) -> Tuple[str, str]:
    """Best-effort parse for YouTube video titles.

    Returns: (artist, title)
    """
    raw = str(title or "")
    cleaned = _clean_title(raw)

    for sep in [" - ", " – ", " — ", " ― ", ": ", " | "]:
        if sep in cleaned:
            left, right = cleaned.split(sep, 1)
            left = left.strip()
            right = right.strip()
            # Heuristic: left part looks like artist
            if 1 <= len(left) <= 60 and 1 <= len(right) <= 120:
                return left, right

    # Fallback
    return (str(channel_title or "Unknown"), cleaned)

def youtube_search(
    query: str,
    max_results: int = 5,
    language: Optional[str] = None,
    order: Optional[str] = None,
    video_category_id: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Search YouTube.

    If YOUTUBE_API_KEY is present → returns structured results.
    If not → returns empty list and a fallback search URL.
    """
    q = str(query or "").strip()
    if not q:
        return [], None

    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        # No key: return a link the UI can open.
        url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(q)
        return [], url

    params = {
        "part": "snippet",
        "q": q,
        "type": "video",
        "maxResults": int(max(1, min(10, max_results))),
        "key": api_key,
        "safeSearch": "none",
        "order": (order or "relevance"),
    }
    if language:
        params["relevanceLanguage"] = language

    if video_category_id:
        params["videoCategoryId"] = str(video_category_id)

    r = requests.get(_YT_SEARCH_ENDPOINT, params=params, timeout=12)
    r.raise_for_status()
    data = r.json()
    items = data.get("items", []) or []
    return items, None

def to_external_results(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for it in items:
        vid = ((it.get("id") or {}).get("videoId") or "").strip()
        sn = it.get("snippet") or {}
        title = sn.get("title") or ""
        channel = sn.get("channelTitle") or ""
        artist, song_title = split_artist_title(title, channel)
        thumbs = (sn.get("thumbnails") or {})
        thumb_url = None
        # choose best available
        for key in ["high","medium","default"]:
            if key in thumbs and thumbs[key].get("url"):
                thumb_url = thumbs[key]["url"]
                break
        if not thumb_url and vid:
            thumb_url = f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"

        if not vid:
            continue
        out.append({
            "title": song_title,
            "artist": artist,
            "youtube_video_id": vid,
            "thumbnail_url": thumb_url,
        })
    return out
