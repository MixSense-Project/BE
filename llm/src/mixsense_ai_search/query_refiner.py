from __future__ import annotations

import json
import os
import re
from typing import Optional

from pydantic import BaseModel, Field


class RefinedYouTubeQuery(BaseModel):
    """A minimal plan for YouTube search (NOT a song recommendation)."""
    query: str = Field(..., description="A concise YouTube search query derived from the user's input.")
    language: Optional[str] = Field(
        default=None,
        description="Optional relevanceLanguage hint for YouTube search (e.g., 'ko', 'en', 'ja').",
    )


_SYSTEM = """\
You generate YouTube search queries to HELP IDENTIFY a song.

Input: user text that may contain partial lyrics, onomatopoeia, or a description.
Output: JSON with fields:
- query: a concise search query (<= 15 words if possible)
- language: optional ISO language hint ('ko', 'en', 'ja') if obvious

Rules:
- Do NOT guess the song title/artist.
- Do NOT invent lyrics. You may reuse ONLY the exact words the user provided.
- Remove filler phrases like '이 노래 뭐야', '찾아줘', '제목이 뭐야', etc.
- Prefer keeping distinctive phrases (including the user's lyric snippet) + 1-2 hints like 'lyrics' or '가사' if useful.
Return JSON only.
"""


def _detect_language_hint(text: str) -> Optional[str]:
    s = str(text or "")
    if re.search(r"[\uac00-\ud7a3]", s):
        return "ko"
    if re.search(r"[\u3040-\u30ff]", s):  # Hiragana/Katakana
        return "ja"
    return None


def refine_youtube_query_with_gpt(user_text: str, model: Optional[str] = None) -> Optional[RefinedYouTubeQuery]:
    """Best-effort GPT-based refinement for YouTube search.

    Returns None if OPENAI_API_KEY is not set or if the OpenAI SDK is not available.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None

    client = OpenAI()
    m = model or os.environ.get("MIXSENSE_OPENAI_MODEL", "gpt-4o-mini")

    payload = {
        "user_text": user_text,
        "language_hint": _detect_language_hint(user_text),
    }

    # Prefer parse helper if available
    try:
        completion = client.chat.completions.parse(
            model=m,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format=RefinedYouTubeQuery,
        )
        plan: RefinedYouTubeQuery = completion.choices[0].message.parsed  # type: ignore
        if plan.language is None:
            plan.language = payload["language_hint"]
        plan.query = (plan.query or "").strip()
        if not plan.query:
            return None
        return plan
    except Exception:
        # Fallback: Responses API returning a json_object
        try:
            r = client.responses.create(
                model=m,
                input=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                text={"format": {"type": "json_object"}},
            )
            text = getattr(r, "output_text", "") or ""
            data = json.loads(text)
            plan = RefinedYouTubeQuery(**data)
            if plan.language is None:
                plan.language = payload["language_hint"]
            plan.query = (plan.query or "").strip()
            if not plan.query:
                return None
            return plan
        except Exception:
            return None
