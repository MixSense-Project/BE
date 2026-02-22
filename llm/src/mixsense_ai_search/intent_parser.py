from __future__ import annotations

import json
import os
from datetime import date, timedelta
from typing import Dict, Optional

from .schemas import LLMIntent, Filters, RankingWeights, DiversityRules

# ----------------------------
# Rule-based parser (fallback)
# ----------------------------
# Keep this so you can run locally without API credits / when debugging.
_KO_HINTS = {
    "비": {"context": "Drive"},
    "드라이브": {"context": "Drive"},
    "출근": {"context": "Commute"},
    "퇴근": {"context": "Commute"},
    "운동": {"context": "Workout"},
    "파티": {"context": "Party"},
    "새벽": {"context": "Night"},
    "밤": {"context": "Night"},
    "아침": {"context": "Morning"},
    "여름": {"context": "Summer"},
    "집중": {"context": "Focus"},
    "공부": {"context": "Study"},
    "잔잔": {"mood": "Calm"},
    "몽환": {"mood": "Dreamy"},
    "우울": {"mood": "Melancholic"},
    "신나": {"mood": "Energetic"},
    "행복": {"mood": "Happy"},
    "햇빛": {"mood": "Cheerful"},
    "맑은": {"mood": "Cheerful"},
    "로맨틱": {"mood": "Romantic"},
    "설날": {"context": "Holiday"},
    "명절": {"context": "Holiday"},
    "추석": {"context": "Holiday"},
}
_GENRE_KO = {
    "힙합": "Hip Hop",
    "알앤비": "R&B",
    "r&b": "R&B",
    "rnb": "R&B",
    "팝": "Pop",
    "재즈": "Jazz",
    "락": "Rock",
    "인디": "Indie",
    "로파이": "Lofi",
    "일렉": "Electronic",
    "edm": "Electronic",
    "케이팝": "K-Pop",
    "kpop": "K-Pop",
}


def rule_based_parse(query: str) -> LLMIntent:
    q = query.strip()
    f = Filters()

    q_low = q.lower()
    for k, v in _GENRE_KO.items():
        if k in q_low:
            f.genre_in.append(v)

    for k, info in _KO_HINTS.items():
        if k in q:
            if "mood" in info:
                f.mood_tags_in.append(info["mood"])
            if "context" in info:
                f.context_tags_in.append(info["context"])

    # popularity heuristic (0..100)
    if any(x in q for x in ["너무 유명", "메이저 말고", "덜 유명", "숨은"]):
        f.popularity_max = 60
    if any(x in q for x in ["히트곡", "유명한", "메이저", "핫한"]):
        f.popularity_min = 60

    today = date.today()
    if any(x in q for x in ["요즘", "최근", "최신", "신곡", "새로 나온"]):
        f.release_date_from = (today - timedelta(days=365 * 2)).isoformat()
        f.release_date_to = today.isoformat()

    # weights are ignored by default in the engine unless MIXSENSE_LLM_WEIGHTS=1,
    # but we still provide a reasonable default here.
    w = RankingWeights(
        w_artist_pref=0.35,
        w_popularity=0.40,
        w_recency=0.25,
        w_mood_match=0.10,
        w_context_match=0.10,
    )

    return LLMIntent(
        intent_type="SEARCH",
        query_text=q,
        filters=f,
        ranking_weights=w,
        diversity_rules=DiversityRules(max_tracks_per_artist=1),
        confidence=0.20,
    )


# ----------------------------
# GPT-based Intent Parser
# ----------------------------
_SYSTEM_PROMPT = """\
You are MixSense AI Search Intent Parser.

Your job:
- Convert the user's natural-language query into a JSON that matches the provided schema (LLMIntent).
- You do NOT recommend tracks and you do NOT invent track titles/artists.

Hard constraints:
- Only use values that appear in the provided allowed lists for:
  - filters.genre_in
  - filters.sub_genre_in
  - filters.mood_tags_in
  - filters.context_tags_in
  If you are unsure, leave the list empty (do NOT invent new tags).
- popularity_min/max are 0..100.
- release_date_from/to must be YYYY-MM-DD if set, else null.

Rules:
- Default intent_type to SEARCH for normal user queries.
- For v1 stability, set ranking_weights to all zeros (the engine uses server defaults).
- Keep diversity_rules.max_tracks_per_artist=1 unless user explicitly wants "한 아티스트만 쭉".
- Set confidence in 0..1. If query is vague, set <= 0.3.
"""

# Small bilingual hint map to avoid "dropped tags" (because of taxonomy validation).
# This does NOT force the model, it just nudges it toward existing tags.
_KO_TO_TAG_HINTS = """\
Korean hint → allowed tag examples:
- "설날", "명절", "추석" → context_tags_in: ["Holiday","Family"], mood_tags_in: ["Comforting","Calm"]
- "햇빛 좋은 날", "맑은 날", "기분 좋은 날" → mood_tags_in: ["Happy","Cheerful","Carefree"], context_tags_in: ["Morning","Summer","Feel-good"]
- "비 오는 날" → mood_tags_in: ["Calm","Chill","Melancholic"], context_tags_in: ["Drive","Commute","Cafe"]
- "새벽", "밤" → context_tags_in: ["Night"]
- "카페" → context_tags_in: ["Cafe"]
- "공부", "집중" → context_tags_in: ["Study","Focus"]
- "운동" → context_tags_in: ["Workout"], mood_tags_in: ["Energetic","Motivated"] (if available)
"""


class LLMIntentParser:
    """
    Drop-in replacement:
    - mode="gpt": uses OpenAI API (Responses.parse with pydantic schema)
    - mode="rule": uses rule_based_parse()
    """

    def __init__(
        self,
        allowed_values: Dict[str, list],
        today_iso: Optional[str] = None,
        mode: str = "gpt",
        model: Optional[str] = None,
    ):
        self.allowed_values = allowed_values
        self.today_iso = today_iso or date.today().isoformat()
        self.mode = (mode or "gpt").lower()
        self.model = model or os.environ.get("MIXSENSE_OPENAI_MODEL", "gpt-4o-mini")

        # Lazy imports so rule-based mode works without OpenAI deps.
        self._client = None
        if self.mode == "gpt":
            # Support .env file out-of-the-box (recommended in OpenAI docs).
            try:
                from dotenv import load_dotenv  # type: ignore
                load_dotenv()
            except Exception:
                pass

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. "
                    "Create an OpenAI API key and set it as an environment variable "
                    "(or put it in a .env file)."
                )
            try:
                from openai import OpenAI  # type: ignore
                self._client = OpenAI()
            except Exception as e:
                raise RuntimeError(
                    "Failed to import OpenAI SDK. Did you install dependencies? "
                    "Run: pip install -r requirements.txt"
                ) from e

    def parse(self, query: str) -> LLMIntent:
        if self.mode != "gpt":
            return rule_based_parse(query)

        assert self._client is not None, "OpenAI client not initialized"

        # Keep prompt payload deterministic & auditable.
        payload = {
            "query_text": query,
            "today": self.today_iso,
            "allowed_values": self.allowed_values,
            "hints": _KO_TO_TAG_HINTS,
            "instructions": (
                "Return a single LLMIntent object.\n"
                "Set ranking_weights fields all to 0.0 for v1.\n"
                "If you can't map something to allowed tags, leave those lists empty.\n"
            ),
        }

        # Prefer Structured Outputs parsing via Chat Completions parse helper (Pydantic).
        # This matches OpenAI docs examples.
        # Fallback order:
        #   1) client.chat.completions.parse(..., response_format=PydanticModel)
        #   2) client.responses.create(...) + json.loads
        #   3) rule-based fallback
        try:
            completion = self._client.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                response_format=LLMIntent,
            )
            intent: LLMIntent = completion.choices[0].message.parsed  # type: ignore
            if not intent.query_text:
                intent.query_text = query
            return intent
        except AttributeError:
            # SDK without chat.completions.parse helper → try Responses API create as a best-effort fallback
            r = self._client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                text={"format": {"type": "json_object"}},
            )
            text = getattr(r, "output_text", "") or ""
            try:
                data = json.loads(text)
                return LLMIntent(**data)
            except Exception:
                return rule_based_parse(query)
        except Exception:
            return rule_based_parse(query)
