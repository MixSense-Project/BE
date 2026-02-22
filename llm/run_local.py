"""
Interactive demo runner (local).

Usage:
  python run_local.py

Requirements:
  - Python 3.9+
  - pip install -r requirements.txt
  - set OPENAI_API_KEY (or create .env)

This script loads:
  ./mixsense_outputs/prepared_catalog.pkl
  ./mixsense_outputs/taxonomy_allowed_values.json
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from mixsense_ai_search.service import load_catalog, ai_search, looks_like_song_identification_query
from mixsense_ai_search.intent_parser import LLMIntentParser

CATALOG_PATH = ROOT / "mixsense_outputs" / "prepared_catalog.pkl"
ALLOWED_PATH = ROOT / "mixsense_outputs" / "taxonomy_allowed_values.json"
ALIASES_PATH = ROOT / "mixsense_outputs" / "artist_aliases_ko.json"

# --- Result size / follow-up helpers ----------------------------------------
# v1 design: LLM does NOT decide "how many results".
# But for a chat-like demo, we can interpret simple user phrases like "1곡만".
_KOR_NUM = {
    "한": 1, "하나": 1, "첫": 1,
    "두": 2, "둘": 2,
    "세": 3, "셋": 3,
    "네": 4, "넷": 4,
    "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10,
}

def extract_requested_k(query: str, default: int = 5, min_k: int = 1, max_k: int = 20) -> int:
    q = str(query)

    # 1) Digits: "1곡", "2개"
    m = re.search(r"(\d+)\s*(곡|개)", q)
    if m:
        k = int(m.group(1))
        return max(min_k, min(max_k, k))

    # 2) Korean words: "한 곡", "두 개"
    for word, num in _KOR_NUM.items():
        if re.search(rf"{re.escape(word)}\s*(곡|개)", q):
            return max(min_k, min(max_k, num))

    # 3) Common shortcuts: "딱 하나", "하나만"
    if re.search(r"(딱\s*)?(하나|한)\s*만", q) or ("딱 하나" in q):
        return 1

    return default

def is_control_only_request(query: str) -> bool:
    """True if the query is basically only about result count (e.g. '딱 한 곡만 추천해줘')."""
    q = re.sub(r"\s+", "", str(query))
    # remove common polite/request phrases
    for token in [
        "추천해줘","추천해","추천","골라줘","골라","선택해줘","선택","말해줘","보여줘","줘",
        "딱","그냥","오직","단","만","한곡","한개","1곡","1개","곡","개",
        "하나","한","두","둘","세","셋","네","넷","다섯","여섯","일곱","여덟","아홉","열",
        "top","TOP",
    ]:
        q = q.replace(token, "")
    q = re.sub(r"\d+", "", q)
    return len(q) == 0
# ---------------------------------------------------------------------------


def pretty_print(resp, k=10):
    print(f"\nmode={resp.mode} conf_recalc={resp.confidence_recalc:.2f} candidates={resp.n_candidates}")
    print("used_filters =", resp.used_intent.filters.model_dump())
    if resp.mode == "clarify":
        print("QUESTION:", resp.clarification_question)
        return

    if resp.mode == "external":
        if resp.external_results:
            for i, r in enumerate(resp.external_results[:k], 1):
                yt_url = f"https://www.youtube.com/watch?v={r.youtube_video_id}"
                print(f"{i:02d}. {r.title} — {r.artist}")
                print(f"    {yt_url}")
        elif resp.external_search_url:
            print("Open YouTube search:", resp.external_search_url)
        return

    for i, r in enumerate(resp.results[:k], 1):
        yt_url = f"https://www.youtube.com/watch?v={r.youtube_video_id}"
        print(f"{i:02d}. {r.title} — {r.artist}")
        print(f"    {yt_url}")



def main():
    if not CATALOG_PATH.exists():
        raise SystemExit(f"Missing catalog file: {CATALOG_PATH}")
    if not ALLOWED_PATH.exists():
        raise SystemExit(f"Missing allowed values file: {ALLOWED_PATH}")

    allowed = json.loads(ALLOWED_PATH.read_text(encoding="utf-8"))
    catalog_df = load_catalog(str(CATALOG_PATH))

    mode = "gpt" if os.environ.get("OPENAI_API_KEY") else "rule"
    model = os.environ.get("MIXSENSE_OPENAI_MODEL", "gpt-4o-mini")

    try:
        parser = LLMIntentParser(allowed_values=allowed, mode=mode, model=model)
    except Exception as e:
        print("⚠️ GPT parser not available:", str(e))
        print("→ Falling back to rule-based parser.")
        parser = LLMIntentParser(allowed_values=allowed, mode="rule")
        mode = "rule"

    print("\n=== MixSense AI Search (local) ===")
    print(f"- parser mode: {mode} (set OPENAI_API_KEY to use gpt)")
    print(f"- model: {model}")
    print("- Example queries:")
    print('  1) "오늘 같이 햇빛 좋은 날 듣기 좋은 노래 추천해줘"')
    print('  2) "비 오는 날 드라이브 R&B 잔잔한"')
    print('  3) "운동할 때 신나는 힙합"')
    print("Type empty line to quit.\n")

    last_resp = None

    while True:
        q = input("Query> ").strip()
        if not q:
            break

        # 1) Interpret "how many" from the query (demo convenience)
        default_k = 3 if looks_like_song_identification_query(q) else 5
        k_req = extract_requested_k(q, default=default_k, min_k=1, max_k=20)
        if looks_like_song_identification_query(q):
            k_req = min(k_req, 3)

        # 2) If user only asks for result count (e.g. "딱 한 곡만") and we have a previous result,
        #    reuse the previous result instead of re-parsing.
        if is_control_only_request(q) and last_resp is not None and last_resp.mode != "clarify":
            pretty_print(last_resp, k=k_req)
            continue

        resp = ai_search(q, catalog_df, allowed, llm_parser=parser, k=k_req, artist_aliases=artist_aliases)

        # keep last non-clarify response for follow-up controls
        if resp.mode != "clarify":
            last_resp = resp

        pretty_print(resp, k=k_req)

    print("bye!")


if __name__ == "__main__":
    main()
