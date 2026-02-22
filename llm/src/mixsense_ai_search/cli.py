from __future__ import annotations

import argparse
import json
from pathlib import Path

from .service import load_catalog, ai_search
from .intent_parser import LLMIntentParser


def main():
    ap = argparse.ArgumentParser(description="MixSense AI Search (Intent Parser + Catalog Engine)")
    ap.add_argument("--catalog", required=True, help="prepared_catalog.pkl or prepared_catalog.csv")
    ap.add_argument("--allowed", required=True, help="taxonomy_allowed_values.json")
    ap.add_argument("--query", required=True, help="user query in natural language")
    ap.add_argument("--k", type=int, default=5, help="top-k results")

    ap.add_argument("--parser", choices=["gpt", "rule"], default="gpt", help="intent parser backend")
    ap.add_argument("--model", default=None, help="OpenAI model name (only when --parser gpt)")
    ap.add_argument("--seed_track_id", default=None, help="optional track_id for SIMILAR mode (UI-driven)")

    args = ap.parse_args()

    catalog_df = load_catalog(args.catalog)
    allowed_values = json.loads(Path(args.allowed).read_text(encoding="utf-8"))

    try:
        parser = LLMIntentParser(
            allowed_values=allowed_values,
            mode=args.parser,
            model=args.model,
        )
    except Exception as e:
        print("❌ Failed to initialize parser:", str(e))
        print("Tip: if you haven't set OPENAI_API_KEY, either:")
        print("  - set OPENAI_API_KEY (recommended), then retry with --parser gpt")
        print("  - or run with --parser rule (no API needed)")
        raise SystemExit(1)

    resp = ai_search(
        args.query,
        catalog_df,
        allowed_values,
        llm_parser=parser,
        seed_track_id=args.seed_track_id,
        k=args.k,
    )

    print(f"mode={resp.mode} conf_recalc={resp.confidence_recalc:.2f} candidates={resp.n_candidates}")
    print("used_intent.filters =", resp.used_intent.filters.model_dump())

    if resp.mode == "clarify":
        print("QUESTION:", resp.clarification_question)
        return

    for r in resp.results:
        print(
            f"- {r.title} / {r.artist} | {r.genre}/{r.sub_genre} | "
            f"pop={r.popularity:.0f} | {r.release_date} | score={r.score:.4f} | yt={r.youtube_video_id}"
        )
