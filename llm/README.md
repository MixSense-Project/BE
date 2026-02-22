# MixSense AI Search (Intent Parser + Catalog Engine)

Reference implementation aligned with your MixSense design:

- LLM is **only** an Intent Parser (Natural language → JSON filters/weights)
- The catalog engine is deterministic (filters + ranking over `track_id` rows)
- `genre/sub_genre/popularity/release_date` are **hard filters**
- `mood_tags/context_tags` are **soft ranking bonuses** (recommended for v1 stability)

## Run (local)
```bash
python -m mixsense_ai_search.cli \
  --catalog ../mixsense_outputs/prepared_catalog.pkl \
  --allowed ../mixsense_outputs/taxonomy_allowed_values.json \
  --query "비 오는 날 드라이브 R&B 잔잔한" \
  --k 10
```

## Swap in your GPT Intent Parser
Edit `mixsense_ai_search/intent_parser.py`:
- implement `LLMIntentParser.parse(query) -> LLMIntent` using your GPT wrapper
- keep the same JSON schema
- keep `validate_intent()` enabled (hallucination defense)


## Quick interactive run
```bash
python run_local.py
```
