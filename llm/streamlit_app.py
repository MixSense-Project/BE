from __future__ import annotations

import json
import os
import re
from pathlib import Path

import streamlit as st

# Ensure local package import works when running `streamlit run streamlit_app.py`
ROOT = Path(__file__).resolve().parent
import sys
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

# --- Result size helpers (same policy as run_local.py) -----------------------
_KOR_NUM = {
    "한": 1, "하나": 1, "첫": 1,
    "두": 2, "둘": 2,
    "세": 3, "셋": 3,
    "네": 4, "넷": 4,
    "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10,
}

def extract_requested_k(query: str, default: int = 5, min_k: int = 1, max_k: int = 20) -> int:
    q = str(query)

    m = re.search(r"(\d+)\s*(곡|개)", q)
    if m:
        k = int(m.group(1))
        return max(min_k, min(max_k, k))

    for word, num in _KOR_NUM.items():
        if re.search(rf"{re.escape(word)}\s*(곡|개)", q):
            return max(min_k, min(max_k, num))

    if re.search(r"(딱\s*)?(하나|한)\s*만", q) or ("딱 하나" in q):
        return 1

    return default

def is_control_only_request(query: str) -> bool:
    q = re.sub(r"\s+", "", str(query))
    for token in [
        "추천해줘","추천해","추천","골라줘","골라","선택해줘","선택","말해줘","보여줘","줘",
        "딱","그냥","오직","단","만","한곡","한개","1곡","1개","곡","개",
        "하나","한","두","둘","세","셋","네","넷","다섯","여섯","일곱","여덟","아홉","열",
        "top","TOP",
    ]:
        q = q.replace(token, "")
    q = re.sub(r"\d+", "", q)
    return len(q) == 0
# -----------------------------------------------------------------------------


def card_html(title: str, artist: str, thumb_url: str, yt_url: str) -> str:
    def esc(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    title_e = esc(title)
    artist_e = esc(artist)
    thumb_e = esc(thumb_url)
    yt_e = esc(yt_url)

    return f"""<a href=\"{yt_e}\" target=\"_blank\" style=\"text-decoration:none;color:inherit;\">
  <img src=\"{thumb_e}\" style=\"width:100%;border-radius:14px;\">
  <div style=\"margin-top:8px;font-weight:650;font-size:16px;line-height:1.25;\">{title_e}</div>
  <div style=\"opacity:0.72;font-size:14px;\">{artist_e}</div>
</a>
"""


@st.cache_data(show_spinner=False)
def _load_assets():
    if not CATALOG_PATH.exists():
        raise RuntimeError(f"Missing catalog file: {CATALOG_PATH}")
    if not ALLOWED_PATH.exists():
        raise RuntimeError(f"Missing allowed values file: {ALLOWED_PATH}")

    allowed = json.loads(ALLOWED_PATH.read_text(encoding="utf-8"))
    catalog_df = load_catalog(str(CATALOG_PATH))

    artist_aliases = {}
    if ALIASES_PATH.exists():
        try:
            artist_aliases = json.loads(ALIASES_PATH.read_text(encoding="utf-8"))
        except Exception:
            artist_aliases = {}

    return catalog_df, allowed, artist_aliases


def main():
    st.set_page_config(page_title="MixSense AI Search", page_icon="🎵", layout="wide")
    st.title("🎵 MixSense AI Search (local)")
    st.caption("카드(썸네일/제목/아티스트)만 보여주고, 클릭하면 YouTube로 이동합니다.")

    catalog_df, allowed, artist_aliases = _load_assets()

    mode = "gpt" if os.environ.get("OPENAI_API_KEY") else "rule"
    model = os.environ.get("MIXSENSE_OPENAI_MODEL", "gpt-4o-mini")
    yt_key = bool(os.environ.get("YOUTUBE_API_KEY"))

    # parser init
    try:
        parser = LLMIntentParser(allowed_values=allowed, mode=mode, model=model)
    except Exception:
        parser = LLMIntentParser(allowed_values=allowed, mode="rule", model=model)
        mode = "rule"

    with st.sidebar:
        st.subheader("상태")
        st.write(f"- parser: **{mode}**")
        st.write(f"- model: **{model}**")
        st.write(f"- YouTube search API: **{'ON' if yt_key else 'OFF'}**")
        st.caption("※ '이 노래 뭐야?' 같은 식별 질의는 YouTube API 키가 있으면 더 잘 동작합니다.")
        st.divider()
        st.subheader("예시")
        st.code("오늘 같이 햇빛 좋은 날 듣기 좋은 노래 3곡 추천해줘", language="text")
        st.code("아이유 이번 새로운 앨범 노래들 알려줘", language="text")
        st.code("이번 최신 외국 힙합 노래 10곡 알려줘", language="text")
        st.code("노래 시작할 때 기계음 여자가 아아아... 다메다메 하는 노래 뭐야?", language="text")

    if "last_payload" not in st.session_state:
        st.session_state["last_payload"] = None

    q = st.text_input("Query", placeholder="예: 오늘 같이 햇빛 좋은 날 듣기 좋은 노래 추천해줘")

    col1, col2 = st.columns([1, 5])
    with col1:
        run = st.button("Search", use_container_width=True)

    if run and q.strip():
        default_k = 3 if looks_like_song_identification_query(q) else 5
        k_req = extract_requested_k(q, default=default_k, min_k=1, max_k=20)
        if looks_like_song_identification_query(q):
            k_req = min(k_req, 3)

        # If user only asks for count, reuse last results (demo convenience)
        last = st.session_state.get("last_payload")
        if is_control_only_request(q) and last is not None:
            payload = last
            payload["k"] = min(k_req, 3) if (payload.get("mode") == "external") else k_req
        else:
            resp = ai_search(
                q,
                catalog_df,
                allowed,
                llm_parser=parser,
                k=k_req,
                artist_aliases=artist_aliases,
                enable_external_youtube_search=True,
            )
            payload = {
                "mode": resp.mode,
                "k": k_req,
                "clarify": resp.clarification_question,
                "results": [
                    {
                        "title": r.title,
                        "artist": r.artist,
                        "youtube_video_id": r.youtube_video_id,
                        "thumbnail_url": (r.track_image_url or r.thumbnail_url or f"https://i.ytimg.com/vi/{r.youtube_video_id}/hqdefault.jpg"),
                    }
                    for r in resp.results
                ],
                "external_results": [
                    {
                        "title": r.title,
                        "artist": r.artist,
                        "youtube_video_id": r.youtube_video_id,
                        "thumbnail_url": (r.thumbnail_url or f"https://i.ytimg.com/vi/{r.youtube_video_id}/hqdefault.jpg"),
                    }
                    for r in (resp.external_results or [])
                ],
                "external_search_url": resp.external_search_url,
            }
            st.session_state["last_payload"] = payload

        # Render
        st.subheader("결과")

        if payload["mode"] == "clarify":
            st.warning(payload["clarify"] or "조금만 더 구체화해줘.")
            return

        items = payload["results"][: payload["k"]]
        ext_items = payload.get("external_results", [])[: payload["k"]]

        if payload["mode"] == "external":
            if ext_items:
                items = ext_items
            else:
                # no api key → provide search link
                url = payload.get("external_search_url")
                if url:
                    st.info("카탈로그로는 매칭이 어려워서 YouTube 검색으로 넘깁니다.")
                    st.link_button("YouTube에서 검색 열기", url)
                else:
                    st.error("외부 검색 결과가 없습니다.")
                return

        if not items:
            st.info("결과가 없습니다.")
            return

        # Show cards
        ncols = 5 if payload["k"] >= 10 else 3
        cols = st.columns(ncols)
        for i, it in enumerate(items):
            yt = f"https://www.youtube.com/watch?v={it['youtube_video_id']}"
            thumb = it["thumbnail_url"]
            with cols[i % ncols]:
                st.markdown(card_html(it["title"], it["artist"], thumb, yt), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
