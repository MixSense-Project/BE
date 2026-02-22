import os
import sys
import json
import uuid
import tempfile
import shutil
import requests
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks, HTTPException, Header, Depends, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any
from dotenv import load_dotenv
from supabase import create_client, Client

# 기존 파이프라인 모듈
from track_data_pipeline import run_pipeline, get_spotify_token

load_dotenv()
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent
LLM_ROOT = BASE_DIR / "llm"
AI_RECOMMEND_ROOT = BASE_DIR / "ai_recommend"
AI_MIXING_ROOT = BASE_DIR / "ai_mixing"

# 서버 실행 시 1회만 패키지 경로를 추가합니다.
if str(LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_ROOT))
if str(LLM_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(LLM_ROOT / "src"))
if str(AI_RECOMMEND_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_RECOMMEND_ROOT))
if str(AI_MIXING_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_MIXING_ROOT))
if str(AI_MIXING_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(AI_MIXING_ROOT / "src"))

# Supabase 클라이언트
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----- LLM (AI Search) lazy load -----
_llm_catalog_df = None
_llm_allowed_values = None
_llm_artist_aliases = None
_llm_parser = None
_llm_ai_search_fn = None

def _ensure_llm():
    global _llm_catalog_df, _llm_allowed_values, _llm_artist_aliases, _llm_parser, _llm_ai_search_fn
    if _llm_catalog_df is not None:
        return
    if not (LLM_ROOT / "src").exists():
        raise HTTPException(status_code=503, detail="LLM module not found")
    try:
        from mixsense_ai_search.service import load_catalog, ai_search
        from mixsense_ai_search.intent_parser import LLMIntentParser
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"LLM import error: {str(e)}")
    _llm_ai_search_fn = ai_search

    catalog_path = os.environ.get("LLM_CATALOG_PATH") or str(LLM_ROOT / "mixsense_outputs" / "prepared_catalog.pkl")
    allowed_path = os.environ.get("LLM_TAXONOMY_PATH") or str(LLM_ROOT / "mixsense_outputs" / "taxonomy_allowed_values.json")
    aliases_path = os.environ.get("LLM_ALIASES_PATH") or str(LLM_ROOT / "mixsense_outputs" / "artist_aliases_ko.json")

    if not Path(catalog_path).exists():
        raise HTTPException(status_code=503, detail=f"Catalog file not found: {catalog_path}")
    if not Path(allowed_path).exists():
        raise HTTPException(status_code=503, detail=f"Taxonomy file not found: {allowed_path}")

    _llm_catalog_df = load_catalog(catalog_path)
    with open(allowed_path, "r", encoding="utf-8") as f:
        _llm_allowed_values = json.load(f)

    _llm_artist_aliases = None
    if Path(aliases_path).exists():
        with open(aliases_path, "r", encoding="utf-8") as f:
            _llm_artist_aliases = json.load(f)

    mode = "gpt" if os.environ.get("OPENAI_API_KEY") else "rule"
    model = os.environ.get("MIXSENSE_OPENAI_MODEL", "gpt-4o-mini")
    try:
        _llm_parser = LLMIntentParser(allowed_values=_llm_allowed_values, mode=mode, model=model)
    except Exception:
        _llm_parser = LLMIntentParser(allowed_values=_llm_allowed_values, mode="rule")

app = FastAPI(title="MixSense API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. Pydantic Models
# ==========================================
class AuthRequest(BaseModel):
    email: str
    password: str
    agegroup: Optional[str] = None
    gender: Optional[str] = None

class ProfileCreateReq(BaseModel):
    user_id: Optional[str] = None
    name: str
    agegroup: Optional[str] = None
    gender: Optional[str] = None

class ProfileUpdateReq(BaseModel):
    name: Optional[str] = None
    agegroup: Optional[str] = None
    gender: Optional[str] = None

class ProfileDeleteByNameReq(BaseModel):
    user_id: str
    profile_name: str

class ProfileEditByNameReq(BaseModel):
    user_id: str
    original_name: str
    updated: dict

class MyListReq(BaseModel):
    profile_id: str
    content_id: str

class TrackRequest(BaseModel):
    title: str
    artist: str

class PlayLogRequest(BaseModel):
    track_id: str
    ms_played: int

class AISearchRequest(BaseModel):
    query: str
    k: int = 5
    seed_track_id: Optional[str] = None

class PreferenceUpdateReq(BaseModel):
    favorite_genres: List[str]
    favorite_artists: List[str]

class PlaylistCreateReq(BaseModel):
    title: str

class PlaylistUpdateReq(BaseModel):
    title: str

class PlaylistTrackReq(BaseModel):
    track_id: str

class SearchHistoryAddReq(BaseModel):
    keyword: str

# ==========================================
# 2. Auth Dependency
# ==========================================
def verify_user(authorization: str = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = authorization.split(" ")[1]
    user_response = supabase.auth.get_user(token)
    if not user_response or not user_response.user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user_response.user.id

# ==========================================
# 3. Auth APIs
# ==========================================
@app.post("/auth/signup")
def signup(req: AuthRequest):
    try:
        res = supabase.auth.sign_up({"email": req.email, "password": req.password})
        if not res.user:
            raise HTTPException(status_code=400, detail="Signup failed (user not created)")
        return {"status": "success", "user": res.user}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/login")
def login(req: AuthRequest):
    try:
        res = supabase.auth.sign_in_with_password({"email": req.email, "password": req.password})
        if not res.session:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return {"status": "success", "session": res.session}
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

@app.delete("/auth/unsubscribe")
def unsubscribe_user(user_id: str = Depends(verify_user)):
    """회원 탈퇴: 프로필 삭제 후 Auth 계정 삭제. Auth 삭제는 SERVICE_ROLE_KEY 필요."""
    supabase.table("profiles").delete().eq("user_id", user_id).execute()
    try:
        supabase.auth.admin.delete_user(user_id)
        return {"status": "unsubscribed", "message": "User and profile successfully deleted."}
    except Exception as e:
        print(f"Auth user deletion failed (Service Role Key may be required): {e}")
        return {
            "status": "partial_success",
            "message": "Profile deleted, but admin privileges are required to remove the Auth account."
        }

# ==========================================
# 4. Profile APIs
# ==========================================
@app.get("/profile")
def get_profiles_list(user_id: str = Depends(verify_user)):
    response = supabase.table("profiles").select("*").eq("user_id", user_id).execute()
    return {"profiles": response.data}

@app.post("/profile")
def create_profile(req: ProfileCreateReq, user_id: str = Depends(verify_user)):
    # 본인 계정에만 프로필 생성 (req.user_id 무시)
    response = supabase.table("profiles").insert({
        "user_id": user_id,
        "display_name": req.name,
        "agegroup": req.agegroup,
        "gender": req.gender
    }).execute()
    return {"status": "success", "profile": response.data}

@app.put("/profile/{profile_id}")
def update_profile_full(profile_id: str, req: ProfileUpdateReq, user_id: str = Depends(verify_user)):
    response = supabase.table("profiles").update({
        "display_name": req.name,
        "agegroup": req.agegroup,
        "gender": req.gender
    }).eq("id", profile_id).eq("user_id", user_id).execute()
    return {"status": "success", "profile": response.data}

@app.patch("/profile/{profile_id}")
def update_profile_partial(profile_id: str, req: ProfileUpdateReq, user_id: str = Depends(verify_user)):
    update_data = {k: v for k, v in req.model_dump().items() if v is not None}
    if update_data.get("name"):
        update_data["display_name"] = update_data.pop("name")
        
    response = supabase.table("profiles").update(update_data).eq("id", profile_id).eq("user_id", user_id).execute()
    return {"status": "success", "profile": response.data}

@app.delete("/profile/{profile_id}")
def delete_profile(profile_id: str, user_id: str = Depends(verify_user)):
    supabase.table("profiles").delete().eq("id", profile_id).eq("user_id", user_id).execute()
    return {"status": "deleted"}

@app.get("/profile/by_user")
def get_profiles_by_user(user_id: str = Depends(verify_user)):
    """본인 프로필 목록만 조회 (인증 필수)."""
    response = supabase.table("profiles").select("*").eq("user_id", user_id).execute()
    return {"profiles": response.data}

@app.post("/profile/add_profile")
def add_profile_custom(req: ProfileCreateReq, user_id: str = Depends(verify_user)):
    """본인 계정에 프로필 추가. req.user_id는 무시하고 토큰의 user_id 사용."""
    response = supabase.table("profiles").insert({
        "user_id": user_id,
        "display_name": req.name,
        "agegroup": req.agegroup,
        "gender": req.gender
    }).execute()
    return {"status": "success", "profile": response.data}

@app.post("/profile/delete_profile")
def delete_profile_by_name(req: ProfileDeleteByNameReq, auth_user_id: str = Depends(verify_user)):
    if req.user_id != auth_user_id:
        raise HTTPException(status_code=403, detail="Can only delete your own profiles")
    supabase.table("profiles").delete().eq("user_id", req.user_id).eq("display_name", req.profile_name).execute()
    return {"status": "deleted"}

@app.patch("/profile/edit_profile")
def edit_profile_by_name(req: ProfileEditByNameReq, auth_user_id: str = Depends(verify_user)):
    if req.user_id != auth_user_id:
        raise HTTPException(status_code=403, detail="Can only edit your own profiles")
    response = supabase.table("profiles").update(req.updated).eq("user_id", req.user_id).eq("display_name", req.original_name).execute()
    return {"status": "updated", "profile": response.data}

@app.patch("/api/profile/preferences")
def update_user_preferences(req: PreferenceUpdateReq, user_id: str = Depends(verify_user)):
    """온보딩 과정에서 선택한 선호 장르와 아티스트를 profile에 저장합니다."""
    update_data = {
        "favorite_genres": req.favorite_genres,
        "favorite_artists": req.favorite_artists
    }
    response = supabase.table("profiles").update(update_data).eq("user_id", user_id).execute()
    return {"status": "success", "profile": response.data}

# ==========================================
# 5. User MyList & Like APIs
# ==========================================

def verify_profile_ownership(profile_id: str, user_id: str):
    """해당 프로필이 현재 로그인한 유저의 소유인지 검증합니다."""
    res = supabase.table("profiles").select("id").eq("id", profile_id).eq("user_id", user_id).execute()
    if not res.data:
        raise HTTPException(status_code=403, detail="You do not own this profile")

@app.post("/user/mylist")
def add_to_mylist(req: MyListReq, user_id: str = Depends(verify_user)):
    verify_profile_ownership(req.profile_id, user_id)
    response = supabase.table("likes").insert({"user_id": req.profile_id, "track_id": req.content_id}).execute()
    return {"status": "added", "data": response.data}

@app.get("/user/mylist/{profile_id}")
def get_mylist(profile_id: str, user_id: str = Depends(verify_user)):
    verify_profile_ownership(profile_id, user_id)
    response = supabase.table("likes").select("*, track(*)").eq("user_id", profile_id).execute()
    return {"mylist": response.data}

@app.delete("/user/mylist/{profile_id}/{content_id}")
def delete_from_mylist(profile_id: str, content_id: str, user_id: str = Depends(verify_user)):
    verify_profile_ownership(profile_id, user_id)
    supabase.table("likes").delete().eq("user_id", profile_id).eq("track_id", content_id).execute()
    return {"status": "deleted"}

@app.post("/user/toggle_like")
def toggle_like_custom(req: MyListReq, user_id: str = Depends(verify_user)):
    verify_profile_ownership(req.profile_id, user_id)
    existing = supabase.table("likes").select("*").eq("user_id", req.profile_id).eq("track_id", req.content_id).execute()
    if existing.data:
        supabase.table("likes").delete().eq("user_id", req.profile_id).eq("track_id", req.content_id).execute()
        return {"status": "unliked"}
    supabase.table("likes").insert({"user_id": req.profile_id, "track_id": req.content_id}).execute()
    return {"status": "liked"}

# ==========================================
# 6. Play Logs
# ==========================================
@app.post("/api/logs/play")
def record_play_log(log: PlayLogRequest, user_id: str = Depends(verify_user)):
    supabase.table("play_logs").insert({
        "user_id": user_id,
        "track_id": log.track_id,
        "ms_played": log.ms_played
    }).execute()
    return {"status": "success"}

# ==========================================
# 7. Playlist CRUD APIs
# ==========================================
@app.post("/api/playlists")
def create_playlist(req: PlaylistCreateReq, user_id: str = Depends(verify_user)):
    """새로운 플레이리스트를 생성합니다."""
    response = supabase.table("playlist").insert({
        "user_id": user_id,
        "title": req.title
    }).execute()
    return {"status": "success", "playlist": response.data}

@app.get("/api/playlists")
def get_my_playlists(user_id: str = Depends(verify_user)):
    """현재 로그인한 유저의 플레이리스트 목록을 조회합니다."""
    response = supabase.table("playlist").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
    return {"playlists": response.data}

@app.patch("/api/playlists/{playlist_id}")
def update_playlist_title(playlist_id: str, req: PlaylistUpdateReq, user_id: str = Depends(verify_user)):
    """플레이리스트의 이름을 수정합니다."""
    response = supabase.table("playlist").update({"title": req.title}).eq("playlist_id", playlist_id).eq("user_id", user_id).execute()
    return {"status": "success", "playlist": response.data}

@app.delete("/api/playlists/{playlist_id}")
def delete_playlist(playlist_id: str, user_id: str = Depends(verify_user)):
    """플레이리스트를 삭제합니다."""
    supabase.table("playlist").delete().eq("playlist_id", playlist_id).eq("user_id", user_id).execute()
    return {"status": "deleted"}

# ==========================================
# 8. Playlist Tracks APIs
# ==========================================
@app.post("/api/playlists/{playlist_id}/tracks")
def add_track_to_playlist(playlist_id: str, req: PlaylistTrackReq, user_id: str = Depends(verify_user)):
    """특정 플레이리스트에 곡을 추가합니다."""
    # 플레이리스트 소유권 1차 검증
    playlist_check = supabase.table("playlist").select("playlist_id").eq("playlist_id", playlist_id).eq("user_id", user_id).execute()
    if not playlist_check.data:
        raise HTTPException(status_code=403, detail="Playlist ownership verification failed")
    # user_id 컬럼 제외 후 삽입
    response = supabase.table("playlist_tracks").insert({
        "playlist_id": playlist_id,
        "track_id": req.track_id
    }).execute()
    return {"status": "added", "data": response.data}

@app.get("/api/playlists/{playlist_id}/tracks")
def get_playlist_tracks(playlist_id: str, user_id: str = Depends(verify_user)):
    """특정 플레이리스트에 포함된 곡 목록과 메타데이터를 조회합니다."""
    response = supabase.table("playlist_tracks").select("*, track(*)").eq("playlist_id", playlist_id).order("added_at", desc=False).execute()
    return {"playlist_tracks": response.data}

@app.delete("/api/playlists/{playlist_id}/tracks/{track_id}")
def remove_track_from_playlist(playlist_id: str, track_id: str, user_id: str = Depends(verify_user)):
    """특정 플레이리스트에서 곡을 제외합니다."""
    # 플레이리스트 소유권 1차 검증
    playlist_check = supabase.table("playlist").select("playlist_id").eq("playlist_id", playlist_id).eq("user_id", user_id).execute()
    if not playlist_check.data:
        raise HTTPException(status_code=403, detail="Playlist ownership verification failed")
    # user_id 조건 제외 후 삭제 실행
    supabase.table("playlist_tracks").delete().eq("playlist_id", playlist_id).eq("track_id", track_id).execute()
    return {"status": "deleted"}

# ==========================================
# 9. Data Pipeline & Tracks
# ==========================================
def run_pipeline_background(tracks: List[TrackRequest]):
    df_data = [{"title": t.title, "artist": t.artist} for t in tracks]
    input_df = pd.DataFrame(df_data)
    try:
        run_pipeline(input_df, output_csv="api_accumulated_data.csv", num_workers=1)
    except Exception as e:
        print(f"Pipeline error: {e}")

@app.get("/api/search")
def search_spotify(query: str, limit: int = 5):
    limit = max(1, min(50, limit))
    token = get_spotify_token()
    search_url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"q": query, "type": "track", "limit": limit}
    
    response = requests.get(search_url, headers=headers, params=params)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Spotify API Error")
        
    data = response.json()
    items = data.get("tracks", {}).get("items", [])
    
    results = [{
        "track_id": item["id"],
        "title": item["name"],
        "artist": item["artists"][0]["name"] if item.get("artists") else "",
        "track_image_url": item.get("album", {}).get("images", [{}])[0].get("url")
    } for item in items]
    
    return {"results": results}

@app.post("/api/tracks/process")
def process_tracks(tracks: List[TrackRequest], background_tasks: BackgroundTasks):
    background_tasks.add_task(run_pipeline_background, tracks)
    return {"status": "processing"}

@app.get("/api/tracks")
def get_tracks(limit: int = 50):
    response = supabase.table("track").select("*").order("updated_at", desc=True).limit(limit).execute()
    return {"tracks": response.data}

@app.get("/api/artists")
def get_artists(
    genres: Optional[List[str]] = Query(None),
    keyword: Optional[str] = Query(None),
    index_char: Optional[str] = Query(None),
):
    """선택된 장르, 검색어, 초성을 기반으로 필터링 및 이름순 정렬된 아티스트 목록을 반환합니다."""
    query = supabase.table("artist").select("*")
    if genres:
        query = query.overlaps("genres", genres)
    if keyword:
        query = query.ilike("name", f"%{keyword}%")
    if index_char:
        query = query.eq("index_char", index_char)
    query = query.order("name", desc=False)
    response = query.execute()
    return {"artists": response.data}


@app.get("/api/tracks/{track_id}/lyrics")
def get_track_lyrics(track_id: str):
    """UI 에러 방지용 가사 더미 API입니다. (추후 실제 가사 데이터로 대체 예정)"""
    dummy_lyrics = "가사 준비 중입니다...\n\nWe couldn't turn around 'til we were upside down\nI'll be the bad guy now..."
    return {"track_id": track_id, "lyrics": dummy_lyrics}


@app.get("/api/trending")
def get_trending_tracks(limit: int = 50):
    """일일 Top 50 랭킹 트랙 목록을 반환합니다."""
    response = supabase.table("trending_tracks").select("*, track(*)").order("rank", desc=False).limit(limit).execute()
    return {"trending": response.data}


# ==========================================
# 10. AI Search (LLM)
# ==========================================
@app.post("/api/ai/search")
def api_ai_search(req: AISearchRequest):
    _ensure_llm()
    k = max(1, min(20, req.k))
    resp = _llm_ai_search_fn(
        req.query,
        _llm_catalog_df,
        _llm_allowed_values,
        llm_parser=_llm_parser,
        seed_track_id=req.seed_track_id,
        k=k,
        artist_aliases=_llm_artist_aliases,
        enable_external_youtube_search=True,
    )
    return {
        "mode": resp.mode,
        "confidence_recalc": getattr(resp, "confidence_recalc", 0.0),
        "clarification_question": getattr(resp, "clarification_question", None),
        "n_candidates": getattr(resp, "n_candidates", 0),
        "results": [r.model_dump() for r in resp.results],
        "external_results": [r.model_dump() for r in resp.external_results] if resp.external_results else [],
        "external_search_url": getattr(resp, "external_search_url", None),
    }

# ==========================================
# 11. AI Recommend (Home / Autoplay)
# ==========================================
def _build_meta_from_supabase() -> pd.DataFrame:
    response = supabase.table("track").select("track_id, title, artist, popularity, release_date, youtube_video_id").execute()
    rows = response.data or []
    if not rows:
        return pd.DataFrame(columns=["track_id", "artist_id", "artist", "title", "popularity", "release_date", "youtube_video_id", "artist_norm", "title_norm"])
    df = pd.DataFrame(rows)
    df["track_id"] = df["track_id"].astype(str)
    df["artist_id"] = df["artist"].fillna("").astype(str)
    df["artist"] = df["artist"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    if "youtube_video_id" not in df.columns:
        df["youtube_video_id"] = ""
    df["youtube_video_id"] = df["youtube_video_id"].fillna("").astype(str)

    from recommender_home_autoplay import normalize_text
    df["artist_norm"] = df["artist"].map(normalize_text)
    df["title_norm"] = df["title"].map(normalize_text)
    return df

def _build_user_seen_and_log(user_id: str):
    pl = supabase.table("play_logs").select("track_id, created_at, ms_played").eq("user_id", user_id).order("created_at", desc=False).limit(5000).execute()
    log_rows = pl.data or []
    likes = supabase.table("likes").select("track_id").eq("user_id", user_id).execute()
    like_ids = {r["track_id"] for r in (likes.data or [])}
    
    seen_ids = set()
    log_list = []
    for r in log_rows:
        tid = r.get("track_id")
        if tid:
            seen_ids.add(tid)
        ts = r.get("created_at")
        if ts and tid:
            log_list.append({"track_id": tid, "endTime": pd.to_datetime(ts), "msPlayed": int(r.get("ms_played") or 0)})
    for tid in like_ids:
        seen_ids.add(tid)
        
    log_df = pd.DataFrame(log_list) if log_list else pd.DataFrame(columns=["track_id", "endTime", "msPlayed"])
    return seen_ids, log_df

@app.get("/api/ai/recommend/home")
def api_ai_recommend_home(user_id: str = Depends(verify_user)):
    try:
        from recommender_home_autoplay import build_profile, build_home18
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"AI Recommend module not found: {e}")
        
    meta = _build_meta_from_supabase()
    if meta.empty:
        return {"home": [], "message": "No tracks in catalog."}
        
    seen_ids, _ = _build_user_seen_and_log(user_id)
    seen_df = meta[meta["track_id"].isin(seen_ids)].copy()
    if seen_df.empty:
        seen_df = meta.head(0)
        
    profile = build_profile(seen_df)
    home = build_home18(meta, seen_df, profile)
    return {"home": home}

@app.get("/api/ai/recommend/autoplay")
def api_ai_recommend_autoplay(current_track_id: str, k: int = 10, user_id: str = Depends(verify_user)):
    try:
        from recommender_home_autoplay import (
            build_profile, sessions_from_log, time_split_sessions, train_markov, recommend_autoplay,
        )
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"AI Recommend module not found: {e}")
        
    meta = _build_meta_from_supabase()
    if meta.empty:
        return {"mode": "none", "track_ids": [], "tracks": []}
        
    seen_ids, log_df = _build_user_seen_and_log(user_id)
    seen_df = meta[meta["track_id"].isin(seen_ids)].copy()
    if seen_df.empty:
        seen_df = meta.head(0)
        
    profile = build_profile(seen_df)
    meta_all = meta.drop_duplicates("track_id")
    
    if log_df.empty or "endTime" not in log_df.columns or len(log_df) < 2:
        mode, _, track_ids = "FALLBACK_FIRST", 0, []
        fallback = meta_all[~meta_all["track_id"].isin(seen_ids)].sort_values("popularity", ascending=False).head(k)
        track_ids = fallback["track_id"].tolist()
    else:
        sessions = sessions_from_log(log_df, gap_minutes=30)
        if len(sessions) < 2:
            tr_sess, te_sess = [], sessions
        else:
            tr_sess, te_sess = time_split_sessions(sessions, train_ratio=0.8)
        trans = train_markov(tr_sess) if tr_sess else {}
        mode, _, track_ids = recommend_autoplay(current_track_id, trans, meta_all, seen_ids, profile, k=k)
        
    tracks = []
    for tid in track_ids:
        row = meta_all[meta_all["track_id"] == tid]
        if not row.empty:
            r = row.iloc[0]
            tracks.append({"track_id": tid, "title": r["title"], "artist": r["artist"], "youtube_video_id": r.get("youtube_video_id")})
            
    return {"mode": mode, "track_ids": track_ids, "tracks": tracks}

# ==========================================
# 12. AI Mixing
# ==========================================
@app.post("/api/ai/mix")
async def api_ai_mix(
    zip1: UploadFile = File(...),
    zip2: UploadFile = File(...),
    target_duration: float = 150.0,
    target_k: int = 5,
):
    try:
        from mixsense_mixing import run_mix
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"AI Mixing module not found: {e}")
    
    tmp1 = tmp2 = out_dir = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f1:
            tmp1 = f1.name
            f1.write(await zip1.read())
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f2:
            tmp2 = f2.name
            f2.write(await zip2.read())

        out_dir = tempfile.mkdtemp()
        result = run_mix(tmp1, tmp2, out_dir=out_dir, data_dir=out_dir, target_duration=target_duration, target_k=target_k)
        
        file_name = f"mixed_{uuid.uuid4().hex}.wav"
        with open(result.out_wav_path, "rb") as f:
            supabase.storage.from_("mix_tracks").upload(
                path=file_name,
                file=f,
                file_options={"content-type": "audio/wav"}
            )
            
        public_url = supabase.storage.from_("mix_tracks").get_public_url(file_name)
        
        return {
            "mix_audio_url": public_url,
            "log_json_path": result.log_json_path,
            "used_k": result.used_k,
            "events": [{"t_sec": e.t_sec, "mode": e.mode} for e in result.events],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp1 and os.path.exists(tmp1):
            try:
                os.unlink(tmp1)
            except OSError:
                pass
        if tmp2 and os.path.exists(tmp2):
            try:
                os.unlink(tmp2)
            except OSError:
                pass
        if out_dir and os.path.exists(out_dir):
            try:
                shutil.rmtree(out_dir)
            except OSError:
                pass

# ==========================================
# 13. Search History APIs
# ==========================================
@app.post("/api/search/history")
def add_search_history(req: SearchHistoryAddReq, user_id: str = Depends(verify_user)):
    """검색어를 search_history 테이블에 저장합니다."""
    response = supabase.table("search_history").insert({
        "user_id": user_id,
        "keyword": req.keyword
    }).execute()
    return {"status": "success", "data": response.data}

@app.get("/api/search/history")
def get_search_history(limit: int = 10, user_id: str = Depends(verify_user)):
    """현재 유저의 최근 검색어 이력을 최신순으로 조회합니다."""
    response = supabase.table("search_history").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
    return {"history": response.data}

@app.delete("/api/search/history/{search_history_id}")
def delete_search_history(search_history_id: str, user_id: str = Depends(verify_user)):
    """특정 검색 기록을 삭제합니다."""
    supabase.table("search_history").delete().eq("search_history_id", search_history_id).eq("user_id", user_id).execute()
    return {"status": "deleted"}