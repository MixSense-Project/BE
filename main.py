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
from pydantic import BaseModel, Field, AliasChoices
from typing import List, Optional, Any, Literal
from dotenv import load_dotenv
from supabase import create_client, Client

# 기존 파이프라인 모듈
from track_data_pipeline import run_pipeline, get_spotify_token

load_dotenv()
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # service_role (DB/관리용)
# Auth get_user() 검증용: anon 키 사용 시 JWT 서명 검증이 정상 동작함 (service_role만 쓰면 403/invalid signature 발생 가능)
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
PLAYLIST_COVER_BUCKET = os.getenv("PLAYLIST_COVER_BUCKET", "playlist_covers")
MIX_TRACKS_BUCKET = os.getenv("MIX_TRACKS_BUCKET", "mix_tracks")
AI_MIX_MAX_UPLOAD_BYTES = int(os.getenv("AI_MIX_MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))  # 200MB
AI_MIX_ALLOWED_MIME_TYPES = {
    "application/zip",
    "application/x-zip-compressed",
    "multipart/x-zip",
    "application/octet-stream",
}

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in environment")
if not SUPABASE_ANON_KEY:
    raise RuntimeError("SUPABASE_ANON_KEY must be set in environment (do not fallback to service key)")

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

# Supabase 클라이언트: DB는 service_role, Auth 검증은 anon 키로
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
supabase_auth: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

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
    username: Optional[str] = None  # 앱에서 입력한 이름 → Supabase user_metadata / Display name 반영

class SendOTPRequest(BaseModel):
    email: str

class VerifyOTPRequest(BaseModel):
    email: str
    token: str  # 이메일로 발송된 6자리 인증번호
    type: Literal["signup", "email"] = "signup"  # signup: 가입 인증, email: 로그인 OTP 검증

class ProfileCreateReq(BaseModel):
    """마이페이지용: 이름(display_name)만 받습니다."""
    user_id: Optional[str] = None
    name: str

class ProfileUpdateReq(BaseModel):
    """마이페이지용: 이름(display_name)만 수정 가능합니다."""
    name: Optional[str] = None

class ProfileDeleteByNameReq(BaseModel):
    user_id: str
    profile_name: str

class ProfileEditByNameReq(BaseModel):
    user_id: str
    original_name: str
    updated: dict

class MyListReq(BaseModel):
    profile_id: str
    track_id: str = Field(
        validation_alias=AliasChoices("track_id", "content_id"),
        serialization_alias="track_id",
    )

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
    favorite_genres: Optional[List[str]] = None
    favorite_artists: Optional[List[str]] = None

class PlaylistCreateReq(BaseModel):
    title: str

class PlaylistUpdateReq(BaseModel):
    title: str

class PlaylistTrackReq(BaseModel):
    track_id: str

class PlaylistCoverUploadUrlReq(BaseModel):
    filename: str
    content_type: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("content_type", "contentType", "mime_type"),
    )

class PlaylistCoverSaveReq(BaseModel):
    cover_path: str

class SearchHistoryAddReq(BaseModel):
    event_type: Literal["query", "track_click"] = "query"
    keyword: Optional[str] = None
    track_id: Optional[str] = Field(default=None, validation_alias=AliasChoices("track_id", "trackId"))
    artist: Optional[str] = Field(default=None, validation_alias=AliasChoices("artist", "artist_name", "artistName"))
    track_image_url: Optional[str] = Field(default=None, validation_alias=AliasChoices("track_image_url", "trackImageUrl", "image_url", "imageUrl"))
    youtube_video_id: Optional[str] = Field(default=None, validation_alias=AliasChoices("youtube_video_id", "youtubeVideoId"))

# ==========================================
# 2. Auth Dependency
# ==========================================
def verify_user(authorization: str = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = authorization.split(" ")[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    try:
        # Auth 검증은 anon 키 클라이언트 사용 (service_role만 쓰면 403 / invalid JWT signature 발생 가능)
        user_response = supabase_auth.auth.get_user(token)
    except Exception as e:
        print(f"[AUTH] get_user error: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if not user_response or not user_response.user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user_response.user.id

# ==========================================
# 3. Auth APIs (디버깅 로그 포함)
# ==========================================
def _auth_log(tag: str, msg: str, **kwargs):
    """인증 디버깅 로그. 비밀번호는 출력하지 않음."""
    extra = " | ".join(f"{k}={v}" for k, v in kwargs.items() if k != "password")
    print(f"[AUTH] {tag} | {msg}" + (f" | {extra}" if extra else ""))

@app.post("/auth/signup")
def signup(req: AuthRequest, authorization: str = Header(None)):
    _auth_log("signup", "요청", email=req.email, username=req.username or "")
    try:
        # 1. Header에서 access_token 추출
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid token")
        token = authorization.split(" ")[1]

        # 2. 본인의 토큰을 사용하여 직접 비밀번호 업데이트 API 호출 (관리자 권한 불필요)
        headers = {
            "Authorization": f"Bearer {token}",
            "apikey": SUPABASE_KEY,
            "Content-Type": "application/json"
        }
        
        update_data = {"password": req.password}
        if req.username:
            update_data["data"] = {
                "display_name": req.username,
                "username": req.username
            }
        
        # Supabase GoTrue 엔드포인트로 직접 PUT 요청
        res = requests.put(f"{SUPABASE_URL}/auth/v1/user", headers=headers, json=update_data)
        
        if res.status_code != 200:
            error_msg = res.json().get("msg", "Update failed")
            raise Exception(error_msg)

        _auth_log("signup", "성공", email=req.email)
        return {"status": "success", "message": "Password and username updated"}
        
    except Exception as e:
        _auth_log("signup", "실패", email=req.email, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/send-otp")
def send_otp(req: SendOTPRequest):
    _auth_log("send_otp", "요청", email=req.email)
    try:
        supabase.auth.sign_in_with_otp({"email": req.email})
        _auth_log("send_otp", "성공", email=req.email)
        return {"status": "success", "message": "OTP sent to email"}
    except Exception as e:
        _auth_log("send_otp", "실패", email=req.email, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/verify")
def verify_otp(req: VerifyOTPRequest):
    token = (req.token or "").strip()
    _auth_log("verify", "요청", email=req.email, type=req.type, token_len=len(token))
    try:
        res = supabase.auth.verify_otp({
            "email": req.email,
            "token": token,
            "type": req.type
        })
        if not res.session:
            _auth_log("verify", "실패", email=req.email, reason="session 없음")
            raise HTTPException(status_code=401, detail="Verification failed")
        _auth_log("verify", "성공", email=req.email)
        return {"status": "success", "session": res.session}
    except Exception as e:
        _auth_log("verify", "실패", email=req.email, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/login")
def login(req: AuthRequest):
    _auth_log("login", "요청", email=req.email)
    try:
        res = supabase.auth.sign_in_with_password({"email": req.email, "password": req.password})
        if not res.session:
            _auth_log("login", "실패", email=req.email, reason="session 없음")
            raise HTTPException(status_code=401, detail="Invalid credentials")
        _auth_log("login", "성공", email=req.email)
        return {"status": "success", "session": res.session}
    except Exception as e:
        _auth_log("login", "실패", email=req.email, error=str(e))
        raise HTTPException(status_code=401, detail=str(e))

@app.post("/auth/logout")
def logout_user(authorization: str = Header(None)):
    """로그아웃: 현재 access token 기준으로 Supabase 세션을 종료합니다."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = authorization.split(" ")[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    _auth_log("logout", "요청")
    try:
        res = requests.post(
            f"{SUPABASE_URL}/auth/v1/logout",
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": SUPABASE_ANON_KEY,
                "Content-Type": "application/json",
            },
            timeout=15,
        )

        # 이미 만료/로그아웃된 토큰은 멱등적으로 성공 처리
        if res.status_code in (200, 204, 401):
            _auth_log("logout", "성공", status_code=res.status_code)
            return {"status": "success", "message": "Logged out"}

        err_msg = ""
        try:
            err_json = res.json()
            err_msg = (
                err_json.get("msg")
                or err_json.get("message")
                or err_json.get("error")
                or err_json.get("error_description")
                or str(err_json)
            )
        except Exception:
            err_msg = res.text or "Logout failed"

        _auth_log("logout", "실패", status_code=res.status_code, error=err_msg)
        raise HTTPException(status_code=400, detail=err_msg)
    except HTTPException:
        raise
    except Exception as e:
        _auth_log("logout", "실패", error=str(e))
        raise HTTPException(status_code=500, detail=f"Logout failed: {e}")

@app.delete("/auth/unsubscribe")
def unsubscribe_user(user_id: str = Depends(verify_user)):
    """회원 탈퇴: Auth 계정 삭제 성공 시 해당 계정 데이터만 정리합니다."""
    try:
        # 로그인 자체를 막기 위해 Auth 계정 삭제를 먼저 수행
        # 관리자 삭제는 서비스키를 Bearer로 강제하여 권한 오류(User not allowed)를 방지합니다.
        delete_res = requests.delete(
            f"{SUPABASE_URL}/auth/v1/admin/users/{user_id}",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
            },
            timeout=15,
        )
        if delete_res.status_code not in (200, 204):
            err_msg = ""
            try:
                err_json = delete_res.json()
                err_msg = (
                    err_json.get("msg")
                    or err_json.get("message")
                    or err_json.get("error")
                    or err_json.get("error_description")
                    or str(err_json)
                )
            except Exception:
                err_msg = delete_res.text or "unknown error"
            raise RuntimeError(f"{delete_res.status_code} {err_msg}")
    except Exception as e:
        print(f"Auth user deletion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete auth user: {e}")

    # Auth 삭제 성공 후, 해당 user_id 데이터만 정리
    cleanup_errors = []
    cleaned_targets = []

    def _safe_cleanup(label: str, fn):
        try:
            fn()
            cleaned_targets.append(label)
        except Exception as e:
            cleanup_errors.append(f"{label}: {e}")

    # playlist_tracks는 playlist_id 기준이라 먼저 대상 playlist 조회 후 삭제
    playlist_ids = []
    try:
        playlist_res = supabase.table("playlist").select("playlist_id").eq("user_id", user_id).execute()
        playlist_ids = [r.get("playlist_id") for r in (playlist_res.data or []) if r.get("playlist_id")]
    except Exception as e:
        cleanup_errors.append(f"playlist lookup: {e}")

    if playlist_ids:
        _safe_cleanup(
            "playlist_tracks",
            lambda: supabase.table("playlist_tracks").delete().in_("playlist_id", playlist_ids).execute(),
        )

    _safe_cleanup("playlist", lambda: supabase.table("playlist").delete().eq("user_id", user_id).execute())
    _safe_cleanup("likes", lambda: supabase.table("likes").delete().eq("user_id", user_id).execute())
    _safe_cleanup("play_logs", lambda: supabase.table("play_logs").delete().eq("user_id", user_id).execute())
    _safe_cleanup("search_history", lambda: supabase.table("search_history").delete().eq("user_id", user_id).execute())
    _safe_cleanup("user_recommendations", lambda: supabase.table("user_recommendations").delete().eq("user_id", user_id).execute())
    _safe_cleanup("profiles", lambda: supabase.table("profiles").delete().eq("user_id", user_id).execute())

    if cleanup_errors:
        print("[UNSUBSCRIBE] data cleanup warnings:", cleanup_errors)
        return {
            "status": "unsubscribed_with_warning",
            "message": "Auth account deleted, but some user data cleanup failed.",
            "cleaned_targets": cleaned_targets,
            "cleanup_errors": cleanup_errors,
        }

    return {
        "status": "unsubscribed",
        "message": "Auth account and user data deleted successfully.",
        "cleaned_targets": cleaned_targets,
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
    """본인 계정 프로필 생성/수정(1유저 1로우). 마이페이지용으로 이름(display_name)만 받습니다."""
    response = supabase.table("profiles").upsert({
        "user_id": user_id,
        "display_name": req.name,
    }, on_conflict="user_id").execute()
    return {"status": "success", "profile": response.data}

@app.put("/profile/{profile_id}")
def update_profile_full(profile_id: str, req: ProfileUpdateReq, user_id: str = Depends(verify_user)):
    """마이페이지용: 이름(display_name)만 수정합니다."""
    if profile_id != user_id:
        raise HTTPException(status_code=403, detail="Can only update your own profile")
    update_payload = {}
    if req.name is not None:
        update_payload["display_name"] = req.name
    if not update_payload:
        raise HTTPException(status_code=400, detail="At least 'name' is required for update")
    # profiles는 user_id 단일 PK 구조이므로 path profile_id 대신 user_id 기준으로 수정
    response = supabase.table("profiles").update(update_payload).eq("user_id", user_id).execute()
    return {"status": "success", "profile": response.data}

@app.patch("/profile/{profile_id}")
def update_profile_partial(profile_id: str, req: ProfileUpdateReq, user_id: str = Depends(verify_user)):
    """마이페이지용: 이름(display_name)만 부분 수정합니다."""
    if profile_id != user_id:
        raise HTTPException(status_code=403, detail="Can only update your own profile")
    update_data = {k: v for k, v in req.model_dump().items() if v is not None}
    if update_data.get("name") is not None:
        update_data["display_name"] = update_data.pop("name")
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")
    # profiles는 user_id 단일 PK 구조이므로 path profile_id 대신 user_id 기준으로 수정
    response = supabase.table("profiles").update(update_data).eq("user_id", user_id).execute()
    return {"status": "success", "profile": response.data}

@app.delete("/profile/{profile_id}")
def delete_profile(profile_id: str, user_id: str = Depends(verify_user)):
    if profile_id != user_id:
        raise HTTPException(status_code=403, detail="Can only delete your own profile")
    # profiles는 user_id 단일 PK 구조이므로 path profile_id 대신 user_id 기준으로 삭제
    supabase.table("profiles").delete().eq("user_id", user_id).execute()
    return {"status": "deleted"}

@app.get("/profile/by_user")
def get_profiles_by_user(user_id: str = Depends(verify_user)):
    """본인 프로필 목록만 조회 (인증 필수)."""
    response = supabase.table("profiles").select("*").eq("user_id", user_id).execute()
    return {"profiles": response.data}

@app.post("/profile/add_profile")
def add_profile_custom(req: ProfileCreateReq, user_id: str = Depends(verify_user)):
    """본인 계정 프로필 생성/수정(1유저 1로우). 마이페이지용으로 이름(display_name)만 받습니다."""
    response = supabase.table("profiles").upsert({
        "user_id": user_id,
        "display_name": req.name,
    }, on_conflict="user_id").execute()
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
    raw = req.updated or {}
    mapped = {}
    if raw.get("name") is not None:
        mapped["display_name"] = raw.get("name")
    if raw.get("display_name") is not None:
        mapped["display_name"] = raw.get("display_name")
    if raw.get("favorite_genres") is not None:
        mapped["favorite_genres"] = [str(x) for x in (raw.get("favorite_genres") or [])]
    if raw.get("favorite_artists") is not None:
        mapped["favorite_artists"] = [str(x) for x in (raw.get("favorite_artists") or [])]
    if not mapped:
        raise HTTPException(status_code=400, detail="No updatable fields provided")
    response = supabase.table("profiles").update(mapped).eq("user_id", req.user_id).eq("display_name", req.original_name).execute()
    return {"status": "updated", "profile": response.data}

@app.patch("/api/profile/preferences")
def update_user_preferences(req: PreferenceUpdateReq, user_id: str = Depends(verify_user)):
    """온보딩 과정에서 선택한 선호 장르와 아티스트를 profile에 저장합니다.
    artist_id는 DB 조회 없이 그대로 저장합니다 (Spotify artist_id 문자열 사용 가능).
    """
    # DB는 text[] 타입이므로 문자열 리스트로만 전달 (numeric id 조회 없음)
    favorite_genres = [str(x) for x in (req.favorite_genres or [])]
    favorite_artists = [str(x) for x in (req.favorite_artists or [])]
    update_data = {
        "favorite_genres": favorite_genres,
        "favorite_artists": favorite_artists,
    }
    try:
        response = supabase.table("profiles").update(update_data).eq("user_id", user_id).execute()
        # 프로필 row가 아직 없는 경우를 대비해 upsert로 보강
        if not response.data:
            response = supabase.table("profiles").upsert({
                "user_id": user_id,
                "favorite_genres": favorite_genres,
                "favorite_artists": favorite_artists,
            }, on_conflict="user_id").execute()
        return {"status": "success", "profile": response.data}
    except Exception as e:
        err_msg = str(e)
        print(f"[PATCH /api/profile/preferences] error: {err_msg}")
        raise HTTPException(status_code=500, detail=err_msg)

# ==========================================
# 5. User MyList & Like APIs
# ==========================================

def verify_profile_ownership(profile_id: str, user_id: str):
    """해당 프로필이 현재 로그인한 유저의 소유인지 검증합니다."""
    # profiles는 user_id 단일 PK 구조
    if profile_id != user_id:
        raise HTTPException(status_code=403, detail="You do not own this profile")
    res = supabase.table("profiles").select("user_id").eq("user_id", user_id).execute()
    if not res.data:
        raise HTTPException(status_code=403, detail="You do not own this profile")

@app.post("/user/mylist")
def add_to_mylist(req: MyListReq, user_id: str = Depends(verify_user)):
    verify_profile_ownership(req.profile_id, user_id)
    response = supabase.table("likes").insert({"user_id": req.profile_id, "track_id": req.track_id}).execute()
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
    existing = supabase.table("likes").select("*").eq("user_id", req.profile_id).eq("track_id", req.track_id).execute()
    if existing.data:
        supabase.table("likes").delete().eq("user_id", req.profile_id).eq("track_id", req.track_id).execute()
        return {"status": "unliked"}
    supabase.table("likes").insert({"user_id": req.profile_id, "track_id": req.track_id}).execute()
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
def _playlist_to_client(row: dict) -> dict:
    cover_path = row.get("cover_path")
    cover_url = supabase.storage.from_(PLAYLIST_COVER_BUCKET).get_public_url(cover_path) if cover_path else None
    return {
        "id": row.get("playlist_id"),
        "title": row.get("title"),
        "cover_url": cover_url,
    }


def _create_signed_upload_url_with_service_key(bucket: str, storage_path: str) -> dict:
    """
    storage3 SDK 호출이 환경/버전 이슈로 실패할 때를 대비한 REST fallback.
    service_role 키로 직접 signed upload URL을 생성합니다.
    """
    encoded_parts = [requests.utils.quote(part, safe="") for part in storage_path.strip("/").split("/") if part]
    endpoint = f"{SUPABASE_URL}/storage/v1/object/upload/sign/{bucket}/" + "/".join(encoded_parts)
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }
    # storage sign API는 application/json 요청에서 빈 body를 허용하지 않는 경우가 있어 {}를 보냅니다.
    res = requests.post(endpoint, headers=headers, json={}, timeout=15)
    if res.status_code not in (200, 201):
        try:
            err = res.json()
            err_msg = err.get("message") or err.get("error") or str(err)
        except Exception:
            err_msg = res.text or "unknown error"
        raise RuntimeError(f"storage sign fallback failed ({res.status_code}): {err_msg}")
    data = res.json() or {}
    relative_url = (data.get("url") or "").lstrip("/")
    if not relative_url:
        raise RuntimeError("storage sign fallback failed: missing signed url")
    full_url = f"{SUPABASE_URL}/storage/v1/{relative_url}"
    token = data.get("token")
    if not token and "token=" in full_url:
        token = full_url.split("token=", 1)[1].split("&", 1)[0]
    return {
        "signed_url": full_url,
        "signedUrl": full_url,
        "token": token,
        "path": storage_path,
    }

@app.post("/api/playlists")
def create_playlist(req: PlaylistCreateReq, user_id: str = Depends(verify_user)):
    """새로운 플레이리스트를 생성합니다."""
    response = supabase.table("playlist").insert({
        "user_id": user_id,
        "title": req.title
    }).execute()
    row = (response.data or [{}])[0]
    return {"status": "success", "playlist": _playlist_to_client(row)}

@app.get("/api/playlists")
def get_my_playlists(user_id: str = Depends(verify_user)):
    """현재 로그인한 유저의 플레이리스트 목록을 조회합니다."""
    response = supabase.table("playlist").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
    rows = response.data or []
    return {"playlists": [_playlist_to_client(r) for r in rows]}

@app.patch("/api/playlists/{playlist_id}")
def update_playlist_title(playlist_id: str, req: PlaylistUpdateReq, user_id: str = Depends(verify_user)):
    """플레이리스트의 이름을 수정합니다."""
    response = supabase.table("playlist").update({"title": req.title}).eq("playlist_id", playlist_id).eq("user_id", user_id).execute()
    row = (response.data or [{}])[0]
    return {"status": "success", "playlist": _playlist_to_client(row)}

@app.post("/api/playlists/{playlist_id}/cover/upload-url")
@app.post("/playlists/{playlist_id}/cover/upload-url")
def create_playlist_cover_upload_url(
    playlist_id: str,
    req: PlaylistCoverUploadUrlReq,
    user_id: str = Depends(verify_user),
):
    """플레이리스트 커버 이미지 업로드용 presigned URL을 발급합니다."""
    playlist_check = supabase.table("playlist").select("playlist_id").eq("playlist_id", playlist_id).eq("user_id", user_id).execute()
    if not playlist_check.data:
        raise HTTPException(status_code=403, detail="Playlist ownership verification failed")

    ext = Path(req.filename).suffix.lower()
    allowed_ext = {".jpg", ".jpeg", ".png", ".webp"}
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail="Unsupported file extension. Allowed: jpg, jpeg, png, webp")

    if req.content_type:
        allowed_ct = {"image/jpeg", "image/png", "image/webp"}
        if req.content_type.lower() not in allowed_ct:
            raise HTTPException(status_code=400, detail="Unsupported content_type. Allowed: image/jpeg, image/png, image/webp")

    storage_path = f"{user_id}/{playlist_id}/{uuid.uuid4().hex}{ext}"
    try:
        try:
            signed = supabase.storage.from_(PLAYLIST_COVER_BUCKET).create_signed_upload_url(storage_path)
        except Exception as sdk_err:
            # SDK 실패 시 service_role 직접 REST 호출로 fallback
            print(f"[PLAYLIST_COVER] SDK sign failed, fallback to REST: {sdk_err}")
            signed = _create_signed_upload_url_with_service_key(PLAYLIST_COVER_BUCKET, storage_path)
    except Exception as e:
        print(
            f"[PLAYLIST_COVER] sign failed | bucket={PLAYLIST_COVER_BUCKET} "
            f"| url={SUPABASE_URL} | user_id={user_id} | playlist_id={playlist_id} | error={e}"
        )
        raise HTTPException(status_code=500, detail=f"Failed to create signed upload url: {e}")
    return {
        "bucket": PLAYLIST_COVER_BUCKET,
        "storage_path": storage_path,
        "storagePath": storage_path,
        "signed_url": signed.get("signed_url") or signed.get("signedUrl"),
        "signedUrl": signed.get("signed_url") or signed.get("signedUrl"),
        "token": signed.get("token"),
        "path": signed.get("path"),
    }

@app.patch("/api/playlists/{playlist_id}/cover")
@app.patch("/playlists/{playlist_id}/cover")
def update_playlist_cover(
    playlist_id: str,
    req: PlaylistCoverSaveReq,
    user_id: str = Depends(verify_user),
):
    """업로드 완료된 커버 path를 playlist에 저장합니다."""
    playlist_check = supabase.table("playlist").select("playlist_id").eq("playlist_id", playlist_id).eq("user_id", user_id).execute()
    if not playlist_check.data:
        raise HTTPException(status_code=403, detail="Playlist ownership verification failed")

    cover_path = (req.cover_path or "").strip().lstrip("/")
    if not cover_path:
        raise HTTPException(status_code=400, detail="cover_path is required")
    expected_prefix = f"{user_id}/{playlist_id}/"
    if not cover_path.startswith(expected_prefix):
        raise HTTPException(status_code=400, detail="cover_path must match the issued user/playlist path")

    response = supabase.table("playlist").update({"cover_path": cover_path}).eq("playlist_id", playlist_id).eq("user_id", user_id).execute()
    row = (response.data or [{}])[0]
    return {"status": "success", "playlist": _playlist_to_client(row)}

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
    playlist_check = supabase.table("playlist").select("playlist_id").eq("playlist_id", playlist_id).eq("user_id", user_id).execute()
    if not playlist_check.data:
        raise HTTPException(status_code=403, detail="Playlist ownership verification failed")
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

    # track 테이블의 youtube_video_id를 함께 내려주기 위해 Spotify track_id로 매핑 조회
    track_ids = [item.get("id") for item in items if item.get("id")]
    youtube_map = {}
    if track_ids:
        meta_res = supabase.table("track").select("track_id, youtube_video_id").in_("track_id", track_ids).execute()
        for row in (meta_res.data or []):
            youtube_map[str(row.get("track_id"))] = row.get("youtube_video_id")

    results = [{
        "track_id": item["id"],
        "title": item["name"],
        "artist": item["artists"][0]["name"] if item.get("artists") else "",
        "track_image_url": item.get("album", {}).get("images", [{}])[0].get("url"),
        "youtube_video_id": youtube_map.get(str(item.get("id"))),
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
    response = supabase.table("track").select("track_id, artist_id, title, artist, popularity, release_date, youtube_video_id, track_image_url").execute()
    rows = response.data or []
    if not rows:
        return pd.DataFrame(columns=["track_id", "artist_id", "artist", "title", "popularity", "release_date", "youtube_video_id", "track_image_url", "artist_norm", "title_norm"])
    df = pd.DataFrame(rows)
    df["track_id"] = df["track_id"].astype(str)
    df["artist_id"] = df["artist_id"].fillna("").astype(str)
    df["artist"] = df["artist"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    if "youtube_video_id" not in df.columns:
        df["youtube_video_id"] = ""
    df["youtube_video_id"] = df["youtube_video_id"].fillna("").astype(str)
    if "track_image_url" not in df.columns:
        df["track_image_url"] = ""
    df["track_image_url"] = df["track_image_url"].fillna("").astype(str)

    from recommender_home_autoplay import normalize_text
    df["artist_norm"] = df["artist"].map(normalize_text)
    df["title_norm"] = df["title"].map(normalize_text)
    return df

def _build_user_seen_and_log(user_id: str):
    pl = supabase.table("play_logs").select("track_id, played_at, ms_played").eq("user_id", user_id).order("played_at", desc=False).limit(5000).execute()
    log_rows = pl.data or []
    likes = supabase.table("likes").select("track_id").eq("user_id", user_id).execute()
    like_ids = {r["track_id"] for r in (likes.data or [])}
    
    seen_ids = set()
    log_list = []
    for r in log_rows:
        tid = r.get("track_id")
        if tid:
            seen_ids.add(tid)
        ts = r.get("played_at")
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
            tracks.append({
                "track_id": tid,
                "title": r["title"],
                "artist": r["artist"],
                "youtube_video_id": r.get("youtube_video_id"),
                "track_image_url": r.get("track_image_url"),
            })
            
    return {"mode": mode, "track_ids": track_ids, "tracks": tracks}

# ==========================================
# 12. AI Mixing
# ==========================================
def _validate_ai_mix_inputs(zip1: UploadFile, zip2: UploadFile, target_duration: float, target_k: int):
    if target_duration < 30 or target_duration > 300:
        raise HTTPException(status_code=400, detail="target_duration must be between 30 and 300 seconds")
    if target_k < 1 or target_k > 12:
        raise HTTPException(status_code=400, detail="target_k must be between 1 and 12")

    for label, upload in [("zip1", zip1), ("zip2", zip2)]:
        if not upload or not upload.filename:
            raise HTTPException(status_code=400, detail=f"{label} is required")
        if not upload.filename.lower().endswith(".zip"):
            raise HTTPException(status_code=400, detail=f"{label} must be a .zip file")
        content_type = (upload.content_type or "").lower().strip()
        if content_type and content_type not in AI_MIX_ALLOWED_MIME_TYPES:
            raise HTTPException(status_code=400, detail=f"{label} has unsupported content_type: {content_type}")


async def _save_upload_as_temp_zip(upload: UploadFile, max_bytes: int) -> str:
    total = 0
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
        temp_path = f.name
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(
                    status_code=400,
                    detail=f"Upload size exceeds limit ({max_bytes // (1024 * 1024)}MB): {upload.filename}",
                )
            f.write(chunk)
    return temp_path


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
    
    _validate_ai_mix_inputs(zip1, zip2, target_duration, target_k)

    tmp1 = tmp2 = out_dir = None
    try:
        tmp1 = await _save_upload_as_temp_zip(zip1, max_bytes=AI_MIX_MAX_UPLOAD_BYTES)
        tmp2 = await _save_upload_as_temp_zip(zip2, max_bytes=AI_MIX_MAX_UPLOAD_BYTES)

        out_dir = tempfile.mkdtemp()
        result = run_mix(tmp1, tmp2, out_dir=out_dir, data_dir=out_dir, target_duration=target_duration, target_k=target_k)
        
        file_name = f"mixed_{uuid.uuid4().hex}.wav"
        with open(result.out_wav_path, "rb") as f:
            supabase.storage.from_(MIX_TRACKS_BUCKET).upload(
                path=file_name,
                file=f,
                file_options={"content-type": "audio/wav"}
            )
            
        public_url = supabase.storage.from_(MIX_TRACKS_BUCKET).get_public_url(file_name)

        log_storage_path = f"logs/mix_log_{uuid.uuid4().hex}.json"
        with open(result.log_json_path, "rb") as f:
            supabase.storage.from_(MIX_TRACKS_BUCKET).upload(
                path=log_storage_path,
                file=f,
                file_options={"content-type": "application/json"},
            )
        log_public_url = supabase.storage.from_(MIX_TRACKS_BUCKET).get_public_url(log_storage_path)
        
        return {
            "mix_audio_url": public_url,
            "log_json_path": log_storage_path,
            "log_json_url": log_public_url,
            "used_k": result.used_k,
            "events": [{"t_sec": e.t_sec, "mode": e.mode} for e in result.events],
        }
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
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
def _is_blank(v: Optional[str]) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


@app.post("/api/search/history")
def add_search_history(req: SearchHistoryAddReq, user_id: str = Depends(verify_user)):
    """검색어를 search_history 테이블에 저장합니다.
    - query: 키워드 입력 이력 저장
    - track_click: 트랙 클릭 이력 저장( track_id 필수, track 테이블로 artist/이미지/youtube 보강 )
    """
    event_type = req.event_type
    track_id = (req.track_id or "").strip()
    resolved_keyword = (req.keyword or "").strip()
    resolved_artist = (req.artist or "").strip() or None
    resolved_track_image_url = (req.track_image_url or "").strip() or None
    resolved_youtube_video_id = (req.youtube_video_id or "").strip() or None
    resolved_track_id = track_id or None

    def _apply_track_row(track_row: dict):
        nonlocal resolved_track_id, resolved_keyword, resolved_artist, resolved_track_image_url, resolved_youtube_video_id
        if _is_blank(resolved_track_id):
            resolved_track_id = track_row.get("track_id")
        if _is_blank(resolved_artist):
            resolved_artist = track_row.get("artist")
        if _is_blank(resolved_track_image_url):
            resolved_track_image_url = track_row.get("track_image_url")
        if _is_blank(resolved_youtube_video_id):
            resolved_youtube_video_id = track_row.get("youtube_video_id")
        if _is_blank(resolved_keyword):
            resolved_keyword = track_row.get("title") or ""

    if event_type == "query":
        if not resolved_keyword:
            raise HTTPException(status_code=400, detail="keyword is required when event_type='query'")
        # 프론트가 keyword만 보내는 현재 구조를 지원하기 위해 title 기반 보강
        if (not resolved_artist) or (not resolved_track_image_url) or (not resolved_youtube_video_id) or (not resolved_track_id):
            try:
                # 1) title 정확 일치 우선
                exact_res = (
                    supabase.table("track")
                    .select("track_id, title, artist, track_image_url, youtube_video_id, popularity, updated_at")
                    .eq("title", resolved_keyword)
                    .order("popularity", desc=True)
                    .limit(1)
                    .execute()
                )
                track_row = (exact_res.data or [None])[0]

                # 2) 정확 일치가 없으면 부분 검색으로 fallback
                if not track_row:
                    fuzzy_res = (
                        supabase.table("track")
                        .select("track_id, title, artist, track_image_url, youtube_video_id, popularity, updated_at")
                        .ilike("title", f"%{resolved_keyword}%")
                        .order("popularity", desc=True)
                        .limit(1)
                        .execute()
                    )
                    track_row = (fuzzy_res.data or [None])[0]

                if track_row:
                    _apply_track_row(track_row)
            except Exception as e:
                print(f"[POST /api/search/history] keyword lookup failed: {e}")
    elif event_type == "track_click":
        if not track_id:
            raise HTTPException(status_code=400, detail="track_id is required when event_type='track_click'")
        try:
            track_res = (
                supabase.table("track")
                .select("track_id, title, artist, track_image_url, youtube_video_id")
                .eq("track_id", track_id)
                .limit(1)
                .execute()
            )
            if not track_res.data:
                raise HTTPException(status_code=404, detail="track_id not found in track table")
            track_row = track_res.data[0]
            _apply_track_row(track_row)
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=f"track lookup failed: {e}")

    # track_id가 있으면 마지막으로 한 번 더 메타를 보강하여 null 저장 가능성을 낮춥니다.
    if resolved_track_id and (_is_blank(resolved_artist) or _is_blank(resolved_track_image_url) or _is_blank(resolved_youtube_video_id)):
        try:
            enrich_res = (
                supabase.table("track")
                .select("track_id, title, artist, track_image_url, youtube_video_id")
                .eq("track_id", str(resolved_track_id))
                .limit(1)
                .execute()
            )
            enrich_row = (enrich_res.data or [None])[0]
            if enrich_row:
                _apply_track_row(enrich_row)
        except Exception as e:
            print(f"[POST /api/search/history] track_id enrich failed: {e}")

    response = supabase.table("search_history").insert({
        "user_id": user_id,
        "event_type": event_type,
        "track_id": resolved_track_id,
        "keyword": resolved_keyword or None,
        "artist": resolved_artist,
        "track_image_url": resolved_track_image_url,
        "youtube_video_id": resolved_youtube_video_id,
    }).execute()
    return {"status": "success", "data": response.data}

@app.get("/api/search/history")
def get_search_history(limit: int = 10, user_id: str = Depends(verify_user)):
    """현재 유저의 최근 검색어 이력을 최신순으로 조회합니다."""
    response = supabase.table("search_history").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
    rows = response.data or []

    # 기존에 null로 저장된 row도 응답 시점에 track 메타로 보강합니다.
    track_ids_to_enrich = list({
        str(r.get("track_id"))
        for r in rows
        if r.get("track_id") and (
            _is_blank(r.get("artist")) or _is_blank(r.get("track_image_url")) or _is_blank(r.get("youtube_video_id"))
        )
    })
    track_meta_map = {}
    if track_ids_to_enrich:
        try:
            meta_res = (
                supabase.table("track")
                .select("track_id, title, artist, track_image_url, youtube_video_id")
                .in_("track_id", track_ids_to_enrich)
                .execute()
            )
            for m in (meta_res.data or []):
                tid = str(m.get("track_id") or "")
                if tid:
                    track_meta_map[tid] = m
        except Exception as e:
            print(f"[GET /api/search/history] enrich lookup failed: {e}")

    for r in rows:
        tid = str(r.get("track_id") or "")
        meta = track_meta_map.get(tid)
        if not meta:
            continue
        if _is_blank(r.get("artist")):
            r["artist"] = meta.get("artist")
        if _is_blank(r.get("track_image_url")):
            r["track_image_url"] = meta.get("track_image_url")
        if _is_blank(r.get("youtube_video_id")):
            r["youtube_video_id"] = meta.get("youtube_video_id")
        if _is_blank(r.get("keyword")):
            r["keyword"] = meta.get("title")

    return {"history": rows}

@app.delete("/api/search/history/{search_history_id}")
def delete_search_history(search_history_id: str, user_id: str = Depends(verify_user)):
    """특정 검색 기록을 삭제합니다."""
    supabase.table("search_history").delete().eq("search_history_id", search_history_id).eq("user_id", user_id).execute()
    return {"status": "deleted"}