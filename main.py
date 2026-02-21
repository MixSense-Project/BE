import os
import requests
import pandas as pd
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from supabase import create_client, Client

# 기존에 만든 파이프라인 함수 임포트
from track_data_pipeline import run_pipeline, get_spotify_token

load_dotenv()
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Supabase 클라이언트 초기화
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="Music Metadata API")

# --- CORS 설정 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 프론트엔드에서 받을 데이터 구조 정의
class TrackRequest(BaseModel):
    title: str
    artist: str

def run_pipeline_background(tracks: List[TrackRequest]):
    """
    백그라운드에서 실행될 데이터 수집 및 DB 적재 태스크
    """
    df_data = [{"title": t.title, "artist": t.artist} for t in tracks]
    input_df = pd.DataFrame(df_data)
    
    try:
        # 기존 파이프라인 실행 (안정성을 위해 num_workers=1로 실행)
        run_pipeline(input_df, output_csv="api_accumulated_data.csv", num_workers=1)
        print("✅ 백그라운드 파이프라인 처리 및 DB 적재 완료")
    except Exception as e:
        print(f"❌ 백그라운드 파이프라인 에러: {e}")

@app.get("/search")
def search_spotify(query: str, limit: int = 5):
    """
    1단계: Spotify API를 통해 곡 후보를 빠르게 검색하여 반환
    """
    try:
        token = get_spotify_token()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Spotify 인증 실패")

    search_url = "https://api.spotify.com/v1/search" # 정상적인 Spotify API 주소
    headers = {"Authorization": f"Bearer {token}"}
    params = {"q": query, "type": "track", "limit": limit}
    
    response = requests.get(search_url, headers=headers, params=params)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Spotify 검색 실패")
        
    data = response.json()
    tracks = data.get("tracks", {}).get("items", [])
    
    results = []
    for item in tracks:
        artist_name = item["artists"][0]["name"] if item.get("artists") else ""
        album_images = item.get("album", {}).get("images", [])
        track_image = album_images[0]["url"] if album_images else None
        
        results.append({
            "track_id": item["id"],
            "title": item["name"],
            "artist": artist_name,
            "track_image_url": track_image
        })
        
    return {"results": results}

@app.post("/tracks/process")
def process_tracks(tracks: List[TrackRequest], background_tasks: BackgroundTasks):
    """
    2단계: 프론트엔드에서 선택된 곡들의 상세 정보를 추출하고 DB에 적재
    """
    if not tracks:
        raise HTTPException(status_code=400, detail="요청된 곡이 없습니다.")

    background_tasks.add_task(run_pipeline_background, tracks)
    
    return {
        "status": "processing",
        "message": f"{len(tracks)}곡에 대한 데이터 추출 및 DB 적재가 백그라운드에서 시작되었습니다."
    }

@app.get("/tracks")
def get_tracks(limit: int = 50):
    """
    3단계: DB에 적재된 곡 목록과 메타데이터(youtube_video_id 포함)를 프론트엔드에 제공
    """
    try:
        # 최신 업데이트 기준으로 데이터를 정렬하여 반환
        response = supabase.table("track").select("*").order("updated_at", desc=True).limit(limit).execute()
        return {"tracks": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB 조회 실패: {str(e)}")