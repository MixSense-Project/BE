import os
import json
import re
import requests
import pandas as pd
import time
import concurrent.futures
from datetime import datetime, timezone
from dotenv import load_dotenv
from tqdm import tqdm

from supabase import create_client, Client

# ---------------------------------------------------------
# 0. 환경 변수 및 상수 설정
# ---------------------------------------------------------
load_dotenv()
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

ALLOWED_GENRES = [
    "Pop", "Hip Hop", "R&B", "Electronic", "Rock", "Jazz",
    "Classical", "Country", "Latin", "K-Pop", "Indie",
    "Acoustic", "Lofi", "Reggae", "World"
]

ALLOWED_MOODS = [
    "Happy", "Sad", "Energetic", "Calm", "Romantic", "Angry",
    "Chill", "Dark", "Uplifting", "Melancholic", "Dreamy",
    "Neutral"
]

ALLOWED_CONTEXTS = [
    "Workout", "Study", "Party", "Sleep", "Drive", "Relax",
    "Work", "Commute", "Focus", "Morning", "Night",
    "Gaming"
]

TRACKER_FILE = "artist_image_tracker.json"

# ---------------------------------------------------------
# 1. 메타데이터 추출 및 보충 모듈
# ---------------------------------------------------------
def get_spotify_token() -> str:
    url = "https://accounts.spotify.com/api/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": SPOTIFY_CLIENT_ID,
        "client_secret": SPOTIFY_CLIENT_SECRET
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(url, data=payload, headers=headers, timeout=15)
    response.raise_for_status()
    return response.json().get("access_token")

def get_youtube_video_id(title: str, artist: str) -> str:
    query = f"{title} {artist} official audio"
    url = "https://www.youtube.com/results"
    try:
        response = requests.get(url, params={"search_query": query})
        response.raise_for_status()
        video_ids = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', response.text)
        if video_ids:
            return video_ids[0]
    except Exception as e:
        print(f"YouTube Search Error for '{title}': {e}")
    return None

def fetch_all_tracks_from_supabase() -> list:
    print("Fetching existing track data from Supabase...")
    all_records = []
    limit = 1000
    offset = 0
    
    while True:
        response = supabase.table("track").select("*").range(offset, offset + limit - 1).execute()
        records = response.data
        if not records:
            break
        all_records.extend(records)
        offset += limit
        if len(records) < limit:
            break
            
    print(f"Loaded {len(all_records)} existing records.")
    return all_records

def process_track_record(record: dict, token: str) -> dict:
    title = record.get("title")
    artist = record.get("artist")
    
    if not title or not artist:
        return record
        
    updated_data = record.copy()

    # 1. Spotify 데이터 결측치 확인 및 보충
    needs_spotify = not updated_data.get("track_id") or not updated_data.get("popularity") or not updated_data.get("track_image_url")
    if needs_spotify:
        search_url = "https://api.spotify.com/v1/search"
        headers = {"Authorization": f"Bearer {token}"}
        params = {"q": f"track:{title} artist:{artist}", "type": "track", "limit": 1}
        
        try:
            res = requests.get(search_url, headers=headers, params=params)
            res.raise_for_status()
            tracks = res.json().get("tracks", {}).get("items", [])
            
            if tracks:
                track_info = tracks[0]
                artist_id = track_info.get("artists")[0].get("id") if track_info.get("artists") else None
                album_images = track_info.get("album", {}).get("images", [])
                
                updated_data.update({
                    "track_id": track_info.get("id"),
                    "artist_id": artist_id,
                    "popularity": track_info.get("popularity"),
                    "release_date": track_info.get("album", {}).get("release_date"),
                    "duration_ms": track_info.get("duration_ms"),
                    "track_image_url": album_images[0].get("url") if album_images else None
                })
                
                if artist_id and not updated_data.get("artist_image_url"):
                    artist_url = f"https://api.spotify.com/v1/artists/{artist_id}"
                    artist_res = requests.get(artist_url, headers=headers)
                    if artist_res.status_code == 200:
                        artist_images = artist_res.json().get("images", [])
                        updated_data["artist_image_url"] = artist_images[0].get("url") if artist_images else None
                        
        except Exception as e:
            pass

    # 2. Perplexity API 호출 (분석 메타데이터 결측치 확인)
    analyze_keys = ["genre", "sub_genre", "producer", "writer", "mood_tags", "context_tags", "reference_url"]
    missing_analysis = [k for k in analyze_keys if not updated_data.get(k) and k != "reference_url"]
    
    if missing_analysis and updated_data.get("title") and updated_data.get("artist"):
        genre_str = ", ".join(ALLOWED_GENRES)
        mood_str = ", ".join(ALLOWED_MOODS)
        context_str = ", ".join(ALLOWED_CONTEXTS)
        
        perp_url = "https://api.perplexity.ai/chat/completions"
        system_prompt = (
            "You are a music metadata extraction assistant. "
            f"Output strictly valid JSON with these keys: {', '.join(missing_analysis)}. "
            f"For 'genre' and 'sub_genre', MUST select from: [{genre_str}]. "
            f"For 'mood_tags', select 1 to 3 from: [{mood_str}]. "
            f"For 'context_tags', select 1 to 3 from: [{context_str}]. "
            "Provide a single comma-separated string for tags. If completely unknown, return null."
        )
        user_prompt = f"Provide missing metadata for '{updated_data['title']}' by {updated_data['artist']}."
        
        payload = {
            "model": "sonar-pro",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2
        }
        perp_headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {PERPLEXITY_API_KEY}"
        }
        
        try:
            res = requests.post(perp_url, json=payload, headers=perp_headers)
            res.raise_for_status()
            result_text = res.json()["choices"][0]["message"]["content"]
            cleaned_text = result_text.replace("```json", "").replace("```", "").strip()
            p_json = json.loads(cleaned_text)
            
            for k, v in p_json.items():
                if str(v).lower() not in ["null", "none", "n/a", ""]:
                    updated_data[k] = v
        except Exception as e:
            pass

    # 3. Gemini API Fallback 호출 (Perplexity 이후에도 남은 결측치 보충)
    still_missing = [k for k in analyze_keys if not updated_data.get(k) and k != "reference_url"]
    
    if still_missing and GEMINI_API_KEY:
        genre_str = ", ".join(ALLOWED_GENRES)
        mood_str = ", ".join(ALLOWED_MOODS)
        context_str = ", ".join(ALLOWED_CONTEXTS)
        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
        keys_str = ", ".join(f"'{k}'" for k in still_missing)
        
        gemini_prompt = (
            f"You are a music metadata extraction assistant. "
            f"Find the missing metadata for '{updated_data['title']}' by {updated_data['artist']}. "
            f"Output strictly valid JSON containing ONLY these keys: {keys_str}. "
            f"For 'genre' or 'sub_genre', MUST select from: [{genre_str}]. "
            f"For 'mood_tags', MUST select from: [{mood_str}]. "
            f"For 'context_tags', MUST select from: [{context_str}]. "
            "Make your best educated guess to avoid null."
        )
        
        gemini_payload = {
            "contents": [{"parts": [{"text": gemini_prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.2
            }
        }
        
        try:
            g_res = requests.post(gemini_url, json=gemini_payload)
            g_res.raise_for_status()
            g_data = g_res.json()
            
            if "candidates" in g_data and g_data["candidates"]:
                g_text = g_data["candidates"][0]["content"]["parts"][0]["text"]
                g_cleaned = g_text.replace("```json", "").replace("```", "").strip()
                g_json = json.loads(g_cleaned)
                
                for k, v in g_json.items():
                    if str(v).lower() not in ["null", "none", "n/a", ""]:
                        updated_data[k] = v
                        
                if not updated_data.get("reference_url"):
                     updated_data["reference_url"] = "gemini_inferred"
        except Exception as e:
            pass

    # 4. 장르와 서브장르 중복 후처리 로직
    if updated_data.get("genre") and updated_data.get("sub_genre"):
        if str(updated_data["genre"]).strip().lower() == str(updated_data["sub_genre"]).strip().lower():
            updated_data["sub_genre"] = None

    # 5. YouTube API 호출 (결측치 확인)
    if not updated_data.get("youtube_video_id") and updated_data.get("title") and updated_data.get("artist"):
        updated_data["youtube_video_id"] = get_youtube_video_id(updated_data["title"], updated_data["artist"])

    return updated_data

def upload_to_supabase(df: pd.DataFrame, table_name: str = "track", silent: bool = False):
    if df.empty:
        return

    valid_df = df.dropna(subset=['track_id']).copy()
    if valid_df.empty:
        return

    # upsert 시 동일 track_id 중복 시 "cannot affect row a second time" 방지
    valid_df = valid_df.drop_duplicates(subset=['track_id'], keep='last')
    valid_df['updated_at'] = datetime.now(timezone.utc).isoformat()
    valid_df = valid_df.replace(r'^\s*$', pd.NA, regex=True)
    clean_df = valid_df.where(pd.notnull(valid_df), None)
    records = clean_df.to_dict(orient="records")

    try:
        supabase.table(table_name).upsert(records).execute()
    except Exception as e:
        print(f"Supabase upsert failed: {e}")


def run_pipeline(
    input_df: pd.DataFrame,
    output_csv: str = "pipeline_output.csv",
    num_workers: int = 2,
):
    """
    title, artist 컬럼이 있는 DataFrame을 받아 메타데이터 보충 후 track 테이블에 upsert합니다.
    main.py / daily_ranking_batch.py 에서 호출합니다.
    """
    if input_df.empty or "title" not in input_df.columns or "artist" not in input_df.columns:
        print("run_pipeline: input_df에 title, artist 컬럼이 필요합니다.")
        return
    records = [
        {"title": str(row.get("title", "")), "artist": str(row.get("artist", ""))}
        for _, row in input_df.iterrows()
    ]
    if not records:
        return
    token = get_spotify_token()

    def process_one(record):
        return process_track_record(record, token)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_rec = {executor.submit(process_one, rec): rec for rec in records}
        for future in tqdm(concurrent.futures.as_completed(future_to_rec), total=len(future_to_rec), desc="Pipeline"):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Pipeline item error: {e}")

    if not results:
        return
    out_df = pd.DataFrame(results)
    if output_csv:
        out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    upload_to_supabase(out_df, table_name="track", silent=True)


def run_pipeline_from_db(num_workers: int = 5):
    print("\n[Phase 1] 메타데이터 결측치 보충 프로세스 시작")
    db_records = fetch_all_tracks_from_supabase()
    if not db_records:
        return

    target_records = []
    for record in db_records:
        has_missing = any(not record.get(k) for k in [
            "track_id", "popularity", "genre", "mood_tags", "youtube_video_id"
        ])
        if has_missing:
            target_records.append(record)

    if not target_records:
        print("모든 곡의 데이터가 완전히 채워져 있습니다. DB 보충을 건너뜁니다.")
        return

    token = get_spotify_token()
    
    def fetch_data(record):
        return process_track_record(record, token)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_record = {executor.submit(fetch_data, record): record for record in target_records}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_record), total=len(future_to_record), desc="Parallel DB Update"):
            try:
                updated_data = future.result()
                single_df = pd.DataFrame([updated_data])
                upload_to_supabase(single_df, table_name="track", silent=True)
            except Exception as e:
                pass

# ---------------------------------------------------------
# 2. 이미지 Storage 동기화 모듈
# ---------------------------------------------------------
def load_tracker() -> dict:
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_tracker(data: dict):
    with open(TRACKER_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def get_existing_files(bucket_name: str) -> set:
    try:
        files = supabase.storage.from_(bucket_name).list()
        return {f.get("name") for f in files if f.get("name")}
    except Exception as e:
        return set()

def download_and_upload_image(url: str, bucket_name: str, file_name: str):
    if not url:
        return False
        
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image_bytes = response.content
        
        supabase.storage.from_(bucket_name).upload(
            path=file_name,
            file=image_bytes,
            file_options={
                "content-type": "image/jpeg",
                "x-upsert": "true"
            }
        )
        return True
    except Exception as e:
        return False

def sync_images_to_storage():
    print("\n[Phase 2] Storage 이미지 동기화 프로세스 시작")
    try:
        existing_albums = get_existing_files("album_covers")
        artist_tracker = load_tracker()
        
        response = supabase.table("track").select(
            "track_id, artist_id, track_image_url, artist_image_url, release_date"
        ).execute()
        
        tracks = response.data
        if not tracks:
            return

        df = pd.DataFrame(tracks)
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df = df.sort_values(by='release_date', ascending=False)
        
        processed_artists = set()
        tracker_updated = False

        for index, row in tqdm(df.iterrows(), total=len(df), desc="Syncing Images"):
            track_id = row.get("track_id")
            track_url = row.get("track_image_url")
            artist_id = row.get("artist_id")
            artist_url = row.get("artist_image_url")
            
            # 앨범 커버 처리
            if pd.notna(track_id) and pd.notna(track_url):
                track_file_name = f"{track_id}.jpg"
                if track_file_name not in existing_albums:
                    success = download_and_upload_image(track_url, "album_covers", track_file_name)
                    if success:
                        existing_albums.add(track_file_name)

            # 아티스트 이미지 처리
            if pd.notna(artist_id) and pd.notna(artist_url):
                if artist_id not in processed_artists:
                    processed_artists.add(artist_id)
                    artist_file_name = f"{artist_id}.jpg"
                    previous_url = artist_tracker.get(artist_id)
                    
                    if previous_url != artist_url:
                        success = download_and_upload_image(artist_url, "artist_images", artist_file_name)
                        if success:
                            artist_tracker[artist_id] = artist_url
                            tracker_updated = True

        if tracker_updated:
            save_tracker(artist_tracker)

        print("이미지 동기화가 완료되었습니다.")
        
    except Exception as e:
        print(f"동기화 중 오류 발생: {e}")

# ---------------------------------------------------------
# 3. 통합 파이프라인 실행
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. DB 결측치 보충 실행
    run_pipeline_from_db(num_workers=5)
    
    # 2. 업데이트된 DB를 기반으로 이미지 다운로드 및 Storage 업로드 실행
    sync_images_to_storage()