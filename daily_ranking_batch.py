import os
import time
import pandas as pd
from dotenv import load_dotenv

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# 기존 파이프라인 모듈 임포트
from track_data_pipeline import run_pipeline, supabase

# ============================================================
# 0. 설정
# ============================================================
load_dotenv()
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
PLAYLIST_URL = "https://open.spotify.com/playlist/37i9dQZEVXbMDoHDwVN2tF"

# ============================================================
# 1. 크롤링 및 Spotify API 조회 모듈
# ============================================================
def crawl_playlist_tracks(url):
    """Selenium으로 플레이리스트 페이지에서 순위와 track_id 추출"""
    chrome_options = Options()
    #chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    chrome_options.add_argument("--log-level=3")  # Fatal 에러 외 로그 무시
    chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"]) # 로그 출력 제외
    chrome_options.add_argument("--disable-blink-features=AutomationControlled") # 자동화 도구 감지 우회
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        driver.get(url)
        wait = WebDriverWait(driver, 25)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "main")))
        time.sleep(5)
        
        tracks_data = []
        
        for i in range(1, 51):
            max_retries = 3
            for retry in range(max_retries):
                try:
                    ranking_xpath = f"/html/body/div[4]/div/div[2]/div[6]/div/div[2]/div[1]/div/main/section/div[2]/div[3]/div/div[1]/div/div[2]/div[2]/div[{i}]/div/div[1]/div/div/span"
                    link_xpath = f"/html/body/div[4]/div/div[2]/div[6]/div/div[2]/div[1]/div/main/section/div[2]/div[3]/div/div[1]/div/div[2]/div[2]/div[{i}]/div/div[2]/div/a"
                    
                    if i == 4 or retry > 0:
                        driver.execute_script("window.scrollBy(0, 300);")
                        time.sleep(0.5)
                    
                    wait.until(EC.presence_of_element_located((By.XPATH, ranking_xpath)))
                    rank_element = driver.find_element(By.XPATH, ranking_xpath)
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center', behavior: 'smooth'});", rank_element)
                    time.sleep(0.5)
                    
                    link_element = driver.find_element(By.XPATH, link_xpath)
                    rank_text = rank_element.text
                    full_url = link_element.get_attribute("href")
                    
                    if not full_url:
                        raise Exception("URL이 비어있음")
                    
                    track_id = full_url.split("/")[-1].split("?")[0]
                    
                    tracks_data.append({
                        "rank": int(rank_text),
                        "track_id": track_id
                    })
                    break
                    
                except Exception:
                    if retry < max_retries - 1:
                        driver.execute_script("window.scrollBy(0, 400);")
                        time.sleep(0.8)
                    else:
                        continue
        
        return pd.DataFrame(tracks_data)
        
    except Exception as e:
        print(f"크롤링 오류: {e}")
        return None
    finally:
        driver.quit()

def enrich_track_data(df_tracks):
    """Spotify API로 트랙 상세 정보 가져오기"""
    auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    track_ids = df_tracks['track_id'].tolist()
    
    try:
        all_tracks = []
        for i in range(0, len(track_ids), 50):
            batch = track_ids[i:i+50]
            full_tracks = sp.tracks(batch)
            for track in full_tracks['tracks']:
                if track is None: continue
                main_artist = track['artists'][0]
                all_tracks.append({
                    "track_id": track.get('id'),
                    "title": track.get('name'),
                    "artist": main_artist.get('name'),
                })
        
        df_extended = pd.DataFrame(all_tracks)
        # 중복 제거: 한 리스트에 같은 track_id가 있을 경우 Upsert 에러 방지
        df_extended = df_extended.drop_duplicates(subset=['track_id'])
        df_merged = pd.merge(df_tracks, df_extended, on='track_id', how='left')
        return df_merged
        
    except Exception as e:
        print(f"API 오류: {e}")
        return None

# ============================================================
# 2. DB 업데이트 모듈
# ============================================================
def ensure_basic_tracks_in_db(df: pd.DataFrame):
    """외래키(Foreign Key) 에러 방지를 위해 track_id를 track 테이블에 미리 적재합니다."""
    valid_df = df.dropna(subset=['track_id', 'title', 'artist']).drop_duplicates(subset=['track_id'])
    if valid_df.empty:
        return
    records = valid_df[['track_id', 'title', 'artist']].to_dict(orient="records")
    try:
        supabase.table("track").upsert(records).execute()
        print(f"외래키 방지용 기초 트랙 데이터 적재 완료: {len(records)}건")
    except Exception as e:
        print(f"기초 트랙 데이터 적재 실패: {e}")


def update_trending_tracks(df: pd.DataFrame):
    """추출된 Top 50 랭킹 데이터를 Supabase의 trending_tracks 테이블에 적재"""
    # 테이블명을 소문자로 수정
    valid_df = df.dropna(subset=['track_id', 'rank']).copy()
    # 랭킹 데이터 역시 중복된 track_id가 있을 경우 제거
    valid_df = valid_df.drop_duplicates(subset=['track_id'])
    trending_records = valid_df[['track_id', 'rank']].to_dict(orient="records")
    
    try:
        # 소문자 테이블명 사용
        response = supabase.table("trending_tracks").insert(trending_records).execute()
        print(f"Supabase trending_tracks 랭킹 업데이트 완료: {len(response.data)}건")
    except Exception as e:
        print(f"Supabase trending_tracks 업데이트 실패: {e}")

# ============================================================
# 3. 통합 실행
# ============================================================
def run_daily_top50_update():
    """크롤링, 파이프라인 적재, 랭킹 업데이트를 순차적으로 실행"""
    print("1. 플레이리스트 크롤링 시작...")
    df_tracks = crawl_playlist_tracks(PLAYLIST_URL)
    
    if df_tracks is None or df_tracks.empty:
        print("크롤링 데이터가 없습니다.")
        return
        
    print(f"2. {len(df_tracks)}개 트랙 API 메타데이터 매핑 중...")
    top50_df = enrich_track_data(df_tracks)
    
    if top50_df is None or top50_df.empty:
        print("메타데이터 매핑 실패")
        return

    # 파이프라인 실행 전 중복 제거 확인
    pipeline_input_df = top50_df[['title', 'artist']].drop_duplicates().copy()
    
    print(f"3. 데이터 파이프라인 실행 (public.track 적재)...")
    try:
        run_pipeline(
            input_df=pipeline_input_df, 
            output_csv="daily_top50_metadata.csv", 
            num_workers=2
        )
    except Exception as e:
        print(f"파이프라인 실행 중 오류: {e}")

    print("3.5. 외래키 제약조건 방지 처리...")
    ensure_basic_tracks_in_db(top50_df)

    print("4. 일일 랭킹 데이터(trending_tracks) 기록...")
    update_trending_tracks(top50_df)

if __name__ == "__main__":
    run_daily_top50_update()