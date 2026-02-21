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
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
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
                        print(f"경고: {i}위 추출 실패")
                        continue
        
        return pd.DataFrame(tracks_data)
        
    except Exception as e:
        print(f"크롤링 오류: {e}")
        return None
    finally:
        time.sleep(2)
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
                if track is None:
                    continue
                
                main_artist = track['artists'][0]
                all_tracks.append({
                    "track_id": track.get('id'),
                    "title": track.get('name'),          # 파이프라인 호환을 위해 컬럼명 변경
                    "artist": main_artist.get('name'),   # 파이프라인 호환을 위해 컬럼명 변경
                })
        
        df_extended = pd.DataFrame(all_tracks)
        df_merged = pd.merge(df_tracks, df_extended, on='track_id', how='left')
        return df_merged
        
    except Exception as e:
        print(f"API 오류: {e}")
        return None

# ============================================================
# 2. DB 업데이트 모듈
# ============================================================
def update_trending_tracks(df: pd.DataFrame):
    """추출된 Top 50 랭킹 데이터를 Supabase의 Trending_Tracks 테이블에 적재"""
    valid_df = df.dropna(subset=['track_id', 'rank']).copy()
    trending_records = valid_df[['track_id', 'rank']].to_dict(orient="records")
    
    try:
        response = supabase.table("Trending_Tracks").insert(trending_records).execute()
        print(f"Supabase Trending_Tracks 랭킹 업데이트 완료: {len(response.data)}건")
    except Exception as e:
        print(f"Supabase Trending_Tracks 업데이트 실패: {e}")

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

    pipeline_input_df = top50_df[['title', 'artist']].copy()
    
    print(f"3. 데이터 파이프라인 실행 (public.track 적재)...")
    run_pipeline(
        input_df=pipeline_input_df, 
        output_csv="daily_top50_metadata.csv", 
        num_workers=2
    )
    
    print("4. 일일 랭킹 데이터(Trending_Tracks) 기록...")
    update_trending_tracks(top50_df)

if __name__ == "__main__":
    run_daily_top50_update()