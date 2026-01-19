import os
import time
import schedule
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Selenium 관련
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Spotify & DB 관련
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from supabase import create_client, Client

# ============================================================
# 1. 설정 및 초기화
# ============================================================
load_dotenv()

# 환경 변수 (client_id, secret 등은 .env 파일에서 로드)
SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# 크롤링 타겟 URL (코드에 있던 URL 사용)
TARGET_PLAYLIST_URL = "https://open.spotify.com/playlist/37i9dQZEVXbMDoHDwVN2tF"

# 클라이언트 초기화
auth_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================================================
# 2. 크롤링 함수 (제공해주신 검증된 코드)
# ============================================================
def crawl_playlist_tracks(url):
    """Selenium으로 플레이리스트 페이지에서 순위와 track_id 추출"""
    print("🕷️ Selenium 크롤링 시작...")
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # 브라우저 창 숨기기
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
            success = False
            
            for retry in range(max_retries):
                try:
                    ranking_xpath = f"/html/body/div[4]/div/div[2]/div[6]/div/div[2]/div[1]/div/main/section/div[2]/div[3]/div/div[1]/div/div[2]/div[2]/div[{i}]/div/div[1]/div/div/span"
                    link_xpath = f"/html/body/div[4]/div/div[2]/div[6]/div/div[2]/div[1]/div/main/section/div[2]/div[3]/div/div[1]/div/div[2]/div[2]/div[{i}]/div/div[2]/div/a"
                    
                    # 요소가 보일 때까지 대기 (특히 4위 같은 경우)
                    if i == 4 or retry > 0:
                        driver.execute_script("window.scrollBy(0, 300);")
                        time.sleep(0.5)
                    
                    # 요소 찾기 전에 명시적 대기
                    wait.until(EC.presence_of_element_located((By.XPATH, ranking_xpath)))
                    
                    rank_element = driver.find_element(By.XPATH, ranking_xpath)
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center', behavior: 'smooth'});", rank_element)
                    time.sleep(0.5)  # 스크롤 후 대기 시간 증가
                    
                    link_element = driver.find_element(By.XPATH, link_xpath)
                    rank_text = rank_element.text
                    full_url = link_element.get_attribute("href")
                    
                    if not full_url:
                        raise Exception("URL이 비어있음")
                    
                    track_id = full_url.split("/")[-1].split("?")[0]
                    
                    tracks_data.append({
                        "rank": rank_text,
                        "track_id": track_id
                    })
                    
                    success = True
                    break
                    
                except Exception as e:
                    if retry < max_retries - 1:
                        # 재시도 전에 더 많이 스크롤
                        driver.execute_script("window.scrollBy(0, 400);")
                        time.sleep(0.8)
                    else:
                        # 마지막 시도 실패 시에도 계속 진행
                        print(f"경고: {i}위 추출 실패 (재시도 {max_retries}회 실패)")
                        continue
        
        return pd.DataFrame(tracks_data)
        
    except Exception as e:
        print(f"크롤링 오류: {e}")
        return None
    finally:
        time.sleep(2)
        driver.quit()

# ============================================================
# 3. 데이터 병합 및 DB 저장 로직 (통합 부분)
# ============================================================
def process_and_save_daily():
    # 1. 크롤링 (기존 코드 사용)
    df_tracks = crawl_playlist_tracks(TARGET_PLAYLIST_URL)
    
    if df_tracks is None or df_tracks.empty:
        print("❌ 수집된 데이터가 없습니다.")
        return

    print(f"✅ 크롤링 완료: {len(df_tracks)}개 트랙 ID 확보")
    
    # 2. Spotify API 조회
    track_ids = df_tracks['track_id'].tolist()
    full_tracks_info = []
    
    for i in range(0, len(track_ids), 50):
        batch = track_ids[i:i+50]
        try:
            response = sp.tracks(batch)
            full_tracks_info.extend(response['tracks'])
        except Exception as e:
            print(f"API 호출 오류 (Batch {i}): {e}")

    # 3. 데이터 매핑
    trending_payload = []
    track_payload = []
    current_time = datetime.now().isoformat()
    
    # 순위 매핑용 딕셔너리
    rank_map = {}
    for _, row in df_tracks.iterrows():
        try:
            rank_val = int(row['rank'])
        except:
            rank_val = row['rank']
        rank_map[row['track_id']] = rank_val
    
    for t in full_tracks_info:
        if t is None: continue
        
        t_id = t['id']
        if t_id not in rank_map: continue
        
        rank = rank_map[t_id]
        main_artist = t['artists'][0]
        img_url = t['album']['images'][0]['url'] if t['album']['images'] else None
        
        # [NEW] Track 테이블 Payload (아티스트 정보 포함)
        track_payload.append({
            "track_id": t_id,
            "track_name": t['name'],              # 컬럼명 주의
            "artist_id": main_artist['id'],       # FK 아님, 단순 텍스트
            "artist_name": main_artist['name'],   # 필수 정보
            "album_id": t['album']['id'],
            "popularity": t['popularity'],
            "image_url": img_url,                 # 컬럼명 주의
            "release_date": t['album']['release_date']
        })
        
        # Trending 테이블 Payload (artist_id 불필요)
        trending_payload.append({
            "track_id": t_id,
            "rank": rank,
            "crawled_at": current_time
        })
        
    # 4. DB 저장 (순서: Track -> Trending)
    try:
        # Step 1: Track 정보 저장 (Upsert)
        if track_payload:
            supabase.table('track').upsert(
                track_payload, 
                on_conflict='track_id', 
                ignore_duplicates=False # 정보 갱신을 위해 False 추천 (인기도 등 변할 수 있음)
            ).execute()
            print(f"💾 곡 정보 저장 완료: {len(track_payload)}개")

        # Step 2: 순위 정보 저장 (Insert)
        if trending_payload:
            supabase.table('trending_tracks').insert(trending_payload).execute()
            print(f"📊 랭킹 정보 저장 완료: {len(trending_payload)}개")
            
    except Exception as e:
        print(f"❌ DB 저장 실패: {e}")

# ============================================================
# 4. 스케줄러 실행
# ============================================================
if __name__ == "__main__":
    print("🚀 스케줄러 시작. 매일 자정에 실행됩니다.")
    
    # [테스트용] 앱 실행 시 1회 즉시 작동 (잘 되는지 확인하려면 주석 해제 유지)
    process_and_save_daily()
    
    # 매일 00:00에 실행 예약
    schedule.every().day.at("00:00").do(process_and_save_daily)
    
    while True:
        schedule.run_pending()
        time.sleep(60)
