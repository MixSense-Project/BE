import os
import time
import requests
import pandas as pd
from tqdm import tqdm

# 1. 토큰 발급 함수
def get_spotify_access_token(client_id, client_secret):
    """
    Spotify API 액세스 토큰을 발급받습니다.
    """
    auth_url = "https://accounts.spotify.com/api/token"
    
    # 헤더와 데이터 설정
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }

    try:
        response = requests.post(auth_url, headers=headers, data=data, timeout=10)
        response.raise_for_status()
        
        token_info = response.json()
        return token_info.get("access_token")
        
    except requests.exceptions.RequestException as e:
        print(f"Token generation failed: {e}")
        return None

# 2. 단일 트랙 ID 검색 함수
def get_track_id(track_name, artist_name, access_token):
    """
    트랙명과 아티스트명으로 Spotify Track ID를 검색합니다.
    """
    if not track_name or not artist_name:
        return None
        
    search_url = "https://api.spotify.com/v1/search"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    
    # 검색 정확도를 위해 query 구성
    query = f"track:{str(track_name)} artist:{str(artist_name)}"
    
    params = {
        "q": query,
        "type": "track",
        "limit": 1
    }

    try:
        response = requests.get(search_url, headers=headers, params=params, timeout=5)
        
        # Rate Limit(429) 처리
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 1))
            print(f"Rate limited. Sleeping for {retry_after} seconds.")
            time.sleep(retry_after)
            return get_track_id(track_name, artist_name, access_token) # 재귀 호출
            
        response.raise_for_status()
        
        data = response.json()
        items = data.get("tracks", {}).get("items", [])
        
        if items:
            return items[0]["id"]
        else:
            return None
            
    except Exception as e:
        # 개별 검색 실패 시 전체 중단 방지
        # print(f"Search failed for {track_name}: {e}") 
        return None

# 3. 메인 실행 함수 (데이터프레임 처리)
def append_spotify_ids(df, client_id, client_secret, track_col='track_name', artist_col='artist_name'):
    """
    데이터프레임을 받아 Spotify ID를 조회하고, 결과를 포함한 새 데이터프레임을 반환합니다.
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        client_id (str): Spotify Client ID
        client_secret (str): Spotify Client Secret
        track_col (str): 트랙명이 있는 컬럼 이름 (기본값: 'track_name')
        artist_col (str): 아티스트명이 있는 컬럼 이름 (기본값: 'artist_name')
        
    Returns:
        pd.DataFrame: 'spotify_track_id' 컬럼이 추가된 데이터프레임
    """
    # 1. 토큰 발급
    print("🔑 Authenticating with Spotify...")
    token = get_spotify_access_token(client_id, client_secret)
    
    if not token:
        raise ValueError("Failed to retrieve access token. Check your Client ID and Secret.")
    
    print("✅ Token received. Starting search...")
    
    # 2. 복사본 생성 (원본 보존)
    result_df = df.copy()
    spotify_ids = []
    
    # 3. tqdm을 사용한 진행 상황 표시
    # apply 대신 for문을 사용하여 진행률을 시각화합니다.
    for _, row in tqdm(result_df.iterrows(), total=len(result_df), desc="Fetching Spotify IDs"):
        track = row.get(track_col)
        artist = row.get(artist_col)
        
        tid = get_track_id(track, artist, token)
        spotify_ids.append(tid)
        
        # API 부하 조절을 위한 아주 짧은 대기 (선택 사항)
        time.sleep(0.05)
        
    # 4. 결과 컬럼 추가
    result_df['spotify_track_id'] = spotify_ids
    
    found_count = result_df['spotify_track_id'].notna().sum()
    print(f"\n🎉 완료! 총 {len(result_df)}개 중 {found_count}개의 ID를 찾았습니다.")
    
    return result_df