import os
import re
from dotenv import load_dotenv
import pandas as pd
from supabase import create_client, Client

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 한글 초성 리스트
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

def get_index_char(name: str) -> str:
    """아티스트 이름의 첫 글자를 기반으로 초성 또는 알파벳 인덱스를 추출합니다."""
    if not name:
        return '#'
    
    first_char = name.strip()[0]
    
    # 1. 한글인 경우 (유니코드 계산)
    if re.match(r'[가-힣]', first_char):
        char_code = ord(first_char) - 44032
        chosung_index = char_code // 588
        return CHOSUNG_LIST[chosung_index]
    
    # 2. 영문인 경우 (대문자로 변환)
    elif re.match(r'[a-zA-Z]', first_char):
        return first_char.upper()
    
    # 3. 숫자 및 특수기호 등 그 외
    else:
        return '#'

def process_artists_data():
    """track 테이블에서 데이터를 가져와 아티스트별로 그룹화하고 DB에 적재합니다."""
    # 1. track 테이블에서 필요한 데이터 전체 조회
    # 실제 운영 시에는 페이징 처리가 필요할 수 있습니다.
    response = supabase.table("track").select("artist_id, artist, genre, artist_image_url").execute()
    tracks = response.data
    
    if not tracks:
        print("트랙 데이터가 없습니다.")
        return

    df = pd.DataFrame(tracks)
    
    # 결측치 처리
    df['artist_id'] = df['artist_id'].fillna('unknown_id')
    df['artist'] = df['artist'].fillna('Unknown Artist')
    df['genre'] = df['genre'].fillna('Unknown')
    
    # 2. 아티스트별 데이터 집계
    artist_records = []
    grouped = df.groupby('artist_id')
    
    for artist_id, group in grouped:
        artist_name = group['artist'].iloc[0]
        artist_image_url = group['artist_image_url'].dropna().iloc[0] if not group['artist_image_url'].dropna().empty else None
        
        # 중복 제거된 장르 리스트 생성 (다중 장르)
        unique_genres = list(set(group['genre'].tolist()))
        
        # 초성 추출
        index_char = get_index_char(artist_name)
        
        artist_records.append({
            "artist_id": artist_id,
            "name": artist_name,
            "genres": unique_genres, # Postgres Array로 저장됨
            "index_char": index_char,
            "image_url": artist_image_url
        })
    
    # 3. artist 테이블에 Upsert
    artist_df = pd.DataFrame(artist_records)
    
    # name 컬럼 기준으로 가나다순 정렬 (미리 정렬해두면 DB 조회 시 유리함)
    artist_df = artist_df.sort_values(by='name')
    
    records_to_insert = artist_df.to_dict(orient="records")
    
    try:
        supabase.table("artist").upsert(records_to_insert).execute()
        print(f"총 {len(records_to_insert)}명의 아티스트 데이터 처리가 완료되었습니다.")
    except Exception as e:
        print(f"DB Upsert 중 오류 발생: {e}")

if __name__ == "__main__":
    process_artists_data()