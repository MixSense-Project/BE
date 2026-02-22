import os
import pywhatkit
import webbrowser
from dotenv import load_dotenv
from supabase import create_client, Client

# 환경 변수 로드 및 Supabase 클라이언트 설정
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def play_songs_from_supabase():
    try:
        # 1. Supabase에서 Track 테이블 데이터 조회
        print("Supabase에서 데이터를 불러오는 중...")
        response = supabase.table("Track").select("title, artist, youtube_video_id").execute()
        tracks = response.data
        
        if not tracks:
            print("재생할 곡 데이터가 없습니다.")
            return

        print(f"총 {len(tracks)}곡의 데이터를 불러왔습니다.")

        # 2. 데이터 순회 및 2단계 재생 로직 적용
        for index, row in enumerate(tracks):
            song_title = row.get('title')
            artist_name = row.get('artist')
            video_id = row.get('youtube_video_id')
            
            print(f"\n[{index + 1}] Target: {artist_name} - {song_title}")
            
            # Step 1: youtube_video_id가 유효한지 확인 (None 또는 빈 문자열 검사)
            if video_id and str(video_id).strip():
                print(f"-> Step 1: Found youtube_video_id ({video_id}). Playing direct URL.")
                target_url = f"https://www.youtube.com/watch?v={video_id}"
                webbrowser.open(target_url)
            
            # Step 2: youtube_video_id가 없는 경우 검색으로 대체
            else:
                search_query = f"{artist_name} - {song_title}"
                print(f"-> Step 2: No video ID. Searching and playing: {search_query}")
                pywhatkit.playonyt(search_query)
            
            # 다음 곡 재생 대기 및 종료 처리
            user_input = input("Press Enter to play the next song (or type 'q' to quit): ")
            if user_input.lower() == 'q':
                print("Playback terminated.")
                break
                
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    play_songs_from_supabase()