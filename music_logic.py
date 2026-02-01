from database import supabase

def upsert_track(track_data: dict):
    """곡 정보 저장 또는 업데이트 (track_id 기준)"""
    return supabase.table("Track").upsert(track_data, on_conflict="track_id").execute()

def like_track(user_id: str, track_id: str):
    """곡 좋아요 추가"""
    data = {"user_id": user_id, "track_id": track_id}
    return supabase.table("Likes").insert(data).execute()

def create_playlist(user_id: str, title: str):
    """플레이리스트 생성"""
    data = {"user_id": user_id, "title": title}
    return supabase.table("Playlist").insert(data).execute()

def add_search_history(user_id: str, keyword: str):
    """검색 기록 저장"""
    data = {"user_id": user_id, "keyword": keyword}
    return supabase.table("Search_History").insert(data).execute()