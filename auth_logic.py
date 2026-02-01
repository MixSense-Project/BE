from database import supabase

async def sign_up_email(email, password):
    """Email 회원가입"""
    return supabase.auth.sign_up({"email": email, "password": password})

async def sign_in_email(email, password):
    """Email 로그인"""
    return supabase.auth.sign_in_with_password({"email": email, "password": password})

async def get_google_auth_url():
    """Google OAuth 로그인 URL 생성"""
    # React/Vite 주소를 redirect_to에 설정합니다.
    res = supabase.auth.sign_in_with_oauth({
        "provider": "google",
        "options": {"redirect_to": "http://localhost:5173"} 
    })
    return res.url

def get_current_user():
    """현재 로그인된 사용자 정보 반환"""
    res = supabase.auth.get_user()
    return res.user if res else None