from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import auth_logic
import music_logic

app = FastAPI()

# React(Vite) 개발 서버와의 통신을 위한 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터 모델 정의
class AuthSchema(BaseModel):
    email: str
    password: str

class TrackSchema(BaseModel):
    track_id: str
    track_name: str
    artist_name: str

# API Endpoints
@app.post("/auth/signup")
async def signup(data: AuthSchema):
    res = await auth_logic.sign_up_email(data.email, data.password)
    return res

@app.get("/auth/google")
async def google_login():
    url = await auth_logic.get_google_auth_url()
    return {"url": url}

@app.post("/tracks/upsert")
async def upsert_track(track: TrackSchema):
    res = music_logic.upsert_track(track.dict())
    return {"status": "success", "data": res.data}

@app.post("/likes/{track_id}")
async def like(track_id: str):
    user = auth_logic.get_current_user()
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return music_logic.like_track(user.id, track_id)

# 서버 실행: uvicorn main:app --reload