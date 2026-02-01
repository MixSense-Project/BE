import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# 전역 Supabase 클라이언트
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)