-- 실행 환경: Supabase Postgres / 프로젝트 DB / SQL Editor
-- 목적: AI 믹싱 결과 저장(ai_mix) 및 플레이리스트에서 mix 항목 지원

create extension if not exists pgcrypto;

create table if not exists public.ai_mix (
  mix_id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  title text not null,
  mix_audio_url text not null,
  mix_audio_path text,
  log_json_url text,
  log_json_path text,
  source_track_id_1 text references public.track(track_id) on delete set null,
  source_track_id_2 text references public.track(track_id) on delete set null,
  target_duration_sec numeric(8,2),
  used_k integer,
  created_at timestamptz not null default now()
);

create index if not exists idx_ai_mix_user_created_at
  on public.ai_mix (user_id, created_at desc);

alter table public.playlist_tracks
  add column if not exists mix_id uuid;

do $$
begin
  if not exists (
    select 1
    from pg_constraint
    where conname = 'playlist_tracks_mix_id_fkey'
  ) then
    alter table public.playlist_tracks
      add constraint playlist_tracks_mix_id_fkey
      foreign key (mix_id) references public.ai_mix(mix_id) on delete cascade;
  end if;
end $$;

alter table public.playlist_tracks
  drop constraint if exists playlist_tracks_item_check;

alter table public.playlist_tracks
  add constraint playlist_tracks_item_check
  check (
    (track_id is not null and mix_id is null) or
    (track_id is null and mix_id is not null)
  );

-- 기존 데이터에 중복 (playlist_id, track_id) 가 있을 수 있어 unique index 생성 전 정리합니다.
with track_dupes as (
  select
    playlist_track_id,
    row_number() over (
      partition by playlist_id, track_id
      order by added_at asc nulls last, playlist_track_id asc
    ) as rn
  from public.playlist_tracks
  where track_id is not null
)
delete from public.playlist_tracks pt
using track_dupes d
where pt.playlist_track_id = d.playlist_track_id
  and d.rn > 1;

-- 기존 데이터에 중복 (playlist_id, mix_id) 가 있을 수 있어 unique index 생성 전 정리합니다.
with mix_dupes as (
  select
    playlist_track_id,
    row_number() over (
      partition by playlist_id, mix_id
      order by added_at asc nulls last, playlist_track_id asc
    ) as rn
  from public.playlist_tracks
  where mix_id is not null
)
delete from public.playlist_tracks pt
using mix_dupes d
where pt.playlist_track_id = d.playlist_track_id
  and d.rn > 1;

create unique index if not exists uq_playlist_tracks_track
  on public.playlist_tracks (playlist_id, track_id)
  where track_id is not null;

create unique index if not exists uq_playlist_tracks_mix
  on public.playlist_tracks (playlist_id, mix_id)
  where mix_id is not null;


