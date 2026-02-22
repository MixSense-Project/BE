# data 폴더 (로컬 전용)

## inputs
- data/inputs/zips/ : 원본 stem zip (필수 입력)
- data/inputs/audio/ : (옵션) mix-only 오디오

## processed
- data/processed/bus/<track_id>/ : 파이프라인이 자동 생성한 bus 캐시
  - drum_bus.wav / bass_bus.wav / harmony_bus.wav / mix_bus.wav / meta.json

## cache
- data/cache/features/ : bpm/beat/bar 후보 등 캐시

주의: data 폴더는 GitHub에 올리지 않습니다(.gitignore).
