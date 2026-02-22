import argparse
from mixsense_mixing.pipeline import run_mix

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="첫 번째 트랙 zip 경로")
    p.add_argument("--b", required=True, help="두 번째 트랙 zip 경로")
    p.add_argument("--out", default="outputs", help="출력 폴더")
    p.add_argument("--data", default="data", help="data 폴더(버스 캐시 저장 위치)")
    p.add_argument("--dur", type=float, default=150.0, help="목표 길이(초)")
    p.add_argument("--k", type=int, default=5, help="목표 전이 횟수(자동 감소 가능)")
    p.add_argument("--min_gap", type=float, default=24.0, help="전이 최소 간격(초)")
    p.add_argument("--end", type=float, default=26.0, help="엔딩 takeover 길이(초)")
    p.add_argument("--inject_ratio", type=float, default=0.9, help="inject 주입 강도(0~1)")
    args = p.parse_args()

    res = run_mix(
        zip1=args.a,
        zip2=args.b,
        out_dir=args.out,
        data_dir=args.data,
        target_duration=args.dur,
        target_k=args.k,
        min_gap_sec=args.min_gap,
        end_takeover_sec=args.end,
        inject_ratio=args.inject_ratio,
    )
    print("✅ done")
    print("wav:", res.out_wav_path)
    print("log:", res.log_json_path)
    print("backbone:", res.backbone, "inject:", res.inject, "used_k:", res.used_k)

if __name__ == "__main__":
    main()
