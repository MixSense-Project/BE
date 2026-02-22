import re
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from datetime import datetime

# =========================
# PATH
# =========================
PATH_META = "./data/raw/meta_data_ppc.csv"
PATH_SEEN = "./data/raw/test_ind.csv"
PATH_LOG  = "./data/raw/StreamingHistory_music_0.csv"

CSV_ENCODING = "utf-8"
CSV_ERRORS = "replace"

# =========================
# CONFIG
# =========================
MIN_MS_PLAYED = 60_000
SESSION_GAP_MINUTES = 30
TRAIN_RATIO = 0.8

HOME_TOTAL = 18
HOME_BUCKETS = {"repeat": 4, "like_similar": 6, "discovery": 8}

# Home 18: 아티스트 최대 1곡(전체)
MAX_PER_ARTIST_HOME_ALL = 1

# Autoplay: 아티스트 최대 1곡
MAX_PER_ARTIST_AUTOPLAY = 1

# Discovery (확장 강하게)
W_ARTIST = 0.35
W_POP = 0.40
W_RECENCY = 0.25
TOP_ARTISTS_FOR_POOL = 60
GLOBAL_FALLBACK_CANDIDATES = 5000
DISCOVERY_POOL_LIMIT = 3000

# 시즌곡 페널티
SEASONAL_KEYWORDS = ["christmas", "xmas", "santa", "snow", "holiday", "last christmas", "all i want for christmas"]
SEASONAL_PENALTY = 0.15

# 로그 -> track_id 매칭(정규화+fuzzy)
FUZZY_MIN_RATIO = 0.86
FUZZY_MAX_PER_LOG = 5

# Autoplay: Markov 우선 보장 개수
AUTOPLAY_MARKOV_MIN = 5

# ✅ 최종: Markov 게이팅 threshold
MARKOV_GATE_THRESHOLD = 5  # outgoing_sum >= 5 일 때만 Markov 중심

# Markov-strong seed 선택 기준(최소 전이 수)
MARKOV_STRONG_MIN_OUTGOING = 5

# =========================
# UTILS
# =========================
_feat_pat = re.compile(r"\b(feat\.?|ft\.?|featuring|with)\b.*$", flags=re.IGNORECASE)
_bracket_pat = re.compile(r"[\(\[\{].*?[\)\]\}]")
_non_alnum_pat = re.compile(r"[^a-z0-9가-힣]+")

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = _feat_pat.sub("", s)
    s = _bracket_pat.sub("", s)
    s = _non_alnum_pat.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def contains_seasonal(title: str) -> bool:
    t = (title or "").lower()
    return any(k in t for k in SEASONAL_KEYWORDS)

def norm01(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mn, mx = s.min(), s.max()
    if np.isclose(mx, mn):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)

def safe_int64_datetime(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    ts = dt.astype("int64", errors="ignore")
    if ts.dtype == "O":
        ts = pd.to_datetime(series, errors="coerce").astype("int64")
    ts = pd.Series(ts, index=series.index)
    ts = ts.replace(-9223372036854775808, np.nan)
    ts = ts.fillna(ts.min())
    return ts

def read_csv_safe(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding=CSV_ENCODING, encoding_errors=CSV_ERRORS)

def dedup_preserve_order(items):
    out = []
    seen = set()
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def apply_artist_cap(track_ids, meta_all: pd.DataFrame, max_per_artist: int):
    m = meta_all.set_index("track_id")[["artist_id", "artist"]].to_dict(orient="index")
    out = []
    cnt = Counter()
    for tid in track_ids:
        info = m.get(tid, {})
        aid = info.get("artist_id", None)
        if aid is None or pd.isna(aid):
            aid = info.get("artist", "UNKNOWN")
        if cnt[aid] >= max_per_artist:
            continue
        out.append(tid)
        cnt[aid] += 1
    return out

def print_markov_debug(current_tid, trans, meta_all, topn=10):
    if current_tid in trans and len(trans[current_tid]) > 0:
        print("\n[DEBUG] Top Markov transitions from seed:")
        for tid, cnt in trans[current_tid].most_common(topn):
            row = meta_all[meta_all["track_id"] == tid]
            if len(row) > 0:
                row = row.iloc[0]
                print(f"- {row['artist']} - {row['title']} (count={cnt})")
            else:
                print(f"- {tid} (count={cnt})")
    else:
        print("\n[DEBUG] No Markov transitions for this seed (fallback will dominate)")

def show_reco_list(title, ids, meta_all):
    print(title)
    rec_df = meta_all[meta_all["track_id"].isin(ids)][["track_id", "artist", "title", "popularity"]].copy()
    order = {tid: i for i, tid in enumerate(ids)}
    rec_df["ord"] = rec_df["track_id"].map(order)
    rec_df = rec_df.sort_values("ord")
    for i, row in enumerate(rec_df.itertuples(index=False), 1):
        print(f"{i:02d}. {row.artist} - {row.title} (pop={row.popularity})")

# =========================
# LOAD
# =========================
def load_all():
    meta = read_csv_safe(PATH_META)
    seen = read_csv_safe(PATH_SEEN)
    log = read_csv_safe(PATH_LOG)

    for df in (meta, seen):
        df["track_id"] = df["track_id"].astype(str)
        df["artist_id"] = df["artist_id"].astype(str)
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["artist"] = df["artist"].astype(str)
        df["title"] = df["title"].astype(str)
        df["artist_norm"] = df["artist"].map(normalize_text)
        df["title_norm"] = df["title"].map(normalize_text)

    log["endTime"] = pd.to_datetime(log["endTime"], errors="coerce")
    log = log.dropna(subset=["endTime"])
    log["msPlayed"] = pd.to_numeric(log["msPlayed"], errors="coerce").fillna(0).astype(int)
    log = log[log["msPlayed"] >= MIN_MS_PLAYED].copy()

    log["artist_raw"] = log["artistName"].astype(str)
    log["title_raw"] = log["trackName"].astype(str)
    log["artist_norm"] = log["artist_raw"].map(normalize_text)
    log["title_norm"] = log["title_raw"].map(normalize_text)

    return meta, seen, log

# =========================
# MAP log -> track_id
# =========================
def build_track_index(meta_all: pd.DataFrame):
    idx_df = meta_all.drop_duplicates(subset=["artist_norm", "title_norm"])
    exact_map = dict(zip(zip(idx_df["artist_norm"], idx_df["title_norm"]), idx_df["track_id"]))

    by_artist = defaultdict(list)
    for r in idx_df[["artist_norm", "title_norm", "track_id"]].itertuples(index=False):
        by_artist[r.artist_norm].append((r.title_norm, r.track_id))
    return exact_map, by_artist

def map_log_to_track_id(log: pd.DataFrame, exact_map, by_artist):
    keys = list(zip(log["artist_norm"], log["title_norm"]))
    log = log.copy()
    log["track_id"] = [exact_map.get(k, None) for k in keys]
    exact_hit = log["track_id"].notna().sum()

    unmatched = log.index[log["track_id"].isna()].tolist()
    fuzzy_hit = 0
    for idx in unmatched:
        a = log.at[idx, "artist_norm"]
        t = log.at[idx, "title_norm"]
        cand = by_artist.get(a, [])
        if not cand:
            continue

        cand_titles = cand
        if len(cand_titles) > FUZZY_MAX_PER_LOG:
            cand_titles = sorted(cand, key=lambda x: abs(len(x[0]) - len(t)))[:FUZZY_MAX_PER_LOG]

        best_r = 0.0
        best_tid = None
        for cand_title, tid in cand_titles:
            r = similarity(t, cand_title)
            if r > best_r:
                best_r = r
                best_tid = tid

        if best_tid and best_r >= FUZZY_MIN_RATIO:
            log.at[idx, "track_id"] = best_tid
            fuzzy_hit += 1

    before = len(log)
    log = log.dropna(subset=["track_id"]).copy()
    after = len(log)
    print(f"[mapping] exact_hit={exact_hit}, fuzzy_hit={fuzzy_hit}")
    print(f"[mapping] log rows: {before} -> matched rows: {after} (dropped {before-after})")
    return log

# =========================
# PROFILE
# =========================
def build_profile(seen: pd.DataFrame) -> dict:
    return {"artist_freq": seen["artist_id"].value_counts()}

# =========================
# HOME 18
# =========================
def score_pool(pool: pd.DataFrame, profile: dict) -> pd.DataFrame:
    artist_score = profile["artist_freq"].to_dict()
    pool = pool.copy()
    pool["artist_pref"] = pool["artist_id"].map(artist_score).fillna(0)
    pool["artist_norm2"] = norm01(pool["artist_pref"].fillna(0))
    pool["pop_norm"] = norm01(pool["popularity"].fillna(0))
    pool["rec_ts"] = safe_int64_datetime(pool["release_date"])
    pool["rec_norm"] = norm01(pool["rec_ts"].fillna(0))
    pool["score"] = W_ARTIST * pool["artist_norm2"] + W_POP * pool["pop_norm"] + W_RECENCY * pool["rec_norm"]
    pool["seasonal"] = pool["title"].map(lambda x: contains_seasonal(str(x)))
    pool.loc[pool["seasonal"], "score"] *= SEASONAL_PENALTY
    return pool

def pick_with_global_artist_cap(pool: pd.DataFrame, picked_track_ids: set, artist_cnt: Counter, k: int):
    out_rows = []
    for _, row in pool.iterrows():
        tid = row["track_id"]
        if tid in picked_track_ids:
            continue
        aid = row.get("artist_id", None)
        if aid is None or pd.isna(aid):
            aid = row.get("artist", "UNKNOWN")
        if artist_cnt[aid] >= MAX_PER_ARTIST_HOME_ALL:
            continue

        out_rows.append(row)
        picked_track_ids.add(tid)
        artist_cnt[aid] += 1
        if len(out_rows) >= k:
            break
    return pd.DataFrame(out_rows)

def build_home18(meta: pd.DataFrame, seen: pd.DataFrame, profile: dict):
    seen_ids = set(seen["track_id"].tolist())
    picked_track_ids = set()
    artist_cnt = Counter()

    # Repeat
    rep = seen[["track_id", "title", "artist", "artist_id", "popularity", "release_date", "youtube_video_id"]].copy()
    rep["pop_norm"] = norm01(rep["popularity"].fillna(0))
    rep["rec_ts"] = safe_int64_datetime(rep["release_date"])
    rep["rec_norm"] = norm01(rep["rec_ts"])
    rep["score"] = 0.55 * rep["rec_norm"] + 0.45 * rep["pop_norm"]
    rep = rep.sort_values("score", ascending=False)
    rep_pick = pick_with_global_artist_cap(rep, picked_track_ids, artist_cnt, HOME_BUCKETS["repeat"]).assign(bucket="repeat")

    # LikeSimilar
    top_artists = profile["artist_freq"].head(TOP_ARTISTS_FOR_POOL).index.tolist()
    like_pool = meta[(meta["artist_id"].isin(top_artists)) & (~meta["track_id"].isin(seen_ids))].copy()
    like_pool = score_pool(like_pool, profile).sort_values("score", ascending=False)
    like_pick = pick_with_global_artist_cap(like_pool, picked_track_ids, artist_cnt, HOME_BUCKETS["like_similar"]).assign(bucket="like_similar")

    # Discovery
    pool1 = meta[(meta["artist_id"].isin(top_artists)) & (~meta["track_id"].isin(seen_ids))].copy()
    pool2 = meta[(~meta["track_id"].isin(seen_ids))].copy()
    pool2 = pool2.sort_values("popularity", ascending=False).head(GLOBAL_FALLBACK_CANDIDATES)
    disc_pool = pd.concat([pool1, pool2], ignore_index=True).drop_duplicates("track_id")
    disc_pool = score_pool(disc_pool, profile).sort_values("score", ascending=False).head(DISCOVERY_POOL_LIMIT)
    disc_pick = pick_with_global_artist_cap(disc_pool, picked_track_ids, artist_cnt, HOME_BUCKETS["discovery"]).assign(bucket="discovery")

    home_df = pd.concat([rep_pick, like_pick, disc_pick], ignore_index=True)

    if len(home_df) < HOME_TOTAL:
        need = HOME_TOTAL - len(home_df)
        extra_pool = meta[(~meta["track_id"].isin(seen_ids))].copy()
        extra_pool = score_pool(extra_pool, profile).sort_values("score", ascending=False)
        extra_pick = pick_with_global_artist_cap(extra_pool, picked_track_ids, artist_cnt, need).assign(bucket="discovery_fill")
        home_df = pd.concat([home_df, extra_pick], ignore_index=True)

    home_df = home_df.head(HOME_TOTAL).copy()

    home = []
    for _, r in home_df.iterrows():
        home.append({
            "bucket": r["bucket"],
            "track_id": r["track_id"],
            "artist": r["artist"],
            "title": r["title"],
            "popularity": float(r.get("popularity", 0)),
            "youtube_video_id": r.get("youtube_video_id", None),
        })
    return home

# =========================
# AUTOPLAY (Markov + fallback + gating)
# =========================
def sessions_from_log(log_mapped: pd.DataFrame, gap_minutes: int = 30):
    df = log_mapped.sort_values("endTime").copy()
    gap = df["endTime"].diff().dt.total_seconds().fillna(0)
    br = gap >= (gap_minutes * 60)

    sessions = []
    cur = []
    for is_break, tid in zip(br.tolist(), df["track_id"].tolist()):
        if is_break and len(cur) > 0:
            if len(cur) >= 2:
                sessions.append(cur)
            cur = []
        cur.append(tid)
    if len(cur) >= 2:
        sessions.append(cur)
    return sessions

def time_split_sessions(sessions, train_ratio=0.8):
    total = sum(len(s) for s in sessions)
    cut = int(total * train_ratio)
    tr, te = [], []
    seen = 0
    for s in sessions:
        if seen >= cut:
            te.append(s); continue
        if seen + len(s) <= cut:
            tr.append(s); seen += len(s)
        else:
            idx = cut - seen
            left, right = s[:idx], s[idx:]
            if len(left) >= 2: tr.append(left)
            if len(right) >= 2: te.append(right)
            seen = cut
    return tr, te

def train_markov(train_sessions):
    trans = defaultdict(Counter)
    for s in train_sessions:
        for a, b in zip(s[:-1], s[1:]):
            trans[a][b] += 1
    return trans

def outgoing_sum(counter: Counter) -> int:
    return sum(counter.values())

def select_markov_strong_seed(trans: dict, min_outgoing: int):
    candidates = [(tid, outgoing_sum(cnt)) for tid, cnt in trans.items() if outgoing_sum(cnt) >= min_outgoing]
    if not candidates:
        return None, 0
    tid, score = sorted(candidates, key=lambda x: x[1], reverse=True)[0]
    return tid, score

def build_fallback_pool(meta_all: pd.DataFrame, seen_ids: set, profile: dict, limit: int = 3000):
    top_artists = profile["artist_freq"].head(TOP_ARTISTS_FOR_POOL).index.tolist()
    pool1 = meta_all[(meta_all["artist_id"].isin(top_artists)) & (~meta_all["track_id"].isin(seen_ids))].copy()
    pool2 = meta_all[(~meta_all["track_id"].isin(seen_ids))].copy()
    pool2 = pool2.sort_values("popularity", ascending=False).head(GLOBAL_FALLBACK_CANDIDATES)
    pool = pd.concat([pool1, pool2], ignore_index=True).drop_duplicates("track_id")
    pool = score_pool(pool, profile).sort_values("score", ascending=False).head(limit)
    return pool["track_id"].tolist()

def recommend_autoplay_markov_first(current_track_id: str, trans, meta_all: pd.DataFrame, seen_ids: set, profile: dict, k: int = 10):
    markov_raw = []
    if current_track_id in trans and len(trans[current_track_id]) > 0:
        markov_raw = [tid for tid, _ in trans[current_track_id].most_common(k * 20)]
    markov_raw = dedup_preserve_order(markov_raw)

    markov_capped = apply_artist_cap(markov_raw, meta_all, MAX_PER_ARTIST_AUTOPLAY)
    markov_keep = markov_capped[:min(AUTOPLAY_MARKOV_MIN, k)]

    fallback = build_fallback_pool(meta_all, seen_ids, profile, limit=DISCOVERY_POOL_LIMIT)

    composed = markov_keep + [tid for tid in fallback if tid not in set(markov_keep)]
    composed = dedup_preserve_order(composed)
    composed = apply_artist_cap(composed, meta_all, MAX_PER_ARTIST_AUTOPLAY)
    return composed[:k]

def recommend_autoplay_fallback_first(current_track_id: str, trans, meta_all: pd.DataFrame, seen_ids: set, profile: dict, k: int = 10):
    """
    전이가 약한 곡에서는 Markov를 억지로 붙이지 않고 fallback을 먼저 쓰되,
    Markov 후보가 있으면 1개 정도만 맨 앞에 넣어주는 정도로 최소 반영.
    """
    fallback = build_fallback_pool(meta_all, seen_ids, profile, limit=DISCOVERY_POOL_LIMIT)

    markov_one = []
    if current_track_id in trans and len(trans[current_track_id]) > 0:
        markov_one = [tid for tid, _ in trans[current_track_id].most_common(5)]
    markov_one = apply_artist_cap(dedup_preserve_order(markov_one), meta_all, MAX_PER_ARTIST_AUTOPLAY)[:1]

    composed = markov_one + [tid for tid in fallback if tid not in set(markov_one)]
    composed = dedup_preserve_order(composed)
    composed = apply_artist_cap(composed, meta_all, MAX_PER_ARTIST_AUTOPLAY)
    return composed[:k]

def recommend_autoplay(current_track_id: str, trans, meta_all: pd.DataFrame, seen_ids: set, profile: dict, k: int = 10):
    out_sum = outgoing_sum(trans[current_track_id]) if (current_track_id in trans) else 0
    if out_sum >= MARKOV_GATE_THRESHOLD:
        mode = "MARKOV_FIRST"
        ids = recommend_autoplay_markov_first(current_track_id, trans, meta_all, seen_ids, profile, k=k)
    else:
        mode = "FALLBACK_FIRST"
        ids = recommend_autoplay_fallback_first(current_track_id, trans, meta_all, seen_ids, profile, k=k)
    return mode, out_sum, ids

# =========================
# MAIN
# =========================
def main():
    meta, seen, log = load_all()
    meta_all = pd.concat([meta, seen], ignore_index=True).drop_duplicates("track_id")
    profile = build_profile(seen)

    # HOME
    home18 = build_home18(meta, seen, profile)
    print("\n===== HOME 18 (FINAL v6) =====")
    for i, r in enumerate(home18, 1):
        print(f"{i:02d}. [{r['bucket']}] {r['artist']} - {r['title']} (pop={r['popularity']})")

    # MAP + MARKOV
    exact_map, by_artist = build_track_index(meta_all)
    log_mapped = map_log_to_track_id(log, exact_map, by_artist)

    sessions = sessions_from_log(log_mapped, SESSION_GAP_MINUTES)
    tr_sess, te_sess = time_split_sessions(sessions, TRAIN_RATIO)
    trans = train_markov(tr_sess)

    seen_ids = set(seen["track_id"].tolist())

    # CASE 1: last played
    last_tid = log_mapped.sort_values("endTime").iloc[-1]["track_id"]
    print("\n===== AUTOPLAY CASE 1 (Last-played seed) =====")
    cur_row = meta_all[meta_all["track_id"] == last_tid]
    if len(cur_row) > 0:
        cur_row = cur_row.iloc[0]
        print(f"Seed(Last): {cur_row['artist']} - {cur_row['title']} (track_id={last_tid})")
    else:
        print(f"Seed(Last) track_id={last_tid}")

    print_markov_debug(last_tid, trans, meta_all, topn=10)
    mode1, out1, ids1 = recommend_autoplay(last_tid, trans, meta_all, seen_ids, profile, k=10)
    print(f"\n[Autoplay Mode] {mode1}  (outgoing_sum={out1}, threshold={MARKOV_GATE_THRESHOLD})")
    show_reco_list("[Autoplay Top10 - Last seed]", ids1, meta_all)

    # CASE 2: markov strong seed
    strong_tid, strong_score = select_markov_strong_seed(trans, MARKOV_STRONG_MIN_OUTGOING)
    print("\n===== AUTOPLAY CASE 2 (Markov-strong seed) =====")
    if strong_tid is None:
        print(f"No Markov-strong seed found with outgoing >= {MARKOV_STRONG_MIN_OUTGOING}.")
        mode2, out2, ids2 = "NONE", 0, []
    else:
        srow = meta_all[meta_all["track_id"] == strong_tid]
        if len(srow) > 0:
            srow = srow.iloc[0]
            print(f"Seed(Strong): {srow['artist']} - {srow['title']} (track_id={strong_tid})  outgoing_sum={strong_score}")
        else:
            print(f"Seed(Strong) track_id={strong_tid} outgoing_sum={strong_score}")

        print_markov_debug(strong_tid, trans, meta_all, topn=10)
        mode2, out2, ids2 = recommend_autoplay(strong_tid, trans, meta_all, seen_ids, profile, k=10)
        print(f"\n[Autoplay Mode] {mode2}  (outgoing_sum={out2}, threshold={MARKOV_GATE_THRESHOLD})")
        show_reco_list("[Autoplay Top10 - Strong seed]", ids2, meta_all)

    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "paths": {"meta": PATH_META, "seen": PATH_SEEN, "log": PATH_LOG},
        "home18": home18,
        "autoplay": {
            "threshold": MARKOV_GATE_THRESHOLD,
            "last_seed_track_id": last_tid,
            "last_seed_outgoing_sum": out1,
            "last_seed_mode": mode1,
            "last_seed_top10": ids1,
            "markov_strong_seed_track_id": strong_tid,
            "markov_strong_seed_outgoing_sum": out2,
            "markov_strong_seed_mode": mode2,
            "markov_strong_top10": ids2,
        },
        "notes": [
            "Autoplay uses gating: if outgoing_sum >= threshold -> Markov-first, else Fallback-first.",
            "This avoids forcing Markov on sparse transitions and matches real product behavior."
        ]
    }
    pd.Series([out]).to_json("home_autoplay_final_v6_output.json", force_ascii=False, orient="records")
    print("\nSaved: home_autoplay_final_v6_output.json")


if __name__ == "__main__":
    main()
