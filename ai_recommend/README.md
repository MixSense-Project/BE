# AI_recommend — MixSense Recommendation Module

## Overview

This module implements the core recommendation system for MixSense:

* **Home Feed (18 tracks)**

  * Repeat / LikeSimilar / Discovery structure
  * Max 1 track per artist (global constraint)

* **Autoplay**

  * 1st-order Markov next-track model
  * Gating:

    * outgoing transitions ≥ 5 → Markov-first
    * otherwise → preference-based fallback
  * Max 1 track per artist in Top10

---

## Data Required (local only)

Place CSV files under:

```
data/raw/
  meta_data_ppc.csv
  test_ind.csv
  StreamingHistory_music_0.csv
```

(Data files are excluded from Git.)

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Run

```bash
python recommender_home_autoplay.py
```

---

## Design Notes

* Home feed prioritizes diversity with strict artist-level constraint.
* Autoplay adapts to transition strength via gating.
* Designed for product-level stability under sparse behavioral data.

---

# ✅ 한국어 짧은 버전 (GitHub에 같이 넣어도 괜찮음)

---

## 개요

MixSense의 AI 추천 모듈입니다.

* 홈 추천 18곡 (Repeat / LikeSimilar / Discovery)
* 아티스트 중복 1곡 제한
* 자동재생은 Markov 전이 기반 + 게이팅 구조 적용

  * 전이 ≥ 5 → Markov 우선
  * 전이 부족 → 취향 기반 fallback

---

## 실행 방법

```bash
python recommender_home_autoplay.py
```