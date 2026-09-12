"""
matches_style.jsonl 파싱 및 USER_STYLE_PROFILE feature(6개 비율) 조립.

승률 트랙(winrate/preprocess_winrate.py)과 저장 스키마는 동일하지만(같은 API,
matches.jsonl 통짜 저장 원칙), 스타일 진단은 "선수 카드 스탯"이 아니라 "유저가
실제로 어떻게 플레이했는가"(패스/드리블/슈팅 선택)를 보므로 player_stats_final.csv를
전혀 쓰지 않는다. 대신 팀 단위 matchDetail/pass/shoot 필드를 쓴다.

필드 매핑은 2026-09-05에 data/winrate/matches.jsonl(승률 트랙이 이미 수집해둔 실제
API 응답, 같은 스키마를 공유하므로 그대로 참고 가능)을 1건 읽어 확인했다:
    matchInfo[i]["matchDetail"]["dribble"]          -> 팀 드리블 야드 (정수)
    matchInfo[i]["pass"]["passTry"]                 -> 전체 패스 시도
    matchInfo[i]["pass"]["shortPassTry"]             -> 짧은 패스 시도
    matchInfo[i]["pass"]["longPassTry"]              -> 긴 패스 시도
    matchInfo[i]["pass"]["throughPassTry"]           -> 스루 패스 시도
    matchInfo[i]["pass"]["lobbedThroughPassTry"]     -> 로빙 스루 패스 시도
        (CLAUDE.md 방침대로 스루+로빙스루를 합산해 through_pass_try로 쓴다)
    matchInfo[i]["shoot"]["shootTotal"]              -> 전체 슛 시도
    matchInfo[i]["shoot"]["shootHeading"]            -> 헤딩 슛 시도
    matchInfo[i]["shoot"]["shootInPenalty"]          -> 박스 안 슛 시도
    matchInfo[i]["shoot"]["shootOutPenalty"]         -> 박스 밖 슛 시도
⚠️ shoot.shootPenaltyKick/shootFreekick(페널티킥/프리킥 세트피스)는 상대 반칙에 좌우되는
   값이라 CLAUDE.md의 "유저의 선택이 아니라 상황이 만드는 값" 원칙에 따라 절대 쓰지 않는다.
   shootInPenalty/shootOutPenalty는 세트피스가 아니라 "슛 위치"(박스 안/밖)라 이 규칙과
   무관하고, data-schema.md의 MATCH 엔티티에 그대로 포함된 필드다.
⚠️ goalTotal/goalHeading 등 "골 여부"가 들어간 필드는 승률 트랙과 마찬가지로 쓰지 않는다
   (여기서는 leakage 문제가 아니라 애초에 스타일 진단에 승패 정보 자체가 불필요하기 때문).

TODO(미해결, CLAUDE.md 참고): 스타일 진단용 유저의 최소 표본 경기 수 하한선이 팀 논의로
확정되지 않았다. 아래 MIN_MATCHES_PER_USER=10은 "표본이 너무 적으면 비율이 노이즈로
튄다"는 상식적 기준으로 임시로 잡아둔 값이며, 팀 논의 후 바뀔 수 있다.
"""

import json
import os

import numpy as np
import pandas as pd

# ============ CONFIG ============
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCHES_FILE = os.path.join(REPO_ROOT, "data", "style", "matches_style.jsonl")

NORMAL_MATCH_END_TYPE = 0

# TODO(미해결): 팀 논의로 확정 필요. 우선 임시값.
MIN_MATCHES_PER_USER = 10

# --- 필드 매핑 (위 docstring 참고, 2026-09-05 확인) ---
PASS_TRY_FIELD = "passTry"
SHORT_PASS_TRY_FIELD = "shortPassTry"
LONG_PASS_TRY_FIELD = "longPassTry"
THROUGH_PASS_TRY_FIELDS = ["throughPassTry", "lobbedThroughPassTry"]
DRIBBLE_YARD_FIELD = "dribble"  # matchDetail 소속 (pass/shoot와 달리 최상위가 아님)
SHOOT_TOTAL_FIELD = "shootTotal"
SHOOT_HEADING_FIELD = "shootHeading"
SHOOT_IN_PENALTY_FIELD = "shootInPenalty"
SHOOT_OUT_PENALTY_FIELD = "shootOutPenalty"

# data-schema.md USER_STYLE_PROFILE에 정의된 6개 feature (순서 고정).
FEATURE_COLUMNS = [
    "short_pass_ratio",
    "long_pass_ratio",
    "through_pass_ratio",
    "dribble_intensity",
    "in_penalty_shoot_ratio",
    "heading_shoot_ratio",
]
# ====================================================


def load_matches(path):
    """matches_style.jsonl을 한 줄씩 읽어 dict 리스트로 반환한다."""
    matches = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                matches.append(json.loads(line))
    return matches


def extract_match_style_rows(matches):
    """matches_style.jsonl의 raw match-detail 리스트에서 (match_id, ouid) 단위
    원시 카운트 행을 뽑는다.

    승률 트랙과 달리 승/무/패는 스타일과 무관하므로 결과로는 필터링하지 않는다.
    몰수경기(matchEndType != 0)만 데이터 품질 문제로 제외한다(승률 트랙의 결정을
    그대로 따름 — CLAUDE.md 미해결 항목 참고, 팀 논의로 바뀔 수 있음).
    같은 유저가 여러 매치에 나오는 것은 여기서는 정상이다(오히려 그게 목적 —
    한 유저의 최근 최대 100경기를 모아 집계해야 하므로 dedup하지 않는다).
    """
    rows = []
    excluded_forfeit = 0

    for match in matches:
        match_id = match.get("matchId")
        for info in match.get("matchInfo", []):
            match_detail = info.get("matchDetail", {})
            if match_detail.get("matchEndType") != NORMAL_MATCH_END_TYPE:
                excluded_forfeit += 1
                continue

            pass_stats = info.get("pass", {})
            shoot_stats = info.get("shoot", {})

            rows.append({
                "match_id": match_id,
                "ouid": info.get("ouid"),
                "pass_try": pass_stats.get(PASS_TRY_FIELD, 0),
                "short_pass_try": pass_stats.get(SHORT_PASS_TRY_FIELD, 0),
                "long_pass_try": pass_stats.get(LONG_PASS_TRY_FIELD, 0),
                "through_pass_try": sum(
                    pass_stats.get(f, 0) for f in THROUGH_PASS_TRY_FIELDS
                ),
                "dribble_yard": match_detail.get(DRIBBLE_YARD_FIELD, 0),
                "shoot_total": shoot_stats.get(SHOOT_TOTAL_FIELD, 0),
                "shoot_heading": shoot_stats.get(SHOOT_HEADING_FIELD, 0),
                "shoot_in_penalty": shoot_stats.get(SHOOT_IN_PENALTY_FIELD, 0),
                "shoot_out_penalty": shoot_stats.get(SHOOT_OUT_PENALTY_FIELD, 0),
            })

    print(f"  [필터링] 몰수경기 제외: {excluded_forfeit}건")
    print(f"  [추출] 최종 (match, ouid) 행 수: {len(rows)}")
    return pd.DataFrame(rows)


def _safe_ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else np.nan


def aggregate_user_style(rows_df, min_matches=MIN_MATCHES_PER_USER):
    """(match, ouid) 원시 카운트를 유저 단위로 합산해 6개 비율 feature를 계산한다.

    min_matches 미만인 유저는 비율이 노이즈에 크게 좌우되므로 제외한다.
    분모(pass_try/shoot_total)가 0인 유저도 비율 계산이 불가능해 제외한다.

    ⚠️ 티어(division) 정규화는 의도적으로 하지 않는다. 진단 화면에서 "관찰적 진단"을
    제시하기 위해, 유저 본인의 원본 비율 그대로 쓴다 (style/README.md 참고).
    """
    if rows_df.empty:
        return pd.DataFrame(columns=["ouid", "n_matches"] + FEATURE_COLUMNS)

    grouped = rows_df.groupby("ouid").agg(
        n_matches=("match_id", "count"),
        pass_try=("pass_try", "sum"),
        short_pass_try=("short_pass_try", "sum"),
        long_pass_try=("long_pass_try", "sum"),
        through_pass_try=("through_pass_try", "sum"),
        dribble_yard=("dribble_yard", "sum"),
        shoot_total=("shoot_total", "sum"),
        shoot_heading=("shoot_heading", "sum"),
        shoot_in_penalty=("shoot_in_penalty", "sum"),
    ).reset_index()

    before = len(grouped)
    grouped = grouped[grouped["n_matches"] >= min_matches].copy()
    print(
        f"  [표본 필터] 최근 경기 {min_matches}건 미만 유저 제외: "
        f"{before - len(grouped)}명 (남은 유저: {len(grouped)}명)"
    )

    grouped["short_pass_ratio"] = grouped.apply(
        lambda r: _safe_ratio(r["short_pass_try"], r["pass_try"]), axis=1
    )
    grouped["long_pass_ratio"] = grouped.apply(
        lambda r: _safe_ratio(r["long_pass_try"], r["pass_try"]), axis=1
    )
    grouped["through_pass_ratio"] = grouped.apply(
        lambda r: _safe_ratio(r["through_pass_try"], r["pass_try"]), axis=1
    )
    # TODO(미확정): dribble_intensity를 "경기당 평균 드리블 야드"로 정의했다.
    # 패스 시도 대비 비율 등 다른 정규화 방식이 더 나을 수도 있어, 실제 데이터로
    # 분포를 본 뒤 재검토 필요.
    grouped["dribble_intensity"] = grouped["dribble_yard"] / grouped["n_matches"]
    grouped["in_penalty_shoot_ratio"] = grouped.apply(
        lambda r: _safe_ratio(r["shoot_in_penalty"], r["shoot_total"]), axis=1
    )
    grouped["heading_shoot_ratio"] = grouped.apply(
        lambda r: _safe_ratio(r["shoot_heading"], r["shoot_total"]), axis=1
    )

    before = len(grouped)
    style_df = grouped.dropna(subset=FEATURE_COLUMNS).copy()
    dropped = before - len(style_df)
    print(f"  [비율 계산] 분모가 0이라 비율을 못 낸 유저 제외: {dropped}명")

    return style_df[["ouid", "n_matches"] + FEATURE_COLUMNS]
