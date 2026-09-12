"""
preprocess_style.py / train_style.py 단위 테스트용 합성(fake) 데이터 생성기.

2026-09-05에 data/winrate/matches.jsonl(승률 트랙이 이미 수집해둔 실제 API 응답 —
스타일 트랙과 같은 스키마를 공유한다)을 1건 읽어 확인한 matchDetail/pass/shoot 구조를
그대로 본떠 만든다.

서로 다른 "진짜 스타일"(짧은패스형/롱패스형/스루패스형) 3그룹의 유저를 만들어, 유저별로
같은 그룹 안에서는 비율이 비슷하지만 그룹 사이에는 뚜렷이 다르게 잔차(jitter)를 준다 —
K-means가 그룹을 구분해낼 수 있는지 스모크 테스트로 확인하기 위함이다. 몰수경기 1건과
표본이 모자란(min_matches 미만) 유저 1명도 섞어 필터링 로직을 검증한다.
"""

import random

STYLE_PROFILES = {
    "short_pass_heavy": {"short": 70, "long": 10, "through": 10, "dribble": 40, "in_pen": 60, "heading": 5},
    "long_pass_heavy": {"short": 20, "long": 55, "through": 10, "dribble": 30, "in_pen": 50, "heading": 15},
    "through_pass_heavy": {"short": 30, "long": 10, "through": 45, "dribble": 60, "in_pen": 70, "heading": 5},
}


def _make_match_info(ouid, nickname, profile, rng, match_end_type=0):
    """profile 비율(%) 근처에서 잔차를 줘서 pass/shoot try 카운트를 만든다."""
    pass_try = 100 + rng.randint(-10, 10)
    short_pass_try = int(pass_try * (profile["short"] + rng.uniform(-3, 3)) / 100)
    long_pass_try = int(pass_try * (profile["long"] + rng.uniform(-3, 3)) / 100)
    through_pass_try = int(pass_try * (profile["through"] + rng.uniform(-3, 3)) / 100)
    lobbed_through_pass_try = max(0, int(through_pass_try * 0.1))

    shoot_total = 5 + rng.randint(-2, 2)
    shoot_total = max(shoot_total, 1)
    shoot_in_penalty = min(shoot_total, max(0, int(shoot_total * (profile["in_pen"] + rng.uniform(-5, 5)) / 100)))
    shoot_out_penalty = shoot_total - shoot_in_penalty
    shoot_heading = min(shoot_total, max(0, int(shoot_total * (profile["heading"] + rng.uniform(-3, 3)) / 100)))

    dribble_yard = max(0, int(profile["dribble"] + rng.uniform(-5, 5)))

    return {
        "ouid": ouid,
        "nickname": nickname,
        "division": 2300,
        "matchDetail": {
            "seasonId": 202604,
            "matchResult": rng.choice(["승", "패"]),
            "matchEndType": match_end_type,
            "dribble": dribble_yard,
            "possession": 50,
        },
        "shoot": {
            "shootTotal": shoot_total,
            "shootHeading": shoot_heading,
            "shootInPenalty": shoot_in_penalty,
            "shootOutPenalty": shoot_out_penalty,
            "shootPenaltyKick": 0,
            "shootFreekick": 0,
            "goalTotal": 0,
        },
        "pass": {
            "passTry": pass_try,
            "shortPassTry": short_pass_try,
            "longPassTry": long_pass_try,
            "throughPassTry": through_pass_try,
            "lobbedThroughPassTry": lobbed_through_pass_try,
        },
        "player": [],
    }


def make_fake_style_matches(n_users_per_profile=4, n_matches_per_user=15, seed=13):
    """스타일 그룹별 n_users_per_profile명 x n_matches_per_user경기 + 몰수경기 1건 +
    표본 부족(min_matches 미만) 유저 1명(3경기만)을 만든다."""
    rng = random.Random(seed)
    matches = []
    match_counter = 0

    for profile_name, profile in STYLE_PROFILES.items():
        for u in range(n_users_per_profile):
            ouid = f"{profile_name}_{u}"
            for _ in range(n_matches_per_user):
                opp_ouid = f"opponent_{match_counter}"
                matches.append({
                    "matchId": f"match_{match_counter}",
                    "matchDate": "2026-09-05T12:00:00",
                    "matchType": 50,
                    "matchInfo": [
                        _make_match_info(ouid, f"user_{ouid}", profile, rng),
                        _make_match_info(opp_ouid, f"user_{opp_ouid}", profile, rng),
                    ],
                })
                match_counter += 1

    # 몰수경기 - extract_match_style_rows에서 제외되어야 한다.
    forfeit_profile = STYLE_PROFILES["short_pass_heavy"]
    matches.append({
        "matchId": "match_forfeit",
        "matchDate": "2026-09-05T12:00:00",
        "matchType": 50,
        "matchInfo": [
            _make_match_info("forfeit_a", "user_forfeit_a", forfeit_profile, rng, match_end_type=3),
            _make_match_info("forfeit_b", "user_forfeit_b", forfeit_profile, rng, match_end_type=3),
        ],
    })

    # 표본 부족 유저 - aggregate_user_style의 min_matches 필터에서 제외되어야 한다.
    sparse_profile = STYLE_PROFILES["long_pass_heavy"]
    for i in range(3):
        matches.append({
            "matchId": f"match_sparse_{i}",
            "matchDate": "2026-09-05T12:00:00",
            "matchType": 50,
            "matchInfo": [
                _make_match_info("sparse_user", "user_sparse", sparse_profile, rng),
                _make_match_info(f"sparse_opp_{i}", f"user_sparse_opp_{i}", sparse_profile, rng),
            ],
        })

    return matches
