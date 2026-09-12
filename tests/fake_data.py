"""
preprocess.py / train.py 단위 테스트용 합성(fake) 데이터 생성기.

2026-09-04에 실제 Nexon API(match-detail)를 1회 호출해 확인한 스키마를 그대로 본떠
matchInfo/matchDetail/player 구조를 만든다. 정상 케이스뿐 아니라 몰수경기, 무승부,
SUB(spPosition==28) 포함, ouid 중복 등장 케이스도 함께 만들어 필터링 로직을 검증한다.

매치마다 서로 다른 spId 세트를 새로 발급한다(같은 스쿼드를 여러 매치에서 재사용하지 않음).
재사용하면 attack/mid/defense/gk_avg_score 4개가 사실상 "quality 하나"의 재라벨링이 되어
경기 전체에서 딱 2개 지점(저품질/고품질)만 존재하게 되고, 이 경우 4개 컬럼이 서로 완전히
선형종속(perfectly collinear)이 되어 train_winrate.fit_statsmodels_report()의
statsmodels.Logit이 특이행렬(Singular matrix) 에러로 죽는다. 또한 quality가 승패를
100% 결정하게 만들면(업셋 없음) 완전분리(perfect separation)가 일어나 같은 이유로 죽을 수
있어, 일정 확률로 업셋(품질 낮은 쪽이 이김)을 섞어 완전분리도 피한다.
"""

import random

SUB_POSITION_CODE = 28

UPSET_PROBABILITY = 0.15  # 품질 낮은 쪽이 이변으로 이기는 비율 (완전분리 방지용)


def _make_player(sp_id, sp_position, sp_grade=9):
    return {
        "spId": sp_id,
        "spPosition": sp_position,
        "spGrade": sp_grade,
        "status": {"shoot": 0, "goal": 0, "spRating": 0.0},
    }


def _make_squad(base_sp_id, starting_positions):
    """starting_positions에 있는 포지션 코드로 11명 + SUB 몇 명을 만든다."""
    players = [
        _make_player(base_sp_id + i, pos) for i, pos in enumerate(starting_positions)
    ]
    # 교체선수(SUB) 몇 명 추가 - avg_score 계산에서 제외돼야 한다.
    for i in range(3):
        players.append(_make_player(base_sp_id + 100 + i, SUB_POSITION_CODE))
    return players

# 11자리 포지션 코드(대충 4-4-2 형태, GK 포함) - 실제 포메이션 조합표는 아직 없어 테스트용 값.
# 0=GK, {3,4,6,7}=수비(RB/RCB/LCB/LB), {9,12,14,16}=미드필더(RDM/RM/CM/LM), {25,26}=공격(ST/LS)
# 4그룹(attack/midfield/defense/gk)이 전부 최소 1명씩 채워지도록 구성했다.
STARTING_POSITIONS = [0, 3, 4, 6, 7, 9, 12, 14, 16, 25, 26]

# 한 스쿼드가 쓰는 spId 오프셋 범위(11명 선발 + SUB 3명 = offset 0~10, 100~102)보다
# 넉넉하게 잡아, 매치 인덱스별 base_sp_id끼리 절대 겹치지 않게 한다.
_SQUAD_ID_STEP = 1000
_SIDE_B_OFFSET = 500


def _match_squad_base(i, side):
    """i번째 매치의 A/B측 스쿼드가 쓸 spId 시작값. 매치마다, 진영마다 겹치지 않는다."""
    base = 1_000_000 + i * _SQUAD_ID_STEP
    return base + (_SIDE_B_OFFSET if side == "b" else 0)


def _match_quality(i, side):
    """i번째 매치에서 side(a/b)가 '우세'(고품질 스쿼드)인지를 결정한다.

    실제 승패(a_wins)와는 별개다 - make_fake_matches가 업셋을 섞어 실제 승패를 정하더라도,
    카드 스탯 자체(quality)는 이 함수 기준으로 고정해야 매치 생성 쪽과 카드 생성 쪽이
    같은 스쿼드에 같은 품질을 매긴다.
    """
    a_favored = i % 2 == 0
    if side == "a":
        return "high" if a_favored else "low"
    return "low" if a_favored else "high"


def _make_match_info(ouid, nickname, division, result, base_sp_id, match_end_type=0):
    return {
        "ouid": ouid,
        "nickname": nickname,
        "division": division,
        "matchDetail": {
            "seasonId": 202604,
            "matchResult": result,
            "matchEndType": match_end_type,
            "systemPause": 0,
            "foul": 0,
            "injury": 0,
            "redCards": 0,
            "yellowCards": 0,
            "dribble": 50,
            "cornerKick": 1,
            "possession": 50,
            "offsideCount": 0,
            "averageRating": 6.5,
            "controller": "keyboard",
        },
        "shoot": {"shootTotal": 5, "goalTotal": 1},
        "shootDetail": [],
        "pass": {"passTry": 100, "passSuccess": 80},
        "defence": {"tackleTry": 10, "tackleSuccess": 5},
        "player": _make_squad(base_sp_id, STARTING_POSITIONS),
    }


def make_normal_match(match_id, ouid_a, ouid_b, division_a=2300, division_b=2400,
                       base_sp_id_a=100000, base_sp_id_b=200000):
    """정상 종료, 승/패가 명확히 갈리는 매치 1건 (matchInfo 2건)."""
    return {
        "matchId": match_id,
        "matchDate": "2026-09-03T12:00:00",
        "matchType": 50,
        "matchInfo": [
            _make_match_info(ouid_a, f"user_{ouid_a}", division_a, "승", base_sp_id_a),
            _make_match_info(ouid_b, f"user_{ouid_b}", division_b, "패", base_sp_id_b),
        ],
    }


def make_draw_match(match_id, ouid_a, ouid_b):
    """무승부 매치 - extract_rows에서 제외되어야 한다."""
    return {
        "matchId": match_id,
        "matchDate": "2026-09-03T12:00:00",
        "matchType": 50,
        "matchInfo": [
            _make_match_info(ouid_a, f"user_{ouid_a}", 2300, "무", 300000),
            _make_match_info(ouid_b, f"user_{ouid_b}", 2300, "무", 400000),
        ],
    }


def make_forfeit_match(match_id, ouid_a, ouid_b):
    """몰수경기(matchEndType != 0) - extract_rows에서 제외되어야 한다."""
    return {
        "matchId": match_id,
        "matchDate": "2026-09-03T12:00:00",
        "matchType": 50,
        "matchInfo": [
            _make_match_info(ouid_a, f"user_{ouid_a}", 2300, "승", 500000, match_end_type=3),
            _make_match_info(ouid_b, f"user_{ouid_b}", 2300, "패", 600000, match_end_type=3),
        ],
    }


def make_fake_matches(n_normal=40, seed=7):
    """정상 매치 n_normal건 + 무승부 1건 + 몰수경기 1건 + ouid 중복 매치 1건을 만든다.

    승/패는 스쿼드 품질(quality)과 상관관계를 갖도록 설계하되, UPSET_PROBABILITY 확률로
    품질 낮은 쪽이 이기는 경우를 섞어 완전분리(perfect separation)를 피한다.
    """
    rng = random.Random(seed)
    matches = []

    for i in range(n_normal):
        ouid_a = f"ouid_a_{i}"
        ouid_b = f"ouid_b_{i}"
        a_favored = i % 2 == 0
        upset = rng.random() < UPSET_PROBABILITY
        a_wins = (not a_favored) if upset else a_favored

        base_a = _match_squad_base(i, "a")
        base_b = _match_squad_base(i, "b")
        result_a = "승" if a_wins else "패"
        result_b = "패" if a_wins else "승"
        matches.append({
            "matchId": f"match_{i}",
            "matchDate": "2026-09-03T12:00:00",
            "matchType": 50,
            "matchInfo": [
                _make_match_info(ouid_a, f"user_{ouid_a}", rng.choice([2200, 2300, 2400]),
                                  result_a, base_a),
                _make_match_info(ouid_b, f"user_{ouid_b}", rng.choice([2200, 2300, 2400]),
                                  result_b, base_b),
            ],
        })

    matches.append(make_draw_match("match_draw", "ouid_draw_a", "ouid_draw_b"))
    matches.append(make_forfeit_match("match_forfeit", "ouid_forfeit_a", "ouid_forfeit_b"))

    # 같은 ouid(ouid_a_0)가 다른 match_id에도 등장하는 케이스 - 중복 제거 로직 검증용.
    # ouid_a_0 쪽 스쿼드는 extract_rows에서 어차피 dedup으로 버려지므로 카드 매칭 대상이
    # 아니고, 새 상대측 ouid_c_dup(base 200000)만 실제로 카드가 필요하다.
    dup_match = make_normal_match("match_dup", "ouid_a_0", "ouid_c_dup")
    matches.append(dup_match)

    return matches


def make_fake_player_card_rows(n_normal=40, seed=11):
    """make_fake_matches(n_normal=n_normal)가 만든 모든 스쿼드의 spId를 커버하는
    player 카드 데이터를 만든다.

    quality(low/high) 평균 근처에서 선수별/스탯별로 잔차(jitter)를 줘서, 매치마다 스쿼드가
    전부 다르므로 attack/mid/defense/gk_avg_score 4개가 서로 완전히 비례하지 않게 한다
    (평균만 놓고 보면 quality를 따라가되, 실제 값은 매치마다 조금씩 다름). quality 간
    평균 차이(30점)는 잔차 폭(±3점)보다 훨씬 커서 여전히 승패와 강하게 상관된다.
    """
    field_player_stats = [
        "속력", "가속력", "골 결정력", "슛 파워", "중거리 슛", "위치 선정", "발리슛", "페널티 킥",
        "짧은 패스", "시야", "크로스", "긴 패스", "프리킥", "커브", "드리블", "볼 컨트롤",
        "민첩성", "밸런스", "반응 속도", "대인 수비", "태클", "가로채기", "헤더", "슬라이딩 태클",
        "몸싸움", "스태미너", "적극성", "점프", "침착성",
    ]
    gk_only_stats = ["GK 다이빙", "GK 핸들링", "GK 킥", "GK 반응속도", "GK 위치 선정"]

    rng = random.Random(seed)
    rows = []

    def add_squad_rows(base, quality):
        mean_stat = 60 if quality == "low" else 90
        mean_gk = 40 if quality == "low" else 80
        for offset in list(range(11)) + [100, 101, 102]:
            sp_id = base + offset
            row = {"spid": sp_id}
            row.update({name: mean_stat + rng.uniform(-3, 3) for name in field_player_stats})
            row.update({name: mean_gk + rng.uniform(-3, 3) for name in gk_only_stats})
            rows.append(row)

    for i in range(n_normal):
        add_squad_rows(_match_squad_base(i, "a"), _match_quality(i, "a"))
        add_squad_rows(_match_squad_base(i, "b"), _match_quality(i, "b"))

    # match_dup의 새 상대측(ouid_c_dup, base 200000)용 카드.
    add_squad_rows(200000, "low")

    return rows
