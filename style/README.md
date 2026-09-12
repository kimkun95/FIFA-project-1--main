# style

플레이스타일 진단(PCA 시각화 + K-means 군집화, 비지도학습) 파이프라인.

## 구성
- `collect_style_matches.py` — 유저당 최근 최대 100경기를 깊게 수집 (스노볼 방식으로
  유저 풀도 함께 넓힘). 저장 스키마는 승률 트랙과 동일, `data/style/matches_style.jsonl`에
  통짜 저장.
- `preprocess_style.py` — matches_style.jsonl -> (match, ouid) 원시 카운트 -> 유저 단위
  집계 -> `USER_STYLE_PROFILE`의 6개 비율 feature(short_pass_ratio, long_pass_ratio,
  through_pass_ratio, dribble_intensity, in_penalty_shoot_ratio, heading_shoot_ratio).
  선수 카드 스탯(player_stats_final.csv)은 쓰지 않는다 — 팀 단위 pass/shoot/dribble
  기록만으로 "유저가 어떻게 플레이했는가"를 본다.
- `train_style.py` — StandardScaler로 표준화 후 k=2~6 KMeans를 전부 시도해 실루엣
  점수가 가장 높은 k를 사후에 선택(CLAUDE.md 원칙 #4: 유형 개수를 미리 정하지 않음).
  PCA(2차원)는 군집화가 아니라 시각화 좌표 추출에만 별도로 사용.
- `utils.py` — 저장 공통 함수 (winrate/utils.py와 동일 내용, 트랙 독립성을 위해 복제).

## 확정된 설계 결정

### Feature 계산: 티어(division)별 정규화 없음
- `short_pass_ratio`, `long_pass_ratio`, `through_pass_ratio`, `dribble_intensity` 등은
  **유저 본인의 경기 데이터로만 계산한 원본 비율 그대로** 사용한다. 티어 평균을 빼거나 나누는
  정규화/보정을 절대 하지 않는다.
- **왜**: 진단 화면에서 "당신은 이런 스타일입니다"를 그 유저 **실제 플레이 데이터에 기반한
  관찰적 진단(observational diagnosis)**으로 제시하기 위함. 다른 유저 데이터가 섞인 상대값
  (상대주의적 평가)을 쓰지 않는다.
- **알려진 한계 (의도적 수용)**: `long_pass_ratio`(+0.25), `through_pass_ratio`(+0.33)가
  `match_team_data.csv` 기준 티어와 상관관계가 있다는 것을 팀이 이미 확인했다. 이는 K-means
  군집이 스타일 차이뿐 아니라 티어 분포와도 어느 정도 겹칠 수 있음을 의미한다. **이건 "고쳐야
  할 버그"가 아니라 관찰적 진단이라는 설계 선택 때문에 발생하는 트레이드오프이며, 의도적으로
  받아들인 한계다.**
- **사후 검증 시 제약**: 군집화 결과를 분석할 때 군집 × division crosstab을 **참고용으로만**
  찍어보는 것은 괜찮다. 하지만 그 결과(예: "군집과 티어의 상관계수가 높다")를 근거로 정규화
  로직을 다시 넣거나 feature를 수정하면 안 된다. 이는 이미 끝난 논의다.

## 현재 상태 (뼈대 단계)
아직 `data/style/matches_style.jsonl` 실 데이터가 없다 (`collect_style_matches.py`
미실행). 파이프라인 로직은 `tests/fake_data_style.py` + `tests/test_style_pipeline_smoke.py`
합성 데이터로만 검증됨:
```
./venv/Scripts/python.exe tests/test_style_pipeline_smoke.py
```

## 미해결 (실 데이터 수집 전 확정 필요)
- `MIN_MATCHES_PER_USER`(현재 임시 10) — 스타일 진단용 유저의 최소 표본 경기 수 하한선.
- `dribble_intensity` 정규화 방식(현재 "경기당 평균 야드") — 실제 데이터 분포를 본 뒤
  재검토.
- 몰수경기(matchEndType != 0) 필터링 — 승률 트랙 결정을 그대로 따라뒀지만 팀 논의로
  바뀔 수 있음.
- 군집이 나온 뒤 각 군집에 붙일 스타일 이름(예: "스루패스 위주형") — `style_type`은
  아직 숫자 군집 id일 뿐, `train_style.py`가 저장하는 `style_diagnosis_baseline_summary.txt`의
  군집별 feature 평균을 보고 팀이 사후에 정한다.
