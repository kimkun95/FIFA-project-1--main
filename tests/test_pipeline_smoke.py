"""
preprocess_winrate.py / train_winrate.py / utils.py 스모크 테스트 (pytest 불필요, plain assert).

실제 matches.jsonl / player_1000_final.csv가 아직 없어서, 확인된 실제 API 스키마를 본뜬
합성 데이터(tests/fake_data.py)로 파이프라인 각 단계(파싱 -> feature 조립 -> leakage 체크 ->
3분할 -> sklearn Pipeline 학습 -> statsmodels 리포트 -> 저장)가 에러 없이 동작하는지만
검증한다. 실제 데이터로의 최종 검증은 별도.

실행: ./venv/Scripts/python.exe tests/test_pipeline_smoke.py
"""

import json
import os
import sys
import tempfile

import joblib
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "winrate"))

import preprocess_winrate as preprocess
import train_winrate as train
import utils
from fake_data import make_fake_matches, make_fake_player_card_rows


def test_extract_rows_filters_correctly():
    matches = make_fake_matches(n_normal=10)
    rows_df = preprocess.extract_rows(matches)

    # 무승부/몰수경기 matchId는 결과에 전혀 없어야 한다.
    assert "match_draw" not in set(rows_df["match_id"])
    assert "match_forfeit" not in set(rows_df["match_id"])

    # ouid_a_0은 match_0과 match_dup 양쪽에 등장하지만 1행만 남아야 한다.
    assert (rows_df["ouid"] == "ouid_a_0").sum() == 1

    # 정상 매치 10건 x 2 + dup 매치 2 - dup으로 제거된 1행 = 21행
    # (무승부/몰수경기 매치도 각각 matchInfo 2건씩 만들지만 전부 필터링됨)
    assert len(rows_df) == 21, f"expected 21 rows, got {len(rows_df)}"

    # sp_players에 SUB(spPosition==28) 선수가 섞여있으면 안 된다 (base_sp_id+100~102 제외 확인)
    for sp_players in rows_df["sp_players"]:
        assert len(sp_players) == 11, f"SUB 제외 후 11명이어야 하는데 {len(sp_players)}명"
        assert all(p["sp_position"] != 28 for p in sp_players)

    print("OK: test_extract_rows_filters_correctly")


def test_assert_no_leakage():
    preprocess.assert_no_leakage(preprocess.GROUP_SCORE_COLUMNS + ["tier"])
    try:
        preprocess.assert_no_leakage(["avg_stat_score", "goal_total"])
    except AssertionError:
        pass
    else:
        raise AssertionError("leakage 컬럼을 걸러내지 못함")
    print("OK: test_assert_no_leakage")


def test_full_pipeline_with_fake_data():
    matches = make_fake_matches(n_normal=40)
    card_rows = make_fake_player_card_rows()

    with tempfile.TemporaryDirectory() as tmp_dir:
        matches_path = os.path.join(tmp_dir, "matches.jsonl")
        with open(matches_path, "w", encoding="utf-8") as f:
            for m in matches:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

        csv_path = os.path.join(tmp_dir, "player_1000_final.csv")
        pd.DataFrame(card_rows).to_csv(csv_path, index=False)

        output_dir = os.path.join(tmp_dir, "output")

        preprocess.assert_no_leakage(train.FEATURE_COLUMNS)

        loaded_matches = preprocess.load_matches(matches_path)
        rows_df = preprocess.extract_rows(loaded_matches)

        player_card_df = preprocess.load_player_cards(csv_path)
        feature_df, match_rate = preprocess.assemble_feature_table(rows_df, player_card_df)

        assert match_rate == 100.0, f"fake 데이터는 전원 매칭되어야 하는데 {match_rate}%"
        for col in preprocess.GROUP_SCORE_COLUMNS:
            assert feature_df[col].isna().sum() == 0
        assert set(feature_df.columns) == (
            {"match_id", "ouid", "tier", "result"} | set(preprocess.GROUP_SCORE_COLUMNS)
        )

        train_df, val_df, test_df = train.split_dataset(feature_df, output_dir)
        assert len(train_df) + len(val_df) + len(test_df) == len(feature_df)

        split_path = os.path.join(output_dir, "split_assignment.json")
        assert os.path.exists(split_path)
        with open(split_path, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        assert len(split_data["rows"]) == len(feature_df)
        assert {r["split"] for r in split_data["rows"]} == {"train", "val", "test"}

        pipeline = train.build_pipeline()
        pipeline.fit(
            train_df[train.FEATURE_COLUMNS].astype(float), train_df["result"].astype(int)
        )
        val_metrics = train.evaluate_pipeline(pipeline, val_df, train.FEATURE_COLUMNS)
        test_metrics = train.evaluate_pipeline(pipeline, test_df, train.FEATURE_COLUMNS)

        for accuracy, p_value, _, _ in (val_metrics, test_metrics):
            assert 0.0 <= accuracy <= 1.0
            assert 0.0 <= p_value <= 1.0
            # avg_score 4개(attack/mid/defense/gk)가 승패와 강하게 상관되도록(업셋 섞어서
            # 완전분리는 피하되) fake 데이터를 만들었으니 baseline보다는 확실히 잘 맞아야 한다.
            assert accuracy > 0.5, f"fake 데이터에서 accuracy가 너무 낮음: {accuracy}"

        statsmodels_result = train.fit_statsmodels_report(train_df, train.FEATURE_COLUMNS)

        summary_text = train.build_summary_text(
            pipeline, statsmodels_result, train.FEATURE_COLUMNS, len(train_df), match_rate,
            val_metrics, test_metrics,
        )
        model_path = os.path.join(output_dir, "win_prediction_baseline_pipeline.joblib")
        summary_path = os.path.join(output_dir, "win_prediction_baseline_summary.txt")
        utils.save_model(pipeline, model_path)
        utils.save_text(summary_text, summary_path)

        assert os.path.exists(model_path)
        assert os.path.exists(summary_path)

        # 저장된 pipeline을 다시 불러와도 정상 동작하는지 확인
        reloaded = joblib.load(model_path)
        reloaded_metrics = train.evaluate_pipeline(reloaded, test_df, train.FEATURE_COLUMNS)
        assert reloaded_metrics[0] == test_metrics[0]

    print(
        f"OK: test_full_pipeline_with_fake_data "
        f"(val_accuracy={val_metrics[0]:.3f}, test_accuracy={test_metrics[0]:.3f})"
    )


def test_missing_csv_column_raises_clear_error():
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "bad_player_cards.csv")
        pd.DataFrame([{"spid": 1, "속력": 50}]).to_csv(csv_path, index=False)
        try:
            preprocess.load_player_cards(csv_path)
        except ValueError as e:
            assert "가속력" in str(e)
        else:
            raise AssertionError("컬럼 누락을 걸러내지 못함")
    print("OK: test_missing_csv_column_raises_clear_error")


if __name__ == "__main__":
    test_extract_rows_filters_correctly()
    test_assert_no_leakage()
    test_full_pipeline_with_fake_data()
    test_missing_csv_column_raises_clear_error()
    print("\n모든 스모크 테스트 통과")
