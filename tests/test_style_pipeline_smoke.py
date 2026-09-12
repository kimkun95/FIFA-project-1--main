"""
preprocess_style.py / train_style.py / utils.py 스모크 테스트 (pytest 불필요, plain assert).

실제 matches_style.jsonl이 아직 없어서(수집 스크립트 미실행), 확인된 실제 API 스키마를
본뜬 합성 데이터(tests/fake_data_style.py)로 파이프라인 각 단계(파싱 -> 유저 집계 ->
표준화 -> k 탐색 -> KMeans -> 저장)가 에러 없이 동작하는지, 그리고 뚜렷이 다른 스타일
그룹을 실제로 구분해내는지만 검증한다. 실제 데이터로의 최종 검증은 별도.

실행: ./venv/Scripts/python.exe tests/test_style_pipeline_smoke.py
"""

import json
import os
import sys
import tempfile

import joblib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "style"))

import preprocess_style as preprocess
import train_style as train
import utils
from fake_data_style import make_fake_style_matches


def test_extract_match_style_rows_filters_forfeit():
    matches = make_fake_style_matches(n_users_per_profile=2, n_matches_per_user=5)
    rows_df = preprocess.extract_match_style_rows(matches)

    assert "match_forfeit" not in set(rows_df["match_id"])
    for col in ["pass_try", "short_pass_try", "long_pass_try", "through_pass_try",
                "dribble_yard", "shoot_total", "shoot_heading", "shoot_in_penalty",
                "shoot_out_penalty"]:
        assert col in rows_df.columns
    print("OK: test_extract_match_style_rows_filters_forfeit")


def test_aggregate_user_style_filters_sparse_users():
    matches = make_fake_style_matches(n_users_per_profile=2, n_matches_per_user=15)
    rows_df = preprocess.extract_match_style_rows(matches)
    style_df = preprocess.aggregate_user_style(rows_df, min_matches=10)

    # 표본 부족 유저(3경기)는 min_matches=10 미만이라 제외되어야 한다.
    assert "sparse_user" not in set(style_df["ouid"])

    for col in preprocess.FEATURE_COLUMNS:
        assert col in style_df.columns
        assert style_df[col].notna().all()
        assert (style_df[col] >= 0).all()
    print("OK: test_aggregate_user_style_filters_sparse_users")


def test_full_pipeline_with_fake_data():
    matches = make_fake_style_matches(n_users_per_profile=5, n_matches_per_user=15)

    with tempfile.TemporaryDirectory() as tmp_dir:
        matches_path = os.path.join(tmp_dir, "matches_style.jsonl")
        with open(matches_path, "w", encoding="utf-8") as f:
            for m in matches:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

        output_dir = os.path.join(tmp_dir, "output")
        model_dir = os.path.join(tmp_dir, "models")

        loaded_matches = preprocess.load_matches(matches_path)
        rows_df = preprocess.extract_match_style_rows(loaded_matches)
        style_df = preprocess.aggregate_user_style(rows_df)

        # 스타일 그룹 3개 x 유저 5명 = 15명 (표본 부족/몰수경기 유저는 별도라 안 섞임)
        assert len(style_df) == 15, f"expected 15 users, got {len(style_df)}"

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(style_df[preprocess.FEATURE_COLUMNS].astype(float))

        best_k, k_search_results = train.select_best_k(x_scaled)
        labels, best_score, best_kmeans = k_search_results[best_k]
        assert 2 <= best_k <= 6
        assert -1.0 <= best_score <= 1.0

        style_df = style_df.copy()
        style_df["style_type"] = labels

        # 뚜렷이 다른 3그룹으로 데이터를 만들었으니, 최적 k로 나눈 군집이 실제 프로필
        # 그룹(ouid 접두사)과 강하게 일치해야 한다 (완벽히 3이 아니어도 되지만 최소한
        # 같은 그룹 유저끼리는 대부분 같은 군집으로 묶여야 한다).
        style_df["true_group"] = style_df["ouid"].str.rsplit("_", n=1).str[0]
        # 그룹별로 지배적인 군집 하나를 뽑아, 그 군집에 안 속한 유저 비율을 계산.
        mismatch = 0
        for true_group, sub in style_df.groupby("true_group"):
            dominant_cluster = sub["style_type"].mode()[0]
            mismatch += (sub["style_type"] != dominant_cluster).sum()
        mismatch_rate = mismatch / len(style_df)
        assert mismatch_rate <= 0.2, (
            f"뚜렷이 다른 스타일 그룹인데 군집화가 잘 못 갈랐음 (mismatch_rate={mismatch_rate:.2f})"
        )

        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=train.RANDOM_SEED)
        coords = pca.fit_transform(x_scaled)
        style_df["pca_x"] = coords[:, 0]
        style_df["pca_y"] = coords[:, 1]

        cluster_summary = train.build_cluster_summary(style_df, preprocess.FEATURE_COLUMNS)
        summary_text = train.build_summary_text(
            best_k, k_search_results, cluster_summary, len(style_df),
            preprocess.MIN_MATCHES_PER_USER,
        )

        model_path = os.path.join(model_dir, "style_diagnosis_baseline.joblib")
        csv_path = os.path.join(output_dir, "user_style_profile.csv")
        summary_path = os.path.join(output_dir, "style_diagnosis_baseline_summary.txt")

        utils.save_model({"scaler": scaler, "kmeans": best_kmeans, "pca": pca}, model_path)
        utils.ensure_dir(output_dir)
        style_df.to_csv(csv_path, index=False)
        utils.save_text(summary_text, summary_path)

        assert os.path.exists(model_path)
        assert os.path.exists(csv_path)
        assert os.path.exists(summary_path)

        reloaded = joblib.load(model_path)
        assert "kmeans" in reloaded and "scaler" in reloaded and "pca" in reloaded

    print(
        f"OK: test_full_pipeline_with_fake_data "
        f"(best_k={best_k}, silhouette={best_score:.3f}, mismatch_rate={mismatch_rate:.2f})"
    )


if __name__ == "__main__":
    test_extract_match_style_rows_filters_forfeit()
    test_aggregate_user_style_filters_sparse_users()
    test_full_pipeline_with_fake_data()
    print("\n모든 스타일 진단 스모크 테스트 통과")
