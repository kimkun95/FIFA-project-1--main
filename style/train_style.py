"""
플레이스타일 진단(K-means 군집화, 비지도학습) baseline 실행.

데이터 파싱/feature 조립은 preprocess_style.py, 저장은 utils.py에 맡기고, 여기서는
표준화 -> k 탐색 -> 최종 군집화 -> 결과 저장만 담당한다.

CLAUDE.md 원칙 #4("규칙 기반으로 미리 유형 개수를 정하지 않는다")를 그대로 따른다:
k=2~6을 전부 시도해 실루엣 점수가 가장 높은 k를 사후에 고르고, 각 군집에 사람이 붙일
이름(예: "스루패스 위주形")은 이 스크립트가 정하지 않는다 — cluster_summary에 군집별
feature 평균만 남겨서, 그 숫자를 보고 팀이 나중에 라벨을 붙인다.

군집화 자체는 표준화된 6차원 feature 공간 전체에서 수행한다(정보 손실을 피하기 위해).
PCA(2차원)는 군집화에 쓰지 않고, 대시보드 시각화용 좌표를 뽑는 용도로만 별도로 돌린다.
"""

import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import preprocess_style as preprocess
import utils

# ============ CONFIG ============
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "style")
MODEL_DIR = os.path.join(REPO_ROOT, "models")
RANDOM_SEED = 42
K_RANGE = range(2, 7)  # CLAUDE.md 원칙: k=2~6을 전부 시도
# ====================================================


def select_best_k(x_scaled, k_range=K_RANGE):
    """k별로 KMeans를 돌려 실루엣 점수를 계산하고, 가장 높은 k를 고른다.

    반환값: (best_k, {k: (labels, silhouette_score)} 전체 결과)
    """
    results = {}
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(x_scaled)
        score = silhouette_score(x_scaled, labels)
        results[k] = (labels, score, kmeans)

    best_k = max(results, key=lambda k: results[k][1])
    return best_k, results


def build_cluster_summary(style_df, feature_columns, label_column="style_type"):
    """군집별 표본 수와 feature 평균을 낸다 — 사후 라벨링(이름 붙이기)에 쓰는 참고 표."""
    summary = style_df.groupby(label_column)[feature_columns].mean()
    summary["n_users"] = style_df.groupby(label_column).size()
    return summary.reset_index()


def build_summary_text(best_k, k_search_results, cluster_summary, n_users, min_matches):
    lines = [
        "=== 플레이스타일 진단(K-means) Baseline 요약 ===",
        f"feature: {preprocess.FEATURE_COLUMNS}",
        f"유저 최소 표본 경기 수(MIN_MATCHES_PER_USER): {min_matches}",
        f"군집화 대상 유저 수: {n_users}",
        "",
        "[k별 실루엣 점수] (표준화된 6차원 feature 공간 기준)",
    ]
    for k in sorted(k_search_results):
        _, score, _ = k_search_results[k]
        marker = "  <- 선택" if k == best_k else ""
        lines.append(f"  k={k}: silhouette={score:.4f}{marker}")

    lines.append("")
    lines.append(f"[선택된 k] {best_k}")
    lines.append("")
    lines.append("[군집별 feature 평균] (이 숫자를 보고 사후에 스타일 이름을 붙인다 — "
                  "예: through_pass_ratio가 높은 군집 -> '스루패스 위주형')")
    lines.append(cluster_summary.to_string(index=False))
    lines.append("")
    lines.append("[한계] 최소 표본 경기 수 하한선(MIN_MATCHES_PER_USER)이 팀 논의로 "
                  "확정되지 않아 임시값을 쓰고 있다 (preprocess_style.py 참고).")
    lines.append("[한계] dribble_intensity 정규화 방식(경기당 평균 야드)이 최종 확정이 "
                  "아니며, 실제 데이터 분포를 본 뒤 재검토가 필요하다.")
    lines.append("[한계] 아직 실제 matches_style.jsonl 데이터가 없어(수집 스크립트 "
                  "collect_style_matches.py 실행 전) 이 baseline은 구조 검증용 뼈대이고, "
                  "실제 군집 결과/최적 k는 데이터가 모인 뒤 다시 확인해야 한다.")
    return "\n".join(lines)


def main():
    print("1) matches_style.jsonl 로드 및 (match, ouid) 행 추출")
    matches = preprocess.load_matches(preprocess.MATCHES_FILE)
    rows_df = preprocess.extract_match_style_rows(matches)

    print("2) 유저 단위 집계 및 6개 비율 feature 계산")
    style_df = preprocess.aggregate_user_style(rows_df)
    if len(style_df) < max(K_RANGE):
        raise ValueError(
            f"군집화 대상 유저가 {len(style_df)}명뿐이라 k 최대값({max(K_RANGE)})보다 "
            "적다. 유저 수를 늘리거나 K_RANGE를 줄여야 한다."
        )

    print("3) 표준화 (StandardScaler)")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(style_df[preprocess.FEATURE_COLUMNS].astype(float))

    print("4) k=2~6 탐색 및 실루엣 점수 기준 최적 k 선택")
    best_k, k_search_results = select_best_k(x_scaled)
    labels, best_score, best_kmeans = k_search_results[best_k]
    style_df = style_df.copy()
    style_df["style_type"] = labels
    print(f"  선택된 k: {best_k} (silhouette={best_score:.4f})")

    print("5) 시각화용 2차원 PCA 좌표 계산 (군집화에는 쓰지 않음)")
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(x_scaled)
    style_df["pca_x"] = coords[:, 0]
    style_df["pca_y"] = coords[:, 1]

    print("6) 결과 저장")
    cluster_summary = build_cluster_summary(style_df, preprocess.FEATURE_COLUMNS)
    summary_text = build_summary_text(
        best_k, k_search_results, cluster_summary, len(style_df),
        preprocess.MIN_MATCHES_PER_USER,
    )

    utils.save_model(
        {"scaler": scaler, "kmeans": best_kmeans, "pca": pca},
        os.path.join(MODEL_DIR, "style_diagnosis_baseline.joblib"),
    )
    utils.ensure_dir(OUTPUT_DIR)
    style_df.to_csv(os.path.join(OUTPUT_DIR, "user_style_profile.csv"), index=False)
    utils.save_text(summary_text, os.path.join(OUTPUT_DIR, "style_diagnosis_baseline_summary.txt"))

    print(f"  [저장] 모델(scaler+kmeans+pca) -> {MODEL_DIR}/style_diagnosis_baseline.joblib")
    print(f"  [저장] 유저별 스타일 프로필 -> {OUTPUT_DIR}/user_style_profile.csv")
    print(f"  [저장] 요약 -> {OUTPUT_DIR}/style_diagnosis_baseline_summary.txt")


if __name__ == "__main__":
    main()
