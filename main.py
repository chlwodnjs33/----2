import numpy as np
import scipy.io as sio
import pickle
import os
from pathlib import Path
from scipy.optimize import least_squares


# ── Data loading ────────────────────────────────────────────────────────────

def load_positioning_mat(path):
    data = sio.loadmat(path, squeeze_me=False)
    if 'p_bs' in data:
        p_bs = np.asarray(data['p_bs'], dtype=float)
    elif 'BS_positions' in data:
        p_bs = np.asarray(data['BS_positions'], dtype=float)
    else:
        raise KeyError("MAT file must contain 'p_bs' or 'BS_positions'.")
    d_hat = np.asarray(data['d_hat'], dtype=float)
    p = np.asarray(data['p'], dtype=float) if 'p' in data else None
    return p_bs, d_hat, p


# ── Physics-based positioning helpers ───────────────────────────────────────

def inverse_distance_centroid(d, p_bs):
    """역거리 가중 중심."""
    w = 1.0 / (d + 1e-6)
    return (p_bs * w).sum(axis=1) / w.sum()


def robust_anchor_one(d, p_bs):
    """Cauchy-robust 비선형 최소제곱으로 단일 사용자 위치 추정."""
    bs = p_bs.T
    lo = p_bs.min(axis=1)
    hi = p_bs.max(axis=1)
    margin = np.maximum(0.2 * (hi - lo), 20.0)
    x0 = np.clip(inverse_distance_centroid(d, p_bs), lo - margin, hi + margin)

    def residual(x):
        return np.sqrt(np.sum((bs - x) ** 2, axis=1)) - d

    result = least_squares(
        residual, x0,
        bounds=(lo - margin, hi + margin),
        loss="cauchy", f_scale=5.0, max_nfev=80,
    )
    return result.x


def compute_robust_anchors(d_hat, p_bs):
    """전체 사용자에 대해 robust anchor 계산 → (N, 2)."""
    return np.vstack([
        robust_anchor_one(d_hat[:, u], p_bs)
        for u in range(d_hat.shape[1])
    ])


def multilateration_wls(d, p_bs):
    """가중 최소제곱 삼변측량."""
    x0, y0, d0 = p_bs[0, 0], p_bs[1, 0], d[0]
    A = 2 * (p_bs[:, 1:] - p_bs[:, :1]).T
    rhs = d0**2 - d[1:]**2 + np.sum(p_bs[:, 1:]**2, axis=0) - (x0**2 + y0**2)
    w = 1.0 / (d[1:] + 1e-6)
    W = np.diag(w)
    try:
        return np.linalg.solve(A.T @ W @ A, A.T @ (W @ rhs))
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, rhs, rcond=None)[0]


def compute_multi(d_hat, p_bs):
    """전체 사용자에 대해 WLS 삼변측량 → (N, 2)."""
    N = d_hat.shape[1]
    multi = np.zeros((N, 2))
    for u in range(N):
        multi[u] = multilateration_wls(d_hat[:, u], p_bs)
    return multi


# ── Feature engineering (train.py와 완전 동일) ────────────────────────────

def make_features(d_hat, p_bs):
    """
    101차원 feature vector 생성.
    구성:
      raw RTT         18  (기지국별 거리 측정값)
      anchor          2   (Cauchy-robust 물리 추정 위치)
      range_residual  18  (anchor 기준 예측 거리와의 차이)
      abs_residual    18  (절댓값 잔차)
      statistics      13  (mean, std, min, max, median, top-4 small, top-4 large)
      WLS multi       2   (가중 삼변측량 위치)
      weighted_cent   2   (역거리 가중 중심)
      rank            18  (각 BS까지 정규화 순위: 0=가장 가까운, 1=가장 먼)
      pair_diffs      10  (가장 가까운 5개 BS 거리 쌍 차이, C(5,2)=10)
      ─────────────────
      합계           101
    """
    raw = d_hat.T           # (N, 18)
    bs = p_bs.T             # (18, 2)

    # 1) Cauchy-robust anchor
    anchor = compute_robust_anchors(d_hat, p_bs)                              # (N, 2)

    # 2) Range residual
    anchor_ranges = np.sqrt(
        np.sum((anchor[:, None, :] - bs[None, :, :]) ** 2, axis=2)
    )                                                                          # (N, 18)
    range_residual = raw - anchor_ranges                                       # (N, 18)

    # 3) WLS Multilateration
    multi = compute_multi(d_hat, p_bs)                                        # (N, 2)

    # 4) Inverse-distance weighted centroid
    w = 1.0 / (raw + 1e-6)                                                    # (N, 18)
    wcent = (
        (w[:, :, None] * bs[None, :, :]).sum(axis=1)
        / w.sum(axis=1, keepdims=True)
    )                                                                          # (N, 2)

    # 5) Order statistics
    sorted_raw = np.sort(raw, axis=1)                                         # (N, 18)
    stats = np.column_stack([
        raw.mean(axis=1),
        raw.std(axis=1),
        raw.min(axis=1),
        raw.max(axis=1),
        np.median(raw, axis=1),
        sorted_raw[:, :4],   # 4 smallest RTT
        sorted_raw[:, -4:],  # 4 largest  RTT
    ])                                                                         # (N, 13)

    # 6) Normalized distance rank (0=closest, 1=farthest)
    rank = (
        np.argsort(np.argsort(raw, axis=1), axis=1).astype(float)
        / (raw.shape[1] - 1)
    )                                                                          # (N, 18)

    # 7) Pairwise RTT differences of 5 closest BSes  C(5,2)=10
    top5 = sorted_raw[:, :5]                                                  # (N, 5)
    pair_diffs = np.column_stack([
        top5[:, j] - top5[:, i]
        for i in range(5) for j in range(i + 1, 5)
    ])                                                                         # (N, 10)

    X = np.hstack([
        raw,                    # 18
        anchor,                 # 2
        range_residual,         # 18
        np.abs(range_residual), # 18
        stats,                  # 13
        multi,                  # 2
        wcent,                  # 2
        rank,                   # 18
        pair_diffs,             # 10
    ])                          # → (N, 101)

    return X, anchor


# ── Inference ───────────────────────────────────────────────────────────────

def your_algorithm(d_hat, p_bs):
    """
    Physics-informed stacking 앙상블 측위 (잔차 학습).
    model.pkl 로드 후 p_hat (2, num_user) 반환.
    """
    model_path = Path(__file__).parent.resolve() / 'model.pkl'
    with open(model_path, 'rb') as f:
        saved = pickle.load(f)

    mtype = saved['type']
    X, anchor = make_features(d_hat, p_bs)

    if mtype == 'stacked_residual':
        # ★ 잔차 학습: base 예측(잔차) → meta → 최종 = anchor + 잔차
        base_res = np.hstack([m.predict(X) for m in saved['base_models']])  # (N, n_models*2)
        residual = saved['meta_model'].predict(base_res)                    # (N, 2)
        p_hat = (anchor + residual).T                                       # (2, N)

    elif mtype == 'stacked':
        # 이전 버전 호환
        base_preds = np.hstack([m.predict(X) for m in saved['base_models']])
        p_hat = saved['meta_model'].predict(base_preds).T

    else:
        # 구버전 sklearn 단일 모델 fallback
        p_hat = saved['model'].predict(X).T

    return p_hat


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    # 스크립트 위치 기준으로 .mat 파일 탐색
    # 채점기는 'DH_FR1.mat'을 같은 폴더에 배치, 로컬엔 'InF_DH_FR1.mat' 사용
    script_dir = Path(__file__).parent.resolve()
    candidates = [
        script_dir / 'DH_FR1.mat',
        script_dir / 'InF_DH_FR1.mat',  # 로컬 테스트용 fallback
    ]
    mat_path = next((str(p) for p in candidates if p.exists()), None)
    if mat_path is None:
        raise FileNotFoundError(
            f"No .mat file found. Tried:\n" +
            "\n".join(f"  {p}" for p in candidates)
        )

    p_bs, d_hat, _ = load_positioning_mat(mat_path)

    num_user = d_hat.shape[1]
    p_hat = your_algorithm(d_hat, p_bs)
    p_hat = np.asarray(p_hat, dtype=float)

    if p_hat.shape != (2, num_user):
        raise ValueError(f"p_hat must have shape (2, {num_user}), got {p_hat.shape}.")

    return p_hat


if __name__ == '__main__':
    import time

    script_dir = Path(__file__).parent.resolve()
    candidates = [
        script_dir / 'DH_FR1.mat',
        script_dir / 'InF_DH_FR1.mat',
    ]
    mat_path = next((str(p) for p in candidates if p.exists()), None)
    if mat_path is None:
        raise FileNotFoundError("No .mat file found.")

    # load_positioning_mat 으로 안전하게 로드
    import scipy.io as _sio
    _raw  = _sio.loadmat(mat_path, squeeze_me=False)
    p_bs  = np.asarray(_raw['p_bs'] if 'p_bs' in _raw else _raw['BS_positions'], dtype=float)
    d_hat = np.asarray(_raw['d_hat'], dtype=float)
    p_gt  = np.asarray(_raw['p'], dtype=float) if 'p' in _raw else None

    t0    = time.time()
    p_hat = your_algorithm(d_hat, p_bs)
    p_hat = np.asarray(p_hat, dtype=float)
    elapsed = time.time() - t0

    print(f"\n{'='*40}")
    print(f"  p_hat shape : {p_hat.shape}")
    print(f"  실행 시간   : {elapsed:.2f} 초")

    if p_gt is not None:
        errors = np.sqrt(np.sum((p_hat - p_gt) ** 2, axis=0))
        print(f"\n  ── 측위 오차 (train set 700명) ──")
        print(f"  Mean   : {errors.mean():.4f} m")
        print(f"  Median : {np.median(errors):.4f} m")
        print(f"  Std    : {errors.std():.4f} m")
        print(f"  90th%  : {np.percentile(errors, 90):.4f} m")
        print(f"  Max    : {errors.max():.4f} m")
        print(f"\n  ※ 실제 채점은 hidden 300명 기준 (OOF ≈ 5.18 m 예상)")
    else:
        print("  (정답 없음 — 채점기 환경)")
    print(f"{'='*40}\n")

