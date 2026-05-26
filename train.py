"""
train.py — Physics-informed stacking ensemble (잔차 학습)

구조:
  1. Cauchy-robust NLS → anchor 위치 (NLOS에 강한 물리 추정)
  2. 101D feature 생성 (RTT, 잔차, 통계, rank, pairwise diff)
  3. Base models → anchor 기준 잔차(residual) 예측
  4. OOF stacking → meta model → 최종 잔차
  5. p_hat = anchor + meta_residual
"""

import numpy as np
import scipy.io as sio
import pickle
from scipy.optimize import least_squares
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone


# ── Physics helpers (main.py와 완전 동일) ─────────────────────────────────────

def inverse_distance_centroid(d, p_bs):
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
    return np.vstack([
        robust_anchor_one(d_hat[:, u], p_bs)
        for u in range(d_hat.shape[1])
    ])


def multilateration_wls(d, p_bs):
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
    N = d_hat.shape[1]
    multi = np.zeros((N, 2))
    for u in range(N):
        multi[u] = multilateration_wls(d_hat[:, u], p_bs)
    return multi


# ── Feature engineering (main.py와 완전 동일) ─────────────────────────────────

def make_features(d_hat, p_bs):
    """101D feature: RTT + Cauchy anchor + 잔차 + 통계 + rank + pairwise diff."""
    raw = d_hat.T       # (N, 18)
    bs  = p_bs.T        # (18, 2)

    anchor = compute_robust_anchors(d_hat, p_bs)                              # (N, 2)

    anchor_ranges = np.sqrt(
        np.sum((anchor[:, None, :] - bs[None, :, :]) ** 2, axis=2)
    )                                                                          # (N, 18)
    range_residual = raw - anchor_ranges                                       # (N, 18)

    multi = compute_multi(d_hat, p_bs)                                        # (N, 2)

    w = 1.0 / (raw + 1e-6)
    wcent = (
        (w[:, :, None] * bs[None, :, :]).sum(axis=1)
        / w.sum(axis=1, keepdims=True)
    )                                                                          # (N, 2)

    sorted_raw = np.sort(raw, axis=1)
    stats = np.column_stack([
        raw.mean(axis=1), raw.std(axis=1),
        raw.min(axis=1),  raw.max(axis=1),
        np.median(raw, axis=1),
        sorted_raw[:, :4], sorted_raw[:, -4:],
    ])                                                                         # (N, 13)

    rank = (
        np.argsort(np.argsort(raw, axis=1), axis=1).astype(float)
        / (raw.shape[1] - 1)
    )                                                                          # (N, 18)

    top5 = sorted_raw[:, :5]
    pair_diffs = np.column_stack([
        top5[:, j] - top5[:, i]
        for i in range(5) for j in range(i + 1, 5)
    ])                                                                         # (N, 10)

    X = np.hstack([
        raw, anchor, range_residual, np.abs(range_residual),
        stats, multi, wcent, rank, pair_diffs,
    ])                                                                         # (N, 101)

    return X, anchor


# ── OOF Stacking (잔차 학습) ───────────────────────────────────────────────────

def oof_stacking(base_protos, X, residual, n_splits=5, seed=42):
    """
    Out-of-Fold stacking.
    각 base model이 anchor 기준 잔차를 예측.
    OOF 예측을 쌓아 meta model의 입력으로 사용.
    Returns: oof_preds (N, n_models*2), trained_base_models (list)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n_models = len(base_protos)
    N = X.shape[0]
    oof_preds = np.zeros((N, n_models * 2))
    trained_folds = [[] for _ in range(n_splits)]

    for fold_idx, (tr, va) in enumerate(kf.split(X)):
        for m_idx, proto in enumerate(base_protos):
            m = clone(proto)
            m.fit(X[tr], residual[tr])
            pred = m.predict(X[va])           # (|va|, 2)
            oof_preds[va, m_idx*2:(m_idx+1)*2] = pred
            trained_folds[fold_idx].append(m)

    return oof_preds, trained_folds


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    data  = sio.loadmat('InF_DH_FR1.mat', squeeze_me=False)
    p_bs  = np.asarray(data['BS_positions'], dtype=float)
    d_hat = np.asarray(data['d_hat'],        dtype=float)
    p     = np.asarray(data['p'],            dtype=float)
    y     = p.T   # (N, 2)

    # ── 1. Features & anchor ──────────────────────────────────────────────────
    print('Feature 계산 중 (Cauchy anchor 포함, 시간 소요)...')
    X, anchor = make_features(d_hat, p_bs)       # (700, 101), (700, 2)
    residual   = y - anchor                       # 학습 target: anchor 기준 잔차

    anchor_rmse = float(np.mean(np.sqrt(np.sum((anchor - y)**2, axis=1))))
    print(f'[Cauchy anchor 단독] RMSE: {anchor_rmse:.4f} m')

    # ── 2. Base model 후보 ────────────────────────────────────────────────────
    base_protos = [
        MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.02, max_depth=5,
            subsample=0.7, random_state=42
        )),
        MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=7
        )),
        MultiOutputRegressor(RandomForestRegressor(
            n_estimators=300, max_features='sqrt',
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )),
        Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', MultiOutputRegressor(Ridge(alpha=1.0))),
        ]),
    ]

    # ── 3. OOF stacking ───────────────────────────────────────────────────────
    print('\nOOF stacking 학습 중...')
    oof_preds, trained_folds = oof_stacking(base_protos, X, residual)

    # OOF 성능 (anchor + oof_residual)
    n_models = len(base_protos)
    oof_residual_avg = oof_preds.reshape(700, n_models, 2).mean(axis=1)  # 단순 평균
    oof_pos_avg = anchor + oof_residual_avg
    oof_rmse_avg = float(np.mean(np.sqrt(np.sum((oof_pos_avg - y)**2, axis=1))))
    print(f'[OOF base 평균]      RMSE: {oof_rmse_avg:.4f} m')

    # ── 4. Meta model 학습 ────────────────────────────────────────────────────
    meta_proto = Pipeline([
        ('scaler', StandardScaler()),
        ('gbm', MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=42
        ))),
    ])
    meta_model = clone(meta_proto)
    meta_model.fit(oof_preds, residual)

    oof_meta_residual = meta_model.predict(oof_preds)
    oof_pos_meta = anchor + oof_meta_residual
    oof_rmse_meta = float(np.mean(np.sqrt(np.sum((oof_pos_meta - y)**2, axis=1))))
    print(f'[OOF meta stacking]  RMSE: {oof_rmse_meta:.4f} m  ← CV 추정치')

    # ── 5. 전체 데이터로 base 재학습 ──────────────────────────────────────────
    print('\n전체 데이터로 base model 재학습 중...')
    final_base_models = []
    for proto in base_protos:
        m = clone(proto)
        m.fit(X, residual)
        final_base_models.append(m)

    # train RMSE (참고용)
    final_preds = np.hstack([m.predict(X) for m in final_base_models])
    final_meta  = meta_model.predict(final_preds)
    train_pos   = anchor + final_meta
    train_rmse  = float(np.mean(np.sqrt(np.sum((train_pos - y)**2, axis=1))))
    print(f'[Train RMSE 참고]    RMSE: {train_rmse:.4f} m')

    # ── 6. 저장 ───────────────────────────────────────────────────────────────
    payload = {
        'type':         'stacked_residual',
        'base_models':  final_base_models,
        'meta_model':   meta_model,
        'oof_rmse':     oof_rmse_meta,
        'anchor_rmse':  anchor_rmse,
    }
    with open('model.pkl', 'wb') as f:
        pickle.dump(payload, f)

    print(f'\n=> 저장 완료: model.pkl')
    print(f'   OOF RMSE (hidden test 추정): {oof_rmse_meta:.4f} m')
    print(f'   Cauchy anchor baseline:       {anchor_rmse:.4f} m')


if __name__ == '__main__':
    main()
