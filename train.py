"""
train.py — Physics-Anchored Adaptive NLOS Ensemble (PAANE)

알고리즘 구조:
  1. Cauchy-robust NLS → main anchor  (NLOS 강건 물리 추정)
  2. Sub-anchor k=4,6,8 → 다중 해상도 물리 추정  (앙상블 불확실성 정보)
  3. 187D 피처 생성
       raw RTT (18) · anchor (2) · range_residual (18) · abs_residual (18)
       statistics (13) · WLS multi (2) · IDW centroid (2) · rank (18)
       pair_diffs top-5 (10)
       ── 신규 ──────────────────────────────────────────────────────────
       IRLS 신뢰도 가중치 (18): 1/(|range_residual|+5), 정규화
       sub-anchor 위치 k=4,6,8 (6)
       앵커 불확실성 (3): x-std, y-std, |anchor-multi|
       top-8 쌍별 차이 (28): C(8,2), TDOA-like
       top-8 쌍별 비율 (28): 무차원 NLOS ratio
       sub-anchor 잔차 (3)
  4. OOF 5-fold stacking (잔차 학습)
       Base: LightGBM×4 + RandomForest + Ridge
       Meta: LightGBM (보수적)
  5. p_hat = anchor + meta_residual
"""

import numpy as np
import scipy.io as sio
import pickle
import time
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import lightgbm as lgb


# ── Physics helpers ────────────────────────────────────────────────────────

def inverse_distance_centroid(d, p_bs):
    w = 1.0 / (d + 1e-6)
    return (p_bs * w).sum(axis=1) / w.sum()


def robust_anchor_one(d, p_bs):
    """Cauchy-robust 비선형 최소제곱으로 단일 사용자 위치 추정."""
    bs = p_bs.T
    lo = p_bs.min(axis=1); hi = p_bs.max(axis=1)
    margin = np.maximum(0.2 * (hi - lo), 20.0)
    x0 = np.clip(inverse_distance_centroid(d, p_bs), lo - margin, hi + margin)
    def residual(x): return np.sqrt(np.sum((bs - x) ** 2, axis=1)) - d
    r = least_squares(residual, x0, bounds=(lo - margin, hi + margin),
                      loss='cauchy', f_scale=5.0, max_nfev=80)
    return r.x


def compute_robust_anchors(d_hat, p_bs):
    return np.vstack([robust_anchor_one(d_hat[:, u], p_bs) for u in range(d_hat.shape[1])])


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
    N = d_hat.shape[1]; multi = np.zeros((N, 2))
    for u in range(N): multi[u] = multilateration_wls(d_hat[:, u], p_bs)
    return multi


def compute_sub_anchors(d_hat, p_bs, k):
    """가장 가까운 k개 BS만 사용해 Cauchy anchor 계산."""
    N = d_hat.shape[1]
    raw = d_hat.T
    result = np.zeros((N, 2))
    for u in range(N):
        idx = np.argsort(raw[u])[:k]
        result[u] = robust_anchor_one(d_hat[idx, u], p_bs[:, idx])
    return result


# ── Feature engineering (187D) ─────────────────────────────────────────────

def make_features(d_hat, p_bs):
    """
    187D feature vector.
    구성:
      raw RTT           18
      anchor            2   (Cauchy-robust 물리 추정)
      range_residual    18
      abs_residual      18
      statistics        13
      WLS multi         2
      weighted_cent     2
      rank              18
      pair_diffs top-5  10
      ── 신규 ────────────────────────────────────────────────
      IRLS 신뢰도 가중치 18  (range_residual 기반 BS별 신뢰도)
      sub-anchor k=4,6,8  6  (다중 해상도 물리 앙상블)
      앵커 불확실성       3   (여러 추정치 간 불일치 → 신뢰도 지표)
      top-8 쌍별 차이    28  (C(8,2), TDOA-like NLOS 상쇄)
      top-8 쌍별 비율    28  (무차원, [-1,1])
      sub-anchor 잔차     3  (각 sub-anchor와 main anchor의 거리)
      ────────────────────
      합계              187
    """
    raw = d_hat.T           # (N, 18)
    bs  = p_bs.T            # (18, 2)

    # ── 기존 피처 ────────────────────────────────────────────────────────
    anchor = compute_robust_anchors(d_hat, p_bs)                   # (N, 2)

    anchor_ranges = np.sqrt(
        np.sum((anchor[:, None, :] - bs[None, :, :]) ** 2, axis=2)
    )                                                               # (N, 18)
    range_residual = raw - anchor_ranges                           # (N, 18)

    multi = compute_multi(d_hat, p_bs)                             # (N, 2)

    w_c = 1.0 / (raw + 1e-6)
    wcent = (
        (w_c[:, :, None] * bs[None, :, :]).sum(axis=1)
        / w_c.sum(axis=1, keepdims=True)
    )                                                               # (N, 2)

    sorted_raw = np.sort(raw, axis=1)
    stats = np.column_stack([
        raw.mean(1), raw.std(1), raw.min(1), raw.max(1),
        np.median(raw, 1), sorted_raw[:, :4], sorted_raw[:, -4:],
    ])                                                              # (N, 13)

    rank = (
        np.argsort(np.argsort(raw, axis=1), axis=1).astype(float)
        / (raw.shape[1] - 1)
    )                                                               # (N, 18)

    top5 = sorted_raw[:, :5]
    pair_diffs_5 = np.column_stack([
        top5[:, j] - top5[:, i]
        for i in range(5) for j in range(i + 1, 5)
    ])                                                              # (N, 10)

    # ── 신규 피처 1: IRLS 신뢰도 가중치 (18D) ─────────────────────────
    # range_residual이 작을수록 해당 BS의 측정이 anchor와 일관적 → 신뢰 높음
    # 이 가중치는 ML이 "어떤 BS를 신뢰할지" 학습하도록 도움
    irls_w = 1.0 / (np.abs(range_residual) + 5.0)
    irls_w_norm = irls_w / (irls_w.sum(axis=1, keepdims=True) + 1e-8)  # (N, 18)

    # ── 신규 피처 2: Sub-anchor 위치 (6D) ────────────────────────────
    # 서로 다른 k개 BS 부분집합에서의 추정치 → 서로 다른 NLOS 환경 커버
    sa4 = compute_sub_anchors(d_hat, p_bs, k=4)                    # (N, 2)
    sa6 = compute_sub_anchors(d_hat, p_bs, k=6)                    # (N, 2)
    sa8 = compute_sub_anchors(d_hat, p_bs, k=8)                    # (N, 2)

    # ── 신규 피처 3: 앵커 불확실성 (3D) ──────────────────────────────
    # 여러 물리 추정치 간 불일치 = 위치 불확실성 지표
    all_x = np.column_stack([anchor[:, 0], multi[:, 0], wcent[:, 0],
                              sa4[:, 0], sa6[:, 0], sa8[:, 0]])
    all_y = np.column_stack([anchor[:, 1], multi[:, 1], wcent[:, 1],
                              sa4[:, 1], sa6[:, 1], sa8[:, 1]])
    uncertainty = np.column_stack([
        all_x.std(axis=1),                                          # anchor x-std
        all_y.std(axis=1),                                          # anchor y-std
        np.sqrt(np.sum((anchor - multi) ** 2, axis=1)),            # |anchor - multi|
    ])                                                              # (N, 3)

    # ── 신규 피처 4: Top-8 쌍별 차이 + 비율 (28D + 28D) ─────────────
    # TDOA-like: 공통 NLOS bias가 상쇄되어 상대적 신뢰도 정보 포함
    top8 = sorted_raw[:, :8]
    pair_diffs_8 = np.column_stack([
        top8[:, j] - top8[:, i]
        for i in range(8) for j in range(i + 1, 8)
    ])                                                              # (N, 28)
    ratio_8 = np.column_stack([
        (top8[:, j] - top8[:, i]) / (top8[:, i] + top8[:, j] + 1e-6)
        for i in range(8) for j in range(i + 1, 8)
    ])                                                              # (N, 28) ∈ [-1,1]

    # ── 신규 피처 5: Sub-anchor 잔차 (3D) ────────────────────────────
    sa4_res = np.sqrt(np.sum((sa4 - anchor) ** 2, axis=1, keepdims=True))
    sa6_res = np.sqrt(np.sum((sa6 - anchor) ** 2, axis=1, keepdims=True))
    sa8_res = np.sqrt(np.sum((sa8 - anchor) ** 2, axis=1, keepdims=True))

    X = np.hstack([
        raw,                     # 18
        anchor,                  # 2
        range_residual,          # 18
        np.abs(range_residual),  # 18
        stats,                   # 13
        multi,                   # 2
        wcent,                   # 2
        rank,                    # 18
        pair_diffs_5,            # 10
        # ── 신규 ─────────────────────────────
        irls_w_norm,             # 18
        sa4, sa6, sa8,           # 6
        uncertainty,             # 3
        pair_diffs_8,            # 28
        ratio_8,                 # 28
        sa4_res, sa6_res, sa8_res,  # 3
    ])  # → (N, 187)

    return X, anchor


# ── OOF Stacking ───────────────────────────────────────────────────────────

def oof_stacking(base_protos, X, residual, n_splits=5, seed=42):
    """
    Out-of-Fold stacking.
    각 base model이 anchor 기준 잔차를 예측.
    Returns: oof_preds (N, n_models*2), trained_base_models (list)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n_models = len(base_protos)
    N = X.shape[0]
    oof_preds = np.zeros((N, n_models * 2))
    trained_folds = [[] for _ in range(n_splits)]

    for fold_idx, (tr, va) in enumerate(kf.split(X)):
        print(f'  fold {fold_idx + 1}/{n_splits}...', end=' ', flush=True)
        for m_idx, proto in enumerate(base_protos):
            m = clone(proto)
            m.fit(X[tr], residual[tr])
            oof_preds[va, m_idx * 2:(m_idx + 1) * 2] = m.predict(X[va])
            trained_folds[fold_idx].append(m)
        print('done')

    return oof_preds, trained_folds


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    data  = sio.loadmat('InF_DH_FR1.mat', squeeze_me=False)
    if 'BS_positions' in data:
        p_bs = np.asarray(data['BS_positions'], dtype=float)
    else:
        p_bs = np.asarray(data['p_bs'], dtype=float)
    d_hat = np.asarray(data['d_hat'], dtype=float)
    p     = np.asarray(data['p'],     dtype=float)
    y     = p.T  # (N, 2)

    # ── 1. 피처 생성 ─────────────────────────────────────────────────────
    print('피처 계산 중 (sub-anchor 포함, 약 10초 소요)...')
    t0 = time.time()
    X, anchor = make_features(d_hat, p_bs)       # (700, 187), (700, 2)
    print(f'완료 {time.time()-t0:.1f}s  |  피처 shape: {X.shape}')

    residual = y - anchor    # 학습 target: anchor 기준 잔차

    anchor_rmse = float(np.mean(np.sqrt(np.sum((anchor - y) ** 2, axis=1))))
    print(f'[Cauchy anchor 단독] RMSE: {anchor_rmse:.4f} m')

    # ── 2. Base model 정의 ────────────────────────────────────────────────
    def lgb_base(n_est, lr, num_leaves, min_child, sub, col, ra, rl, seed):
        return MultiOutputRegressor(lgb.LGBMRegressor(
            n_estimators=n_est, learning_rate=lr, num_leaves=num_leaves,
            min_child_samples=min_child, subsample=sub, colsample_bytree=col,
            reg_alpha=ra, reg_lambda=rl, random_state=seed,
            n_jobs=-1, verbose=-1, force_col_wise=True,
        ))

    base_protos = [
        # LightGBM 4종 (하이퍼파라미터 다양성으로 앙상블 효과 극대화)
        lgb_base(3000, 0.02, 63,  20, 0.70, 0.50, 0.1, 1.5, 42),
        lgb_base(2000, 0.03, 31,  25, 0.80, 0.60, 0.5, 2.0,  7),
        lgb_base(4000, 0.01, 127, 15, 0.60, 0.40, 0.0, 1.0, 13),
        lgb_base(2500, 0.02, 63,  25, 0.75, 0.70, 0.2, 2.0, 99),
        # Random Forest: boosting과 상관없는 다양성 제공
        MultiOutputRegressor(RandomForestRegressor(
            n_estimators=500, max_features='sqrt',
            min_samples_leaf=2, random_state=42, n_jobs=-1,
        )),
        # Ridge: 선형 컴포넌트 포착
        Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', MultiOutputRegressor(Ridge(alpha=1.0))),
        ]),
    ]

    # ── 3. OOF stacking ───────────────────────────────────────────────────
    print('\nOOF stacking 학습 중...')
    t0 = time.time()
    oof_preds, trained_folds = oof_stacking(base_protos, X, residual)
    print(f'OOF 완료: {time.time()-t0:.1f}s')

    # OOF 성능 평가
    n_models = len(base_protos)
    oof_res_avg = oof_preds.reshape(700, n_models, 2).mean(axis=1)
    oof_pos_avg = anchor + oof_res_avg
    oof_rmse_avg = float(np.mean(np.sqrt(np.sum((oof_pos_avg - y) ** 2, axis=1))))
    print(f'[OOF base 평균]      RMSE: {oof_rmse_avg:.4f} m')

    # ── 4. Meta model 학습 ────────────────────────────────────────────────
    meta_proto = Pipeline([
        ('scaler', StandardScaler()),
        ('lgb', MultiOutputRegressor(lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.02, num_leaves=31,
            min_child_samples=30, subsample=0.8, colsample_bytree=0.6,
            reg_alpha=0.1, reg_lambda=1.5, random_state=42,
            n_jobs=-1, verbose=-1, force_col_wise=True,
        ))),
    ])
    meta_model = clone(meta_proto)
    meta_model.fit(oof_preds, residual)

    oof_meta_res = meta_model.predict(oof_preds)
    oof_pos_meta = anchor + oof_meta_res
    oof_rmse_meta = float(np.mean(np.sqrt(np.sum((oof_pos_meta - y) ** 2, axis=1))))
    oof_median    = float(np.median(np.sqrt(np.sum((oof_pos_meta - y) ** 2, axis=1))))
    oof_90th      = float(np.percentile(np.sqrt(np.sum((oof_pos_meta - y) ** 2, axis=1)), 90))
    print(f'[OOF meta stacking]  mean: {oof_rmse_meta:.4f} m  median: {oof_median:.4f} m  90th: {oof_90th:.4f} m  ← CV 추정치')

    # ── 5. 전체 데이터로 base 재학습 ──────────────────────────────────────
    print('\n전체 데이터로 base model 재학습 중...')
    t0 = time.time()
    final_base_models = []
    for i, proto in enumerate(base_protos):
        m = clone(proto)
        m.fit(X, residual)
        final_base_models.append(m)
        print(f'  base {i+1}/{len(base_protos)} done', end='\r')
    print(f'\n재학습 완료: {time.time()-t0:.1f}s')

    # train RMSE (과적합 확인용)
    final_preds = np.hstack([m.predict(X) for m in final_base_models])
    final_meta  = meta_model.predict(final_preds)
    train_pos   = anchor + final_meta
    train_rmse  = float(np.mean(np.sqrt(np.sum((train_pos - y) ** 2, axis=1))))
    print(f'[Train RMSE 참고]    RMSE: {train_rmse:.4f} m  (OOF와 차이가 크면 과적합)')

    # ── 6. 저장 ───────────────────────────────────────────────────────────
    payload = {
        'type':         'paane_v1',           # Physics-Anchored Adaptive NLOS Ensemble
        'base_models':  final_base_models,
        'meta_model':   meta_model,
        'oof_rmse':     oof_rmse_meta,
        'oof_median':   oof_median,
        'oof_90th':     oof_90th,
        'anchor_rmse':  anchor_rmse,
        'feature_dim':  X.shape[1],
    }
    with open('model.pkl', 'wb') as f:
        pickle.dump(payload, f)

    print(f'\n=> 저장 완료: model.pkl  ({X.shape[1]}D 피처)')
    print(f'   OOF mean  (hidden test 추정): {oof_rmse_meta:.4f} m')
    print(f'   OOF median:                   {oof_median:.4f} m')
    print(f'   OOF 90th%:                    {oof_90th:.4f} m')
    print(f'   Cauchy anchor baseline:       {anchor_rmse:.4f} m')
    print(f'   Train RMSE (과적합 참고):      {train_rmse:.4f} m')


if __name__ == '__main__':
    main()
