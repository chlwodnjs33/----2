import numpy as np
import scipy.io as sio
import pickle
import os
from pathlib import Path
from scipy.optimize import least_squares


# ── Physics helpers ────────────────────────────────────────────────────────

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


def inverse_distance_centroid(d, p_bs):
    w = 1.0 / (d + 1e-6)
    return (p_bs * w).sum(axis=1) / w.sum()


def robust_anchor_one(d, p_bs):
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
    w = 1.0 / (d[1:] + 1e-6); W = np.diag(w)
    try:
        return np.linalg.solve(A.T @ W @ A, A.T @ (W @ rhs))
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, rhs, rcond=None)[0]


def compute_multi(d_hat, p_bs):
    N = d_hat.shape[1]; multi = np.zeros((N, 2))
    for u in range(N): multi[u] = multilateration_wls(d_hat[:, u], p_bs)
    return multi


def compute_sub_anchors(d_hat, p_bs, k):
    N = d_hat.shape[1]; raw = d_hat.T; result = np.zeros((N, 2))
    for u in range(N):
        idx = np.argsort(raw[u])[:k]
        result[u] = robust_anchor_one(d_hat[idx, u], p_bs[:, idx])
    return result


# ── Feature engineering (train.py와 완전 동일) ────────────────────────────

def make_features(d_hat, p_bs):
    """
    187D feature vector (train.py와 동일 — 두 파일 반드시 동기화 유지).
    """
    raw = d_hat.T; bs = p_bs.T

    anchor = compute_robust_anchors(d_hat, p_bs)

    anchor_ranges = np.sqrt(
        np.sum((anchor[:, None, :] - bs[None, :, :]) ** 2, axis=2)
    )
    range_residual = raw - anchor_ranges

    multi = compute_multi(d_hat, p_bs)

    w_c = 1.0 / (raw + 1e-6)
    wcent = (
        (w_c[:, :, None] * bs[None, :, :]).sum(axis=1)
        / w_c.sum(axis=1, keepdims=True)
    )

    sorted_raw = np.sort(raw, axis=1)
    stats = np.column_stack([
        raw.mean(1), raw.std(1), raw.min(1), raw.max(1),
        np.median(raw, 1), sorted_raw[:, :4], sorted_raw[:, -4:],
    ])

    rank = (
        np.argsort(np.argsort(raw, axis=1), axis=1).astype(float)
        / (raw.shape[1] - 1)
    )

    top5 = sorted_raw[:, :5]
    pair_diffs_5 = np.column_stack([
        top5[:, j] - top5[:, i]
        for i in range(5) for j in range(i + 1, 5)
    ])

    irls_w = 1.0 / (np.abs(range_residual) + 5.0)
    irls_w_norm = irls_w / (irls_w.sum(axis=1, keepdims=True) + 1e-8)

    sa4 = compute_sub_anchors(d_hat, p_bs, k=4)
    sa6 = compute_sub_anchors(d_hat, p_bs, k=6)
    sa8 = compute_sub_anchors(d_hat, p_bs, k=8)

    all_x = np.column_stack([anchor[:, 0], multi[:, 0], wcent[:, 0],
                              sa4[:, 0], sa6[:, 0], sa8[:, 0]])
    all_y = np.column_stack([anchor[:, 1], multi[:, 1], wcent[:, 1],
                              sa4[:, 1], sa6[:, 1], sa8[:, 1]])
    uncertainty = np.column_stack([
        all_x.std(axis=1),
        all_y.std(axis=1),
        np.sqrt(np.sum((anchor - multi) ** 2, axis=1)),
    ])

    top8 = sorted_raw[:, :8]
    pair_diffs_8 = np.column_stack([
        top8[:, j] - top8[:, i]
        for i in range(8) for j in range(i + 1, 8)
    ])
    ratio_8 = np.column_stack([
        (top8[:, j] - top8[:, i]) / (top8[:, i] + top8[:, j] + 1e-6)
        for i in range(8) for j in range(i + 1, 8)
    ])

    sa4_res = np.sqrt(np.sum((sa4 - anchor) ** 2, axis=1, keepdims=True))
    sa6_res = np.sqrt(np.sum((sa6 - anchor) ** 2, axis=1, keepdims=True))
    sa8_res = np.sqrt(np.sum((sa8 - anchor) ** 2, axis=1, keepdims=True))

    X = np.hstack([
        raw, anchor, range_residual, np.abs(range_residual),
        stats, multi, wcent, rank, pair_diffs_5,
        irls_w_norm, sa4, sa6, sa8, uncertainty,
        pair_diffs_8, ratio_8,
        sa4_res, sa6_res, sa8_res,
    ])  # (N, 187)

    return X, anchor


# ── Inference ───────────────────────────────────────────────────────────────

def your_algorithm(d_hat, p_bs):
    """
    PAANE: Physics-Anchored Adaptive NLOS Ensemble.
    model.pkl 로드 후 p_hat (2, num_user) 반환.
    """
    model_path = Path(__file__).parent.resolve() / 'model.pkl'
    with open(model_path, 'rb') as f:
        saved = pickle.load(f)

    X, anchor = make_features(d_hat, p_bs)

    mtype = saved['type']

    if mtype == 'paane_v1':
        base_res = np.hstack([m.predict(X) for m in saved['base_models']])
        residual = saved['meta_model'].predict(base_res)
        p_hat = (anchor + residual).T

    elif mtype == 'stacked_residual':
        base_res = np.hstack([m.predict(X) for m in saved['base_models']])
        residual = saved['meta_model'].predict(base_res)
        p_hat = (anchor + residual).T

    elif mtype == 'stacked':
        base_preds = np.hstack([m.predict(X) for m in saved['base_models']])
        p_hat = saved['meta_model'].predict(base_preds).T

    else:
        p_hat = saved['model'].predict(X).T

    return p_hat


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).parent.resolve()
    candidates = [
        script_dir / 'DH_FR1.mat',
        script_dir / 'InF_DH_FR1.mat',
    ]
    mat_path = next((str(p) for p in candidates if p.exists()), None)
    if mat_path is None:
        raise FileNotFoundError(
            'No .mat file found. Tried:\n' +
            '\n'.join(f'  {p}' for p in candidates)
        )

    p_bs, d_hat, _ = load_positioning_mat(mat_path)

    num_user = d_hat.shape[1]
    p_hat = your_algorithm(d_hat, p_bs)
    p_hat = np.asarray(p_hat, dtype=float)

    if p_hat.shape != (2, num_user):
        raise ValueError(f'p_hat must have shape (2, {num_user}), got {p_hat.shape}.')

    return p_hat


if __name__ == '__main__':
    import time

    script_dir = Path(__file__).parent.resolve()
    candidates = [script_dir / 'DH_FR1.mat', script_dir / 'InF_DH_FR1.mat']
    mat_path = next((str(p) for p in candidates if p.exists()), None)
    if mat_path is None:
        raise FileNotFoundError('No .mat file found.')

    import scipy.io as _sio
    _raw  = _sio.loadmat(mat_path, squeeze_me=False)
    if 'p_bs' in _raw:
        p_bs = np.asarray(_raw['p_bs'], dtype=float)
    else:
        p_bs = np.asarray(_raw['BS_positions'], dtype=float)
    d_hat = np.asarray(_raw['d_hat'], dtype=float)
    p_gt  = np.asarray(_raw['p'],     dtype=float) if 'p' in _raw else None

    t0    = time.time()
    p_hat = your_algorithm(d_hat, p_bs)
    p_hat = np.asarray(p_hat, dtype=float)
    elapsed = time.time() - t0

    print(f'\n{"="*45}')
    print(f'  p_hat shape : {p_hat.shape}')
    print(f'  실행 시간   : {elapsed:.2f} 초')

    if p_gt is not None:
        errors = np.sqrt(np.sum((p_hat - p_gt) ** 2, axis=0))
        print(f'\n  ── 측위 오차 (train set {d_hat.shape[1]}명) ──')
        print(f'  Mean   : {errors.mean():.4f} m')
        print(f'  Median : {np.median(errors):.4f} m')
        print(f'  Std    : {errors.std():.4f} m')
        print(f'  90th%  : {np.percentile(errors, 90):.4f} m')
        print(f'  Max    : {errors.max():.4f} m')
    else:
        print('  (정답 없음 — 채점기 환경)')
    print(f'{"="*45}\n')
