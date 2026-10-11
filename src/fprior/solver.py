"""Finite-grid multi-kernel ridge; numerical failures stop selection."""
from __future__ import annotations
import numpy as np
from .io import finite_array


def simplex_grid(n_kernels: int, denominator: int, *, active=None):
    if n_kernels < 1 or denominator < 1 or not isinstance(n_kernels, int) or not isinstance(denominator, int):
        raise ValueError('SIMPLEX_GRID_INVALID')
    mask = np.ones(n_kernels, dtype=bool) if active is None else np.asarray(active)
    if mask.shape != (n_kernels,) or mask.dtype != np.bool_:
        raise ValueError('ACTIVE_KERNEL_MASK_INVALID')
    indices = np.flatnonzero(mask)
    if not len(indices):
        return [tuple(np.zeros(n_kernels))]
    def compositions(total, size):
        if size == 1:
            yield (total,)
        else:
            for first in range(total + 1):
                for rest in compositions(total - first, size - 1):
                    yield (first,) + rest
    grid = []
    for counts in compositions(denominator, len(indices)):
        weights = np.zeros(n_kernels)
        weights[indices] = np.asarray(counts) / denominator
        grid.append(tuple(weights))
    return grid


def _combine(kernels, weights):
    if not kernels or len(kernels) != len(weights):
        raise ValueError('KERNEL_WEIGHT_SHAPE_INVALID')
    arrays = [finite_array(kernel, 2, 'kernel') for kernel in kernels]
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError('KERNEL_SHAPE_INVALID')
    weights = finite_array(weights, 1, 'weights')
    if (weights < 0).any() or not (np.isclose(weights.sum(), 1) or np.all(weights == 0)):
        raise ValueError('KERNEL_WEIGHTS_INVALID')
    return sum(weight * array for weight, array in zip(weights, arrays))


def solve_weight(kernels_train, kernels_cross, weights, rho_grid, y_train, *, relative_residual_max, spectral_relative_tolerance):
    y = finite_array(y_train, 1, 'training target')
    kernel = _combine(kernels_train, weights)
    cross = _combine(kernels_cross, weights)
    n = len(y)
    if not n or kernel.shape != (n, n) or cross.shape[1] != n:
        raise ValueError('KERNEL_TARGET_SHAPE_INVALID')
    if not np.isfinite(relative_residual_max) or not np.isfinite(spectral_relative_tolerance) or min(relative_residual_max, spectral_relative_tolerance) <= 0:
        raise ValueError('NUMERICAL_TOLERANCE_INVALID')
    scale = max(1.0, np.max(np.abs(kernel)))
    if np.max(np.abs(kernel - kernel.T)) > spectral_relative_tolerance * scale:
        raise ValueError('KERNEL_ASYMMETRIC')
    if np.max(np.abs(kernel.mean(axis=0))) > spectral_relative_tolerance * scale:
        raise ValueError('KERNEL_NOT_TRAINING_CENTERED')
    eigenvalues, vectors = np.linalg.eigh((kernel + kernel.T) / 2)
    if eigenvalues.min() < -spectral_relative_tolerance * max(1.0, np.max(np.abs(eigenvalues))):
        raise ValueError('KERNEL_NOT_PSD')
    rhos = finite_array(rho_grid, 1, 'rho grid')
    if not len(rhos) or (rhos <= 0).any():
        raise ValueError('RIDGE_GRID_INVALID')
    centered = y - y.mean()
    projected = vectors.T @ centered
    records = []
    for rho in rhos:
        divisor = eigenvalues + n * rho
        if (divisor <= 0).any():
            raise ValueError('RIDGE_SYSTEM_NOT_POSITIVE_DEFINITE')
        alpha = vectors @ (projected / divisor)
        residual = float(np.linalg.norm(kernel @ alpha + n * rho * alpha - centered) / max(np.linalg.norm(centered), np.finfo(float).tiny))
        if not np.isfinite(residual) or residual > relative_residual_max:
            raise ValueError(f'NUMERICAL_FAILURE: rho={rho}, relative_residual={residual}')
        prediction = y.mean() + cross @ alpha
        if not np.isfinite(prediction).all():
            raise ValueError('PREDICTION_NONFINITE')
        records.append(dict(weights=tuple(weights), rho=float(rho), alpha=alpha,
                            y_mean=float(y.mean()), predictions=prediction,
                            relative_residual=residual, status='PASSED'))
    return records


def select_configuration(scores, *, tol):
    if not scores or not np.isfinite(tol) or tol < 0:
        raise ValueError('SELECTION_INPUT_INVALID')
    for score in scores:
        if score.get('status') != 'PASSED' or not np.isfinite(score['mse']) or score['mse'] < 0:
            raise ValueError('COMPARISON_INCOMPLETE')
        _combine([np.zeros((1, 1))] * len(score['weights']), score['weights'])
        if not np.isfinite(score['rho']) or score['rho'] <= 0:
            raise ValueError('SELECTION_RHO_INVALID')
    best = min(score['mse'] for score in scores)
    tied = [score for score in scores if score['mse'] <= best + tol * max(1, best)]
    return min(tied, key=lambda score: (np.count_nonzero(score['weights']), -score['weights'][0], -score['rho'], score['config_index']))


def predict(alpha, kernels_cross, weights, y_mean):
    alpha = finite_array(alpha, 1, 'alpha')
    kernel = _combine(kernels_cross, weights)
    if kernel.shape[1] != len(alpha) or not np.isfinite(y_mean):
        raise ValueError('PREDICTION_SHAPE_INVALID')
    result = y_mean + kernel @ alpha
    if not np.isfinite(result).all():
        raise ValueError('PREDICTION_NONFINITE')
    return result
