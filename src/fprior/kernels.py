"""Training-normalized linear and Gaussian control kernels."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .io import finite_array


def trace_normalize(b_train):
    b = finite_array(b_train, 2, 'training features')
    if not len(b):
        raise ValueError('EMPTY_TRAINING_PARTITION')
    c = float(np.sum(b * b) / len(b))
    if not np.isfinite(c):
        raise ValueError('KERNEL_SCALE_NONFINITE')
    return (b @ b.T / c if c > 0 else np.zeros((len(b), len(b)))), c


def cross_kernel(b_test, b_train, c):
    test = finite_array(b_test, 2, 'test features')
    train = finite_array(b_train, 2, 'training features')
    if test.shape[1] != train.shape[1] or not np.isfinite(c) or c < 0:
        raise ValueError('CROSS_KERNEL_CONTRACT_INVALID')
    return test @ train.T / c if c > 0 else np.zeros((len(test), len(train)))


def center_cross_kernel(k_star_t, k_tt):
    cross = finite_array(k_star_t, 2, 'cross kernel')
    train = finite_array(k_tt, 2, 'training kernel')
    if not len(train) or train.shape != (len(train), len(train)) or cross.shape[1] != len(train):
        raise ValueError('KERNEL_SHAPE_INVALID')
    return cross - cross.mean(axis=1, keepdims=True) - train.mean(axis=0) + train.mean()


def _squared_distances(left, right, inner_product=None):
    """Gram distances, with bounded direct repair of cancellation-sensitive pairs.

    A shared translation improves conditioning when no raw Gram is supplied.
    Caller-supplied inner products must correspond to the original feature rows.
    Close pairs use direct subtraction; clipping alone could turn duplicate rows
    into positive distances and change the training bandwidth.
    """
    if inner_product is None:
        origin = right[0]
        a, b = left-origin, right-origin
        gram = a @ b.T
    else:
        a, b = left, right
        gram = finite_array(inner_product, 2, 'raw inner product')
        if gram.shape != (len(left), len(right)):
            raise ValueError('DISTANCE_INNER_PRODUCT_SHAPE_INVALID')
    norms_left = np.einsum('ij,ij->i', a, a)
    norms_right = np.einsum('ij,ij->i', b, b)
    distances = norms_left[:, None] + norms_right - 2*gram
    bound = (8*np.finfo(float).eps*max(1, left.shape[1])
             * (norms_left[:, None]+norms_right+2*np.abs(gram)))
    if (distances < -bound).any():
        raise ValueError('DISTANCE_INNER_PRODUCT_INCONSISTENT')
    uncertain = (np.abs(distances) <= bound) & (bound > 0)
    if left is right:
        np.fill_diagonal(distances, 0)
        np.fill_diagonal(uncertain, False)
    rows, columns = np.where(uncertain)
    chunk = max(1, 1048576 // max(1, left.shape[1]))
    for offset in range(0, len(rows), chunk):
        selected_rows = rows[offset:offset+chunk]
        selected_columns = columns[offset:offset+chunk]
        difference = left[selected_rows] - right[selected_columns]
        distances[selected_rows, selected_columns] = np.einsum('ij,ij->i', difference, difference)
    if not np.isfinite(distances).all():
        raise ValueError('KERNEL_DISTANCE_NONFINITE')
    return distances


@dataclass(frozen=True)
class GaussianKernel:
    train: np.ndarray
    bandwidth: float
    raw_train: np.ndarray
    normalized_train: np.ndarray
    scale: float

    @classmethod
    def fit(cls, features, *, raw_gram=None):
        train = finite_array(features, 2, 'training features').copy()
        if not len(train):
            raise ValueError('EMPTY_TRAINING_PARTITION')
        distances = _squared_distances(train, train, raw_gram)
        positive = distances[np.triu_indices(len(train), k=1)]
        positive = positive[positive > 0]
        if not len(positive):
            zeros = np.zeros((len(train), len(train)))
            return cls(train, 0., zeros, zeros, 0.)
        bandwidth = float(np.median(positive))
        raw_train = np.exp(-distances/(2*bandwidth))
        centered = center_cross_kernel(raw_train, raw_train)
        scale = float(np.trace(centered)/len(train))
        if scale <= 0 or not np.isfinite(scale):
            raise ValueError('RBF_NORMALIZATION_FAILED')
        return cls(train, bandwidth, raw_train, centered/scale, scale)

    def transform(self, features, *, raw_cross=None):
        test = finite_array(features, 2, 'test features')
        if test.shape[1] != self.train.shape[1]:
            raise ValueError('FEATURE_SHAPE_INVALID')
        if self.bandwidth == 0:
            return np.zeros((len(test), len(self.train)))
        distances = _squared_distances(test, self.train, raw_cross)
        raw = np.exp(-distances/(2*self.bandwidth))
        return center_cross_kernel(raw, self.raw_train)/self.scale


def rbf_kernel(b_train, b_test=None, *, raw_train=None, raw_cross=None):
    fitted = GaussianKernel.fit(b_train, raw_gram=raw_train)
    result = (fitted.normalized_train if b_test is None else
              fitted.transform(b_test, raw_cross=raw_cross))
    return result, fitted.bandwidth


def diagonal_annotation_kernel(x_train, v):
    x = finite_array(x_train, 2, 'X')
    v = finite_array(v, 1, 'annotation diagonal')
    if len(v) != x.shape[1] or (v < 0).any():
        raise ValueError('ANNOTATION_DIAGONAL_INVALID')
    return trace_normalize(x * np.sqrt(v))[0]


def haplotype_kernel(counts_train, counts_test):
    """Aligned category columns must be constructed from training haplotypes only.

    NaN denotes an invalid window/category and is filled with its training mean.
    Unknown test haplotypes add no columns. This function does not discover categories.
    """
    train = np.asarray(counts_train, dtype=np.float64)
    test = np.asarray(counts_test, dtype=np.float64)
    if train.ndim != 2 or test.ndim != 2 or not len(train) or train.shape[1] != test.shape[1]:
        raise ValueError('HAPLOTYPE_COUNTS_SHAPE_INVALID')
    for values in (train, test):
        if not np.isin(values[~np.isnan(values)], [0, 1, 2]).all():
            raise ValueError('HAPLOTYPE_COUNTS_INVALID')
    observed = np.isfinite(train)
    means = np.divide(np.nansum(train, axis=0), observed.sum(axis=0),
                      out=np.zeros(train.shape[1]), where=observed.sum(axis=0) > 0)
    retained = np.nansum(train, axis=0) > 0
    centered_train = (np.where(observed, train, means) - means)[:, retained]
    centered_test = (np.where(np.isnan(test), means, test) - means)[:, retained]
    kernel, c = trace_normalize(centered_train)
    return kernel, cross_kernel(centered_test, centered_train, c)


def local_rbf_kernels(x_windowed, test_windowed=None):
    if not x_windowed or (test_windowed is not None and len(test_windowed) != len(x_windowed)):
        raise ValueError('WINDOW_FEATURES_INVALID')
    train_kernels, cross_kernels, removed = [], [], []
    shape = None
    for index, window in enumerate(x_windowed):
        fitted = GaussianKernel.fit(window)
        kernel, bandwidth = fitted.normalized_train, fitted.bandwidth
        if shape is None:
            shape = kernel.shape
        if kernel.shape != shape:
            raise ValueError('WINDOW_SAMPLE_ORDER_MISMATCH')
        cross = None if test_windowed is None else fitted.transform(test_windowed[index])
        if bandwidth == 0:
            removed.append(index)
        else:
            train_kernels.append(kernel)
            if cross is not None:
                cross_kernels.append(cross)
    train = np.mean(train_kernels, axis=0) if train_kernels else np.zeros(shape)
    if test_windowed is None:
        return train, removed
    cross_shapes = [(len(window), shape[0]) for window in test_windowed]
    if len(set(cross_shapes)) != 1:
        raise ValueError('WINDOW_TEST_SAMPLE_ORDER_MISMATCH')
    cross = np.mean(cross_kernels, axis=0) if cross_kernels else np.zeros(cross_shapes[0])
    return train, cross, removed
