"""Configuration-defined EN and FU-GBRT candidates, fitted on training rows only."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
import warnings
import numpy as np
from .io import finite_array


def elasticnet_grid(settings):
    return [dict(l1_ratio=float(l1), alpha_ratio=float(ratio))
            for l1, ratio in product(settings['l1_ratio'], settings['alpha_max_ratios'])]


def gbrt_grid(settings):
    return [dict(max_depth=int(depth), learning_rate=float(rate), n_estimators=int(count))
            for depth, rate, count in product(settings['max_depth'], settings['learning_rate'],
                                               settings['n_estimators'])]


@dataclass(frozen=True)
class BaselineFit:
    estimator: object
    intercept: float
    n_features: int
    diagnostics: dict

    def predict(self, values):
        x = finite_array(values, 2, 'baseline test features')
        if x.shape[1] != self.n_features:
            raise ValueError('BASELINE_FEATURE_ORDER_INVALID')
        prediction = (np.full(len(x), self.intercept) if self.estimator is None else
                      self.estimator.predict(x))
        return finite_array(prediction, 1, 'baseline predictions')


def fit_baseline(family, x_train, y_train, parameters, settings, *, seed):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import ElasticNet
    from sklearn.ensemble import GradientBoostingRegressor

    x = finite_array(x_train, 2, 'baseline training features')
    y = finite_array(y_train, 1, 'baseline training labels')
    if not len(y) or len(y) != len(x) or family not in ['EN', 'FU-GBRT']:
        raise ValueError('BASELINE_TRAINING_CONTRACT_INVALID')
    centered = y-y.mean()
    diagnostics = dict(status='PASSED', family=family, parameters=dict(parameters))
    if family == 'EN':
        l1, ratio = parameters['l1_ratio'], parameters['alpha_ratio']
        if not 0 < l1 <= 1 or not 0 < ratio <= 1:
            raise ValueError('ELASTICNET_PARAMETERS_INVALID')
        alpha_max = float(np.max(np.abs(x.T@centered), initial=0)/(len(y)*l1))
        diagnostics.update(alpha_max=alpha_max, alpha=alpha_max*ratio)
        if alpha_max == 0:
            diagnostics.update(intercept_only=True, dual_gap=0., iterations=0)
            return BaselineFit(None, float(y.mean()), x.shape[1], diagnostics)
        estimator = ElasticNet(alpha=alpha_max*ratio, l1_ratio=l1,
                               max_iter=settings['max_iter'], tol=settings['tol'],
                               fit_intercept=True, selection='cyclic')
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always', ConvergenceWarning)
            estimator.fit(x, y)
        gap = float(estimator.dual_gap_)
        diagnostics.update(dual_gap=gap, iterations=int(estimator.n_iter_))
        limit = settings['tol']*float(centered@centered)/len(y)
        if (any(issubclass(w.category, ConvergenceWarning) for w in captured)
                or not np.isfinite(gap) or gap > limit*(1+1e-10)):
            raise ValueError(f'BASELINE_NOT_CONVERGED:family=EN,parameters={parameters},'
                             f'alpha={alpha_max*ratio},iterations={estimator.n_iter_},'
                             f'dual_gap={gap},limit={limit}')
    else:
        if not x.shape[1] or np.all(centered == 0):
            diagnostics['intercept_only'] = True
            return BaselineFit(None, float(y.mean()), x.shape[1], diagnostics)
        estimator = GradientBoostingRegressor(**parameters, loss=settings['loss'],
            min_samples_leaf=settings['min_samples_leaf'], subsample=settings['subsample'],
            random_state=seed)
        estimator.fit(x, y)
        diagnostics['iterations'] = int(estimator.n_estimators_)
    return BaselineFit(estimator, float(y.mean()), x.shape[1], diagnostics)
