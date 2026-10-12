"""Configuration-defined EN and FU-GBRT candidates, fitted on training rows only."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from copy import deepcopy
import warnings
import time
import numpy as np
from .io import finite_array


class DenseArithmetic:
    """Training-centered float64 products, resident on the selected device."""
    def __init__(self, values, device='cpu'):
        self.values = values
        self.tensor = None
        if device == 'cuda':
            import torch
            if not torch.cuda.is_available():
                raise ValueError('CUDA_UNAVAILABLE')
            self.tensor = torch.as_tensor(values, dtype=torch.float64, device='cuda')
        elif device != 'cpu':
            raise ValueError('COMPUTE_DEVICE_INVALID')

    def dot(self, value, *, transpose=False):
        if self.tensor is None:
            return (self.values.T if transpose else self.values)@value
        import torch
        matrix = self.tensor.T if transpose else self.tensor
        return (matrix@torch.as_tensor(value, dtype=torch.float64, device='cuda')).cpu().numpy()

    def gram(self, columns=None):
        if self.tensor is None:
            x = self.values if columns is None else self.values[:, columns]
            return x@x.T
        x = self.tensor if columns is None else self.tensor[:, columns]
        return (x@x.T).cpu().numpy()


def elasticnet_grid(settings):
    return [dict(l1_ratio=float(l1), alpha_ratio=float(ratio))
            for l1, ratio in product(settings['l1_ratio'], settings['alpha_max_ratios'])]


def gbrt_grid(settings):
    return [dict(max_depth=int(depth), learning_rate=float(rate), n_estimators=int(count))
            for depth, rate, count in product(settings['max_depth'], settings['learning_rate'],
                                               settings['n_estimators'])]


def _elasticnet_dual_newton(x, y, alpha, l1, initial, max_iter, tolerance, arithmetic=None):
    """Solve the identical EN objective in sample space, with a dual certificate.

    The primal objective multiplied by n is
    .5||y-X beta||² + a||beta||_1 + .5 b||beta||².
    Its unconstrained dual minimization has gradient u-y+X beta,
    beta=soft_threshold(X.T u,a)/b. This requires b>0.
    """
    from scipy.linalg import cho_factor, cho_solve
    n = len(y); a = n*alpha*l1; b = n*alpha*(1-l1)
    arithmetic = arithmetic or DenseArithmetic(x-x.mean(axis=0))
    xc = arithmetic.values; yc = y-y.mean()
    u = yc-arithmetic.dot(initial)
    full_gram = None
    limit = tolerance*float(yc@yc)/n
    def state(value):
        projection = arithmetic.dot(value, transpose=True)
        beta = np.sign(projection)*np.maximum(np.abs(projection)-a, 0)/b
        objective = .5*(value@value)-yc@value+.5*b*(beta@beta)
        return beta, float(objective)
    for iteration in range(max_iter+1):
        beta, objective = state(u)
        residual = yc-arithmetic.dot(beta)
        primal = .5*(residual@residual)+a*np.abs(beta).sum()+.5*b*(beta@beta)
        gap = float((primal+objective)/n)
        roundoff = 64*np.finfo(float).eps*(1+abs(primal)+abs(objective))/n
        if np.isfinite(gap) and -roundoff <= gap <= limit*(1+1e-10):
            return beta, max(0., gap), iteration
        if iteration == max_iter:
            break
        active = beta != 0
        if active.sum() > len(active)//2:
            if full_gram is None:
                full_gram = arithmetic.gram()
            gram = full_gram-arithmetic.gram(~active)
        else:
            gram = arithmetic.gram(active)
        hessian = (gram+gram.T)/(2*b)+np.eye(n)
        gradient = u-residual
        direction = -cho_solve(cho_factor(hessian, lower=True), gradient)
        slope = float(gradient@direction)
        step = 1.
        while step >= 2.**-40:
            trial = u+step*direction
            _, candidate = state(trial)
            if candidate <= objective+1e-4*step*slope:
                u = trial
                break
            step *= .5
        else:
            break
    raise ValueError(f'BASELINE_NOT_CONVERGED:solver=dual_newton,dual_gap={gap},limit={limit}')


def _lasso_working_set(x, y, alpha, initial, max_iter, tolerance, arithmetic=None):
    """Coordinate subproblems, certified against every original SNP column."""
    from sklearn.linear_model import Lasso
    from sklearn.exceptions import ConvergenceWarning
    n, p = x.shape
    arithmetic = arithmetic or DenseArithmetic(x-x.mean(axis=0))
    xc = arithmetic.values; yc = y-y.mean()
    beta = initial.copy(); active = set(np.flatnonzero(beta).tolist())
    iterations = 0; limit = tolerance*float(yc@yc)/n
    while True:
        residual = yc-arithmetic.dot(beta)
        correlations = arithmetic.dot(residual, transpose=True)
        norm = float(np.max(np.abs(correlations), initial=0))
        dual = residual*min(1., n*alpha/norm) if norm else residual
        primal = .5*(residual@residual)+n*alpha*np.abs(beta).sum()
        dual_objective = yc@dual-.5*(dual@dual)
        gap = float((primal-dual_objective)/n)
        roundoff = 64*np.finfo(float).eps*(1+abs(primal)+abs(dual_objective))/n
        if np.isfinite(gap) and -roundoff <= gap <= limit*(1+1e-10):
            return beta, max(0., gap), iterations
        if iterations >= max_iter:
            raise ValueError(f'BASELINE_NOT_CONVERGED:solver=lasso_working_set,dual_gap={gap},limit={limit},iterations={iterations},support={np.count_nonzero(beta)},working_columns={len(active)}')
        support = np.flatnonzero(beta)
        if len(support):
            xs = xc[:, support]; old = beta[support]
            u, singular, vt = np.linalg.svd(xs, full_matrices=len(support) > n)
            rank_tolerance = 64*np.finfo(float).eps*max(xs.shape)*singular[0]
            dependent = len(support) > n or singular[-1] <= rank_tolerance
            if dependent:
                # Remove one redundant coefficient along a null direction,
                # preserving fitted values and never increasing the L1 norm.
                direction = vt[-1].copy()
                if np.sign(old)@direction > 0:
                    direction *= -1
                changes = old*direction < 0
                step = float(np.min(-old[changes]/direction[changes]))
            else:
                # Solve the fixed-sign stationarity equation through the SVD,
                # avoiding the squared condition number of a Gram solve.
                polished = vt.T@((u.T@yc)/singular-
                                  n*alpha*(vt@np.sign(old))/singular**2)
                direction = polished-old
                changes = old*polished <= 0
                step = min(1., float(np.min(-old[changes]/direction[changes]))) if changes.any() else 1.
            candidate = beta.copy()
            candidate[support] = old+step*direction
            if changes.any():
                zero = support[changes][np.isclose(-old[changes]/direction[changes], step,
                                                   rtol=1e-12, atol=0)]
                candidate[zero] = 0
            error = yc-arithmetic.dot(candidate)
            candidate_primal = .5*(error@error)+n*alpha*np.abs(candidate).sum()
            iterations += 1
            improves = candidate_primal < primal
            preserves = dependent and candidate_primal <= primal+64*np.finfo(float).eps*(1+abs(primal))
            if improves or preserves:
                beta = candidate
                continue
            if iterations >= max_iter:
                continue
        violations = np.flatnonzero(np.abs(correlations) > n*alpha)
        # Expand deterministically; keep all previously admitted variables so
        # a subproblem cannot silently remove a difficult part of the objective.
        unseen = np.asarray([j for j in violations if int(j) not in active], dtype=int)
        order = np.argsort(-np.abs(correlations[unseen]), kind='stable')
        active.update(unseen[order[:n]].tolist())
        if not active:
            raise ValueError('BASELINE_NOT_CONVERGED:empty_lasso_working_set')
        columns = np.asarray(sorted(active))
        estimator = Lasso(alpha=alpha, fit_intercept=False, warm_start=True,
                          max_iter=min(1000, max_iter-iterations), tol=tolerance*.1,
                          selection='cyclic')
        estimator.coef_ = beta[columns].copy()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always', ConvergenceWarning)
            estimator.fit(xc[:, columns], yc)
        beta[columns] = estimator.coef_
        iterations += int(estimator.n_iter_)


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


def fit_elasticnet_path(x_train, y_train, parameters, settings, *, seed):
    """Descending alpha within each L1 ratio; immutable fits in caller order."""
    fits = [None]*len(parameters)
    arithmetic = DenseArithmetic(x_train-x_train.mean(axis=0), settings.get('compute_device', 'cpu'))
    for l1 in dict.fromkeys(p['l1_ratio'] for p in parameters):
        indices = sorted((i for i, p in enumerate(parameters) if p['l1_ratio'] == l1),
                         key=lambda i: -parameters[i]['alpha_ratio'])
        estimator = None
        for index in indices:
            start = time.perf_counter()
            fitted = fit_baseline('EN', x_train, y_train, parameters[index], settings,
                                  seed=seed, _warm_estimator=estimator, _arithmetic=arithmetic)
            estimator = fitted.estimator
            fits[index] = BaselineFit(deepcopy(estimator), fitted.intercept,
                                      fitted.n_features, dict(fitted.diagnostics,
                                      seconds=time.perf_counter()-start))
    return fits


def fit_baseline(family, x_train, y_train, parameters, settings, *, seed,
                 _warm_estimator=None, _arithmetic=None):
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
        arithmetic = _arithmetic or DenseArithmetic(x-x.mean(axis=0), settings.get('compute_device', 'cpu'))
        l1, ratio = parameters['l1_ratio'], parameters['alpha_ratio']
        if not 0 < l1 <= 1 or not 0 < ratio <= 1:
            raise ValueError('ELASTICNET_PARAMETERS_INVALID')
        alpha_max = float(np.max(np.abs(arithmetic.dot(centered, transpose=True)), initial=0)/(len(y)*l1))
        diagnostics.update(alpha_max=alpha_max, alpha=alpha_max*ratio)
        if alpha_max == 0:
            diagnostics.update(intercept_only=True, dual_gap=0., iterations=0)
            return BaselineFit(None, float(y.mean()), x.shape[1], diagnostics)
        dual_eligible = l1 < 1 and x.shape[1] > len(y)
        coordinate_steps = min(settings['max_iter'], settings.get('coordinate_warm_steps', settings['max_iter'])) if dual_eligible else settings['max_iter']
        estimator = _warm_estimator or ElasticNet(alpha=alpha_max*ratio, l1_ratio=l1,
                               max_iter=coordinate_steps, tol=settings['tol'],
                               fit_intercept=True, selection='cyclic', warm_start=True)
        estimator.set_params(alpha=alpha_max*ratio, max_iter=coordinate_steps)
        diagnostics['warm_start'] = _warm_estimator is not None
        if (l1 == 1 and x.shape[1] > len(y)
                and settings.get('lasso_optimizer') == 'working_set_with_full_dual_gap_check'):
            initial = np.asarray(estimator.coef_) if hasattr(estimator, 'coef_') else np.zeros(x.shape[1])
            try:
                beta, gap, iterations = _lasso_working_set(x, y, alpha_max*ratio, initial,
                                                          settings['max_iter'], settings['tol'], arithmetic)
            except ValueError as exc:
                raise ValueError(f'{exc},parameters={parameters}') from exc
            estimator.coef_ = beta
            estimator.intercept_ = float(y.mean()-x.mean(axis=0)@beta)
            estimator.n_features_in_ = x.shape[1]
            estimator.dual_gap_ = gap
            estimator.n_iter_ = iterations
            diagnostics.update(solver='lasso_working_set', dual_gap=gap, iterations=iterations)
            return BaselineFit(estimator, float(y.mean()), x.shape[1], diagnostics)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always', ConvergenceWarning)
            estimator.fit(x, y)
        gap = float(estimator.dual_gap_)
        limit = settings['tol']*float(centered@centered)/len(y)
        failed = (any(issubclass(w.category, ConvergenceWarning) for w in captured)
                  or not np.isfinite(gap) or gap > limit*(1+1e-10))
        diagnostics['coordinate_iterations'] = int(estimator.n_iter_)
        if failed and dual_eligible and estimator.n_iter_ < settings['max_iter']:
            beta, gap, iterations = _elasticnet_dual_newton(x, y, alpha_max*ratio, l1,
                estimator.coef_, settings['max_iter']-int(estimator.n_iter_), settings['tol'], arithmetic)
            estimator.coef_ = beta
            estimator.intercept_ = float(y.mean()-x.mean(axis=0)@beta)
            estimator.dual_gap_ = gap
            estimator.n_iter_ += iterations
            diagnostics.update(solver='coordinate_warm_path_with_dual_newton', dual_iterations=iterations)
            failed = False
        diagnostics.update(dual_gap=gap, iterations=int(estimator.n_iter_))
        if failed:
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
