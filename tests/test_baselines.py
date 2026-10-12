from pathlib import Path
import numpy as np
import pytest
from fprior.io import load_config
from fprior.baselines import elasticnet_grid, gbrt_grid, fit_baseline, fit_elasticnet_path


def settings():
    return load_config(Path(__file__).parents[1]/'configs/funcprior.toml')[0]['baselines']


def test_en_uses_current_training_alpha_max_and_refits_relative_parameter():
    s = settings()['elasticnet']; rng = np.random.default_rng(2)
    x = rng.normal(size=(40, 8)); x -= x.mean(axis=0)
    y = 3+x[:, 0]+rng.normal(size=40)*.1
    parameters = dict(l1_ratio=.5, alpha_ratio=.1)
    first = fit_baseline('EN', x, y, parameters, s, seed=7)
    second = fit_baseline('EN', x, y*2, parameters, s, seed=7)
    expected = np.max(np.abs(x.T@(y-y.mean())))/(40*.5)
    assert first.diagnostics['alpha_max'] == pytest.approx(expected)
    assert second.diagnostics['alpha'] == pytest.approx(2*first.diagnostics['alpha'])
    # Mixed L1/L2 EN is not homogeneous in the response; verify its KKT equation.
    for fitted, target in [(first, y), (second, y*2)]:
        beta = fitted.estimator.coef_
        gradient = x.T@(target-fitted.predict(x))/len(target)
        active = beta != 0
        alpha = fitted.diagnostics['alpha']
        assert np.allclose(gradient[active], alpha*(.5*np.sign(beta[active])+.5*beta[active]), atol=1e-6)
        assert (np.abs(gradient[~active]) <= alpha*.5+1e-6).all()
    assert len(elasticnet_grid(s)) == 20


def test_intercept_only_and_nonconvergence_are_explicit():
    s = settings()['elasticnet']; rng = np.random.default_rng(4)
    x = rng.normal(size=(30, 80)); y = rng.normal(size=30)
    p = dict(l1_ratio=.5, alpha_ratio=.0001)
    constant = fit_baseline('EN', x, np.full(30, 4.), p, s, seed=7)
    assert constant.diagnostics['intercept_only']
    assert constant.predict(x).tolist() == [4]*30
    with pytest.raises(ValueError, match='BASELINE_NOT_CONVERGED'):
        fit_baseline('EN', x, y, p, dict(s, max_iter=1), seed=7)
    with pytest.raises(ValueError, match='BASELINE_NOT_CONVERGED'):
        fit_baseline('EN', x, y, dict(p, l1_ratio=1.), dict(s, max_iter=1), seed=7)


def test_gbrt_fixed_grid_and_frozen_predict():
    s = settings()['gbrt']; rng = np.random.default_rng(11)
    x = rng.normal(size=(40, 3)); y = x[:, 0]**2
    grid = gbrt_grid(s); assert len(grid) == 12
    fitted = fit_baseline('FU-GBRT', x, y, grid[0], s, seed=7)
    assert np.isfinite(fitted.predict(x[:4])).all()
    assert fitted.predict(x[:4])[0] == fitted.predict(x[:1])[0]


def test_en_path_matches_independent_objective_and_preserves_each_fit():
    s = settings()['elasticnet']; rng = np.random.default_rng(51)
    base = rng.normal(size=(35, 12))
    x = np.column_stack([base, base+.01*rng.normal(size=base.shape)])
    y = base[:, 0]+rng.normal(size=35)*.2
    parameters = [dict(l1_ratio=l1, alpha_ratio=ratio)
                  for l1 in (.5, 1.) for ratio in (.01, 1., .1)]
    path = fit_elasticnet_path(x, y, parameters, s, seed=7)
    for p, fitted in zip(parameters, path):
        cold = fit_baseline('EN', x, y, p, s, seed=7)
        def objective(fit):
            beta = fit.estimator.coef_
            return (np.mean((y-fit.predict(x))**2)/2 + fit.diagnostics['alpha']*
                    (p['l1_ratio']*np.abs(beta).sum()+(1-p['l1_ratio'])*(beta@beta)/2))
        assert objective(fitted) == pytest.approx(objective(cold), abs=2e-6)
        np.testing.assert_allclose(fitted.predict(x), cold.predict(x), atol=1e-3)
    assert path[1].diagnostics['warm_start'] is False
    assert path[0].diagnostics['warm_start'] is True
    assert path[1].estimator is not path[0].estimator
    np.testing.assert_allclose(path[1].predict(x), y.mean(), atol=1e-12)


def test_high_dimensional_dual_solver_preserves_en_loss_and_certificate():
    s = settings()['elasticnet']; rng = np.random.default_rng(91)
    base = rng.normal(size=(32, 10))
    x = np.tile(base, (1, 8))+.03*rng.normal(size=(32, 80))
    y = base[:, 0]+rng.normal(size=32)*.3
    p = dict(l1_ratio=.1, alpha_ratio=.001)
    dual = fit_baseline('EN', x, y, p, dict(s, coordinate_warm_steps=1), seed=7)
    cold = fit_baseline('EN', x, y, p, dict(s, coordinate_warm_steps=s['max_iter']), seed=7)
    assert dual.diagnostics['solver'] == 'coordinate_warm_path_with_dual_newton'
    assert dual.diagnostics['dual_gap'] <= s['tol']*np.var(y)
    alpha = dual.diagnostics['alpha']
    def loss(fit):
        beta = fit.estimator.coef_
        return .5*np.mean((y-fit.predict(x))**2)+alpha*(.1*np.abs(beta).sum()+.45*(beta@beta))
    assert loss(dual) == pytest.approx(loss(cold), abs=1e-6)
    np.testing.assert_allclose(dual.predict(x), cold.predict(x), atol=2e-3)


def test_high_dimensional_lasso_path_has_original_dual_gap_certificate():
    from sklearn.linear_model import Lasso
    s = settings()['elasticnet']; rng = np.random.default_rng(64)
    x = rng.normal(size=(30, 75)); y = x[:, 0]+rng.normal(size=30)*.2
    p = dict(l1_ratio=1., alpha_ratio=.01)
    fitted = fit_baseline('EN', x, y, p, s, seed=7)
    cold = Lasso(alpha=fitted.diagnostics['alpha'], max_iter=s['max_iter'], tol=s['tol']).fit(x, y)
    assert fitted.diagnostics['solver'] == 'lasso_working_set'
    assert fitted.diagnostics['dual_gap'] <= s['tol']*np.var(y)
    np.testing.assert_allclose(fitted.predict(x), cold.predict(x), atol=2e-3)
    beta = fitted.estimator.coef_
    gradient = (x-x.mean(axis=0)).T@(y-fitted.predict(x))/len(y)
    active = beta != 0
    alpha = fitted.diagnostics['alpha']
    np.testing.assert_allclose(gradient[active], alpha*np.sign(beta[active]), atol=1e-6)
    assert (np.abs(gradient[~active]) <= alpha+1e-6).all()


def test_lasso_with_redundant_columns_preserves_original_optimum():
    from sklearn.linear_model import Lasso
    from fprior.baselines import _lasso_working_set
    rng = np.random.default_rng(99)
    base = rng.normal(size=(12, 6)); x = np.tile(base, (1, 4))
    y = 3+base[:, 0]+.4*base[:, 2]+rng.normal(size=12)*.1
    initial = np.tile([.25, .1, 0., 0., .2, 0.], 4)
    beta, gap, _ = _lasso_working_set(x, y, .01, initial, 100000, 1e-6)
    prediction = y.mean()+(x-x.mean(axis=0))@beta
    cold = Lasso(alpha=.01, max_iter=100000, tol=1e-7).fit(x, y)
    assert gap <= 1e-6*np.var(y)
    np.testing.assert_allclose(prediction, cold.predict(x), atol=2e-3)
    loss = .5*np.mean((y-prediction)**2)+.01*np.abs(beta).sum()
    expected = .5*np.mean((y-cold.predict(x))**2)+.01*np.abs(cold.coef_).sum()
    assert loss == pytest.approx(expected, abs=1e-6)
