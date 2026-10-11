from pathlib import Path
import numpy as np
import pytest
from fprior.io import load_config
from fprior.baselines import elasticnet_grid, gbrt_grid, fit_baseline


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


def test_gbrt_fixed_grid_and_frozen_predict():
    s = settings()['gbrt']; rng = np.random.default_rng(11)
    x = rng.normal(size=(40, 3)); y = x[:, 0]**2
    grid = gbrt_grid(s); assert len(grid) == 12
    fitted = fit_baseline('FU-GBRT', x, y, grid[0], s, seed=7)
    assert np.isfinite(fitted.predict(x[:4])).all()
    assert fitted.predict(x[:4])[0] == fitted.predict(x[:1])[0]
