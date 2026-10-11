"""独立基线对照(规范 4.2)。

EN:    与主模型同一 X 的 sklearn ElasticNet;l1_ratio in {0.1,0.5,0.9,1},
       alpha = alpha_max * {1,0.1,0.01,0.001,0.0001},alpha_max 必须每个内层训练集
       重新计算(||X^T y_c||_inf / (n*l1_ratio),为 0 则截距);max_iter=100000,
       tol=1e-6,未收敛记失败不得静默淘汰。
FU:    U-only Ridge,与 MU 同一目录/窗口/缺失/中心化规则,相同 rho 网格。
FU-GBRT: 同一中心化 U 上 sklearn GradientBoostingRegressor(squared_error;
       max_depth {1,2,3},learning_rate {.03,.1},n_estimators {100,300},
       min_samples_leaf=10,subsample=1,固定 seed);称 Epi-PRS 路线适配,
       不称原论文复现。

当前为框架占位,未实现。
"""

from __future__ import annotations

import numpy as np


def elasticnet_grid(x_train: np.ndarray, y_train: np.ndarray) -> list[dict]:
    """返回 EN 候选配置网格;alpha_max 按当前(内层)训练集计算。"""
    raise NotImplementedError


def gbrt_grid(u_train: np.ndarray, y_train: np.ndarray, seed: int) -> list[dict]:
    """返回 FU-GBRT 12 组固定配置;不择优扩展搜索空间。"""
    raise NotImplementedError
