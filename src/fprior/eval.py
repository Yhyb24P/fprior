"""评价指标与决策(规范 6)。

主指标:原量纲 pooled MSE 与配对平方误差差;同时报告相对 MSE 改变、MAE、PCC
(响应或预测常数记 NA)、校准斜率、以各外层训练均值为参照的 pooled Q2。
不确定性:以 group 为抽样单位对配对外层误差重采样(bootstrap 仅作开发描述性区间,
共享训练数据产生的依赖须注明;不实施"独立折 t 检验")。
继续条件:相对 MSE 降低 >= 2%、不少于 4/5 外层方向一致、完整覆盖分析未反转,
且 M2 分别胜过 M1/MU/MW/MWG 及普通局部/单倍型对照;达不到记
INCONCLUSIVE_UNDER_BUDGET 或 COMPARISON_INCOMPLETE,不得记算法成功。

当前为框架占位,未实现。
"""

from __future__ import annotations

import numpy as np


def pooled_mse(y: np.ndarray, yhat: np.ndarray) -> float:
    """sum((y - yhat)^2)/N。"""
    raise NotImplementedError


def q2(y: np.ndarray, yhat: np.ndarray, train_means: np.ndarray) -> float:
    """Q2 = 1 - SSE/SST,参照为各材料所属外层训练集均值。"""
    raise NotImplementedError


def paired_metrics(y: np.ndarray, predictions: dict[str, np.ndarray]) -> dict:
    """对全部对照输出配对 MSE/相对增益/MAE/PCC(常数时 NA)/校准斜率。"""
    raise NotImplementedError


def group_bootstrap_interval(
    y: np.ndarray, yhat_a: np.ndarray, yhat_b: np.ndarray, groups: np.ndarray, replicates: int, seed: int
) -> dict:
    """以 group 为单位的配对误差差 bootstrap 2.5%/97.5% 分位区间(描述性)。"""
    raise NotImplementedError


def continue_decision(relative_gain: float, outer_signs: list[int], min_fraction: float = 4 / 5) -> str:
    """开发阶段去留判断;不构成统计显著性检验。"""
    raise NotImplementedError
