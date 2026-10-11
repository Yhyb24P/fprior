"""功能先验多核岭回归独立项目(fprior)。

实现目标依据 docs/研究方案.md 与 docs/验收规范.md:
外部植物功能模型诱导的有符号效应协方差(A/U/R 分解)能否改善小样本基因组预测。

模块职责:
- io:        输入合同校验与封存 ID 排除(规范 2.1)
- features:  参考锚定分解 d/A/U/R、有效性掩码、列中心化(规范 3)
- kernels:   各对照核与迹归一化(规范 4.2)
- solver:    单纯形网格多核岭求解(规范 4.1)
- teacher:   教师协议与缓存键;真实 plantcad 加载尚未实现
- baselines: ElasticNet / U-only GBRT 对照
- eval:      pooled MSE / Q2 / PCC、组 bootstrap 描述区间(规范 6)

已实现输入预检、功能分解、核及岭求解；教师加载、实验基线与评价仍为占位。
数学单元验证不代表完整合成机制验收或真实训练就绪。
"""

__all__ = ["io", "features", "kernels", "solver", "teacher", "baselines", "eval"]
