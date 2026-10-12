# fprior：功能先验多核基因组预测

研究植物序列模型提供的功能先验，能否在强加性参照和匹配对照之外带来稳定、特异的预测增益。当前使用官方 SoySNP50K 数据，以大豆 `protein` 为主性状；教师为参数固定的 PlantCAD2，功能面板包含35个基因组窗口。

`fprior` 是独立研究项目，使用独立仓库、数据处理与运行目录，仅复用既有 Conda 环境。真实性状结果用于方法开发，独立验证材料的表型继续隔离。

## 使用

在项目根目录执行：

```bash
conda activate atpheno
export PYTHONPATH="$PWD/src:$PWD/.deps:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m fprior validate --config configs/funcprior.toml
python -m fprior diagnose
python -m fprior profile
python -m fprior evaluate
```

`validate` 仅检查必填字段与文件存在性，不代表输入已验证。`diagnose` 不读取表型；`profile` 测量单外折完整搜索与重拟合成本，响应类型由配置指定；`evaluate` 执行真实protein十折开发评价，内层四折调参。主参照为内层从M0／EN选出的 `AdditiveSelection`，同时报告相对M0及全部匹配对照的结果。

`evaluate --resume runs/<运行目录>` 只续算缺失外折；`evaluate --reuse-profile runs/<成本测量目录>` 可复用通过身份、源码、折内搜索、收敛与预测哈希核验的真实protein单折。两项不能同时使用，核验不符时拒绝复用。

所有参数维护于 [configs/funcprior.toml](configs/funcprior.toml)。运行前核对当前版本、预算、响应类型与设备设置；主预算为512，128／2048为敏感性预算。输入身份、开发ID范围、组约束划分、训练侧预处理及全部模型的数值收敛要求必须满足。

依赖由 [pyproject.toml](pyproject.toml) 管理，普通CPU依赖可用 `python -m pip install -e '.[dev,data,baselines]'` 安装。`.deps/` 为本地依赖目录，不入Git。教师CUDA扩展须与PyTorch、CUDA及GPU架构匹配，迁移时依据本地 `runs/environment_build/` 中的版本与构建记录重新构建。

本机为RTX 3090（24 GB，sm_86），驱动595.91.07；`atpheno` 使用PyTorch 2.14.1+cu130。教师扩展使用CUDA 13.0编译器，与PyTorch运行时一致；系统另有CUDA 13.2，不替换既有PyTorch或驱动。CUDA承担双精度大矩阵乘法与EN部分计算，核特征分解使用CPU；实际设备与线程记录随运行保存。

## 文件与数据

| 路径 | 用途 |
| --- | --- |
| `src/fprior/` | 功能特征、模型求解、输入隔离与评价 |
| `preprocessing/` | 数据处理代码；使用方式见[说明](preprocessing/README.md) |
| `data/soybase/raw/` | 官方输入、参考序列与教师文件 |
| `preprocessing/processed/` | 基因型、固定划分、开发标签与教师输出 |
| `configs/funcprior.toml` | 唯一实验配置 |
| `runs/` | 每次运行的配置、输入摘要、收敛记录、预测与指标 |
| `tests/` | 数学与实现的针对性测试 |

数据、模型、本地依赖及运行产物不随GitHub仓库发布，克隆代码不等于取得完整实验输入。官方VCF是当前分型来源；既有论文配套材料及历史结果仅作来源保留，不作为本轮产物。

## 研究文档

- [任务书](docs/任务书.md)：问题、数据与约束、工作、交付及完成条件。
- [研究方案](docs/研究方案.md)：模型定义、科学问题与解释范围。
- [验收规范](docs/验收规范.md)：输入、数学、对照和评价要求。
- [验收记录](docs/验收记录.md)：实际结果、来源证据与未完成事项；所引 `runs/` 文件仅在本地保存。
- [讲义](docs/讲义.md)：方法讲解。
- [AGENTS.md](AGENTS.md)：项目开发约定。
