# fprior：功能先验多核岭回归项目

研究问题：参数固定的植物序列功能模型诱导的有符号、参考背景依赖的效应协方差，能否改善基因组预测？当前基于官方 SoySNP50K Wm82.gnm2 发布数据重建队列，以 `protein` 为主性状；[任务书](docs/任务书.md)定义大豆适配、执行顺序和验收条件。[研究方案](docs/研究方案.md)说明数学定义与验证逻辑，实际结果见[验收记录](docs/验收记录.md)。

`fprior` 是从 `at_pheno` 衍生的独立研究项目，只复用既有 Conda 环境和通用仓库管理原则。数据处理、算法、配置与运行产物均由本项目独立维护，开发约定见 [AGENTS.md](AGENTS.md)。本目录建立独立 Git 仓库，主分支使用 `main`；GitHub 仓库为 [Yhyb24P/fprior](https://github.com/Yhyb24P/fprior)。目录在磁盘上仍嵌套于 `at_pheno`，独立仓库从本项目当前源码副本开始，不导入父仓库历史。父仓库既有历史和暂存内容保持原样。

## 目录与数据

| 路径 | 用途 | 管理方式 |
| --- | --- | --- |
| `data/soybase/` | SoyBase 数据表及既有模型 | 本地保留，整体不入 Git |
| `data/paper_soybase/` | 已拉取的论文配套源码、数据及产物 | 本地保留，作为来源材料 |
| `data/results_soybase/` | 既有结果文件 | 本地保留，不作为本轮结果 |
| `docs/` | 研究方案、验收规范、讲义和配图 | 保留，纳入版本管理 |
| `src/fprior/` | 数学核心、隔离输入、真实教师及开发评价 | 纳入版本管理 |
| `configs/funcprior.toml` | 唯一实验参数与输入预检配置 | 纳入版本管理 |
| `pyproject.toml` | Python 包定义 | 纳入版本管理 |

本版以官方 VCF 为唯一分型来源，旧 BiMSGP 矩阵及历史划分仅本地保留。数据范围及观测规则见[任务书 §二](docs/任务书.md)，来源统计与验收证据见[验收记录](docs/验收记录.md)。原始文件在 `data/soybase/raw/`；下载和完整性证据在 `runs/soybase_official_v2_acquisition/`，均不入 Git。

数据处理代码集中在 `preprocessing/`，产物在 `preprocessing/processed/soybase_official_v2/`，算法只读取通过验收的产物。GT 导出与训练侧转换见 [preprocessing/README.md](preprocessing/README.md)。新运行输出写入 `runs/` 下独立目录，不改写原始文件。

## 文档

- [docs/任务书.md](docs/任务书.md) — SoyBase 目标、数据约束、工作、交付和完成条件
- [docs/验收记录.md](docs/验收记录.md) — 来源统计、固定划分记录、实际验收结果及待完成工作
- [docs/研究方案.md](docs/研究方案.md) — 研究问题、模型选择、等价推导和验证逻辑
- [docs/验收规范.md](docs/验收规范.md) — 输入输出要求、求解与对照定义、检查构造和通过条件
- [docs/讲义.md](docs/讲义.md) — 科普讲解，数学定义与研究方案一致

## 代码

`io` 提供固定 ID、缓存来源与划分哈希核验，以及按授权 ID 偏移读取开发标签。`features` 和 `design` 构建可信单倍型、D/A/U/R、无教师的完整局部交互基和全部对照核；QC、填补、中心化、MAF 分层置乱及单倍型类别均在当前训练分区拟合。全维 RBF 复用原始加性内积，对接近零的距离作直接差分校正。

`baselines` 实现 ElasticNet 和 FU-GBRT 的配置网格、折内拟合及收敛检查；`experiment` 按固定四内折 pooled MSE 选参、外层重拟合和保存预测后不再选优，同权重分解跨模型复用。`eval` 实现 MSE、MAE、PCC、校准、训练均值基准 Q² 和整组 bootstrap。`runner` 提供主预算诊断、使用合成响应的全模型成本测量及受前置验收约束的开发评价。实现与实验分别验收，结果见验收记录。

`teacher` 的本地文件校验、固定 PlantCAD2/LoRA 和分类头恢复、原始 logits、版本／序列缓存沿用已验收实现；模型微批次固定为 1。同一版本的已保存输出可复用，教师保持不变；重复检查仅在相关输入或实现变化时进行。

配置：[configs/funcprior.toml](configs/funcprior.toml)。路径相对项目根目录解析；数值容差也从该配置读取。元数据预检只检查要求字段与文件存在性，不读取基因型或表型内容：

```bash
python -m fprior validate --config configs/funcprior.toml
```

目标未指定时返回 `TARGET_UNSPECIFIED`；必填输入缺失时返回 `MISSING_REQUIRED_INPUTS`，退出码为 2。文件齐全也只返回 `INPUTS_PRESENT_UNVERIFIED`，不能据此声明 `DATA_READY`。改变数据、教师、窗口、参考背景或搜索空间时须先满足输入要求并更新实验版本。

开发命令使用同一份配置：

```bash
python -m fprior diagnose   # 主预算十外折／四内折的无表型核诊断
python -m fprior profile    # 固定一外折完整比较成本，使用合成响应
python -m fprior evaluate   # 前置验收缺失时停止，不读取训练标签
```

`execution.pretraining_acceptance` 必须指向本实验版本、同一固定划分的机器可读验收记录，须核验开发逐Study定向恢复、原720个实际实例、附加1,200个稀疏功效实例及实际结果文件，技术与功效状态分别记录。新的证据要求见[验收规范 §7](docs/验收规范.md)，现有验收接口尚需修改，不能据配置新增字段宣称实现完成。记录缺失时，`evaluate` 返回 `PRETRAINING_ACCEPTANCE_MISSING`。参数、验收字段与解释边界见[任务书](docs/任务书.md)，输出只在独立 `runs/` 目录。

合成数学检查：

```bash
python -m pytest -q tests/test_core.py
```

核心针对性测试涵盖训练侧 QC／缺失填补、相位可信性、有符号窗口效应、分解与中心化恒等式、核训练归一化、测试批次隔离、原始—对偶预测一致性和失败停止。结果写在 `runs/soybase_funcprior_v1_checks/`；这不是验收规范的 720 组合成机制实验，也不包含真实 protein 预测。

## 环境

复用既有 Miniconda 环境 `atpheno`，不另建解释器环境。依赖统一由 [pyproject.toml](pyproject.toml) 管理；本机新增依赖安装在 `.deps/`，复用 Conda 环境已有的 NumPy、PyTorch 和 pytest。在本目录执行：

```bash
conda activate atpheno
export PYTHONPATH="$PWD/src:$PWD/.deps${PYTHONPATH:+:$PYTHONPATH}"
```

`.deps/` 不入 Git，也不是独立 Python 环境。下载复用已有代理环境变量，不在仓库保存代理地址或凭据。常规 CPU 依赖可用 `python -m pip install -e '.[dev,data,baselines]'` 安装；教师扩展需要使用兼容源码构建，不能直接套用其他 Torch 版本的 wheel。

既有硬件记录：RTX 3090（24 GB，sm_86）、驱动 595.91.07；`atpheno` 的 PyTorch 2.14.1+cu130 使用 CUDA 13.0 运行时。系统另有 `/usr/local/cuda-13.2`，nvcc 13.2 可用；教师扩展构建使用 `.deps/nvidia/cu13` 中的 nvcc 13.0，保持编译器与 PyTorch 运行时匹配，不改动系统 CUDA 或 PyTorch。

教师的 Transformers／PEFT 接口版本参考 [PlantCAD 官方依赖](https://github.com/plantcad/plantcad/blob/main/requirements.txt)。Mamba 选择覆盖 PlantCAD Mamba2 接口的 2.x 版本，避免引入当前未使用的 Mamba3 依赖。扩展源码只调整为 C++20 和 sm_86 编译，并补齐本地 CUDA 编译目录的 `libcudart.so` 链接名；模型和内核数学代码保持原样。源码版本由 `teacher` extra 锁定，编译组件由 `cuda-build` extra 锁定。

编译 wheel、源码 commit、补丁、构建日志和 GPU 数值测试结果存放在本项目 `runs/environment_build/`，不入 Git。迁移机器时须按该 manifest 核对 Python、Torch、CUDA、C++ ABI 与 GPU 架构，重新构建；本机源码编译使用 `--no-build-isolation --no-deps`，复用已有 CUDA 版 PyTorch。基础模型及适配器不可变 revision、官方 LFS SHA256、标签方向与教师 manifest 已落实；推理检查结果见[验收记录](docs/验收记录.md)。

GPU 检查使用合成输入，包括卷积与选择性扫描的参考实现对照，以及 600 bp 长度 Mamba2 融合／非融合路径的前向、反向一致性：

```bash
python -m pytest -q tests/test_cuda_extensions.py
```

## 当前状态

尚无本轮真实 protein 预测结果。正式评价前还需完成开发 Study 观测核查、原定720个合成机制数据集、真实稀疏功效分析和完整比较成本验收。具体统计、失败试运行及来源证据统一见[验收记录](docs/验收记录.md)；执行与完成条件见[任务书](docs/任务书.md)。
