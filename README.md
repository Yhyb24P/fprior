# fprior：功能先验多核岭回归项目

研究问题：参数固定的植物序列功能模型诱导的有符号、参考背景依赖的效应协方差，能否改善基因组预测？当前使用本地 SoyBase，以 `protein` 为主性状；[任务书](docs/任务书.md)定义大豆适配、执行顺序和验收条件。原拟南芥研究方案保留数学定义与解释边界，SoyBase 的真实输入适配与完整评价仍待实现。

`fprior` 是从 `at_pheno` 衍生的独立研究项目，只复用已有 Conda 环境和通用仓库管理原则。数据处理、算法、配置与运行产物均由本项目独立维护，开发约定见 [AGENTS.md](AGENTS.md)。本目录建立独立 Git 仓库，主分支使用 `main`；GitHub 仓库为 [Yhyb24P/fprior](https://github.com/Yhyb24P/fprior)。目录在磁盘上仍嵌套于 `at_pheno`，独立仓库从本项目当前源码快照开始，不导入父仓库历史。父仓库既有历史和暂存内容保持原样。

## 目录与数据

| 路径 | 用途 | 管理方式 |
| --- | --- | --- |
| `data/soybase/` | SoyBase 数据表及既有模型 | 本地保留，整体不入 Git |
| `data/paper_soybase/` | 已拉取的论文配套源码、数据及产物 | 本地保留，作为来源材料 |
| `data/results_soybase/` | 既有结果文件 | 本地保留，不作为本轮结果 |
| `docs/` | 研究方案、验收规范、讲义和配图 | 保留，纳入版本管理 |
| `src/fprior/` | 数学核心、输入预检及尚未实现的教师／评价接口 | 纳入版本管理 |
| `configs/funcprior.toml` | 唯一实验参数与输入预检配置 | 纳入版本管理 |
| `pyproject.toml` | Python 包定义 | 纳入版本管理 |

本地基因型与表型文件、已核实规模、主性状选择及来源缺口统一见[任务书 §二–三](docs/任务书.md)。原始 `1/2/3` 编码不能直接按 ALT 剂量使用；来源调用、缺失信息、参考映射及 protein 单位尚待落实。

数据处理代码统一放在本项目 `preprocessing/`，处理产物放在 `preprocessing/processed/`；算法只读取产物。新运行输出写入本项目 `runs/` 下独立目录，保留 `data/` 中既有文件的位置和内容。处理模块见 [preprocessing/README.md](preprocessing/README.md)；目前只实现已核实 ALT 剂量上的训练侧 QC 与填补，尚未生成真实处理产物。

## 文档

- [docs/任务书.md](docs/任务书.md) — SoyBase 执行合同、主性状决策、数据适配与阶段验收
- [docs/研究方案.md](docs/研究方案.md) — 研究问题、决策 D1–D8、等价推导、验证设计(权威定义)
- [docs/验收规范.md](docs/验收规范.md) — 输入输出合同、固定配置、求解与对照定义、测试矩阵、验收判据(权威约束)
- [docs/讲义.md](docs/讲义.md) — 科普讲解，数学定义与研究方案一致

## 代码

已实现 `io` 的元数据预检、ID 封存排除和顺序断言，`features` 的可信单倍型重建及 D/A/U/R 分解，`kernels` 的线性、对角、RBF、局部及已对齐单倍型计数核，`solver` 的有限单纯形、多核岭和并列选择。所有拟合统计量由调用方提供当前训练分区；单倍型类别目录构建、真实窗口筛选与端到端分区编排尚未实现。仅关闭精确零核，真实教师噪声阈值须在教师验证阶段确定。

`teacher` 的真实加载、`baselines` 的实验选参和 `eval` 的完整评价仍为占位。尚无真实训练入口，不将上述数学函数视为完整流水线。

配置：[configs/funcprior.toml](configs/funcprior.toml)。路径相对项目根目录解析；数值容差也从该配置读取。元数据预检只检查合同字段与文件存在性，不读取基因型或表型内容：

```bash
python -m fprior validate --config configs/funcprior.toml
```

目标未指定时返回 `TARGET_UNSPECIFIED`；必填输入缺失时返回 `MISSING_REQUIRED_INPUTS`，退出码为 2。文件齐全也只返回 `INPUTS_PRESENT_UNVERIFIED`，不能据此声明 `DATA_READY`。改变数据、教师、窗口、参考背景或搜索空间时须先落实合同并更新实验版本。

合成数学检查：

```bash
python -m pytest -q tests/test_core.py
```

本机 23 项通过，涵盖训练侧 QC／缺失填补、相位可信性、有符号窗口效应、分解与中心化恒等式、核训练归一化、测试批次隔离、原始—对偶预测一致性和失败停止。结果写在 `runs/soybase_funcprior_v1_checks/`；这不是验收规范的 720 组合成机制实验，也不包含真实 protein 预测。

## 环境

复用既有 Miniconda 环境 `atpheno`，不另建解释器环境。依赖统一由 [pyproject.toml](pyproject.toml) 管理；本机新增依赖安装在 `.deps/`，复用 Conda 环境已有的 NumPy、PyTorch 和 pytest。在本目录执行：

```bash
conda activate atpheno
export PYTHONPATH="$PWD/src:$PWD/.deps${PYTHONPATH:+:$PYTHONPATH}"
```

`.deps/` 不入 Git，也不是独立 Python 环境。下载复用已有代理环境变量，不在仓库保存代理地址或凭据。常规 CPU 依赖可用 `python -m pip install -e '.[dev,data,baselines]'` 安装；教师扩展需要使用兼容源码构建，不能直接套用其他 Torch 版本的 wheel。

本地 CPU 验证已通过：依赖无冲突，合成 Parquet 缺失值往返、FASTA 索引、正定线性求解、EN/GBRT 拟合、无界面绘图及微型 Transformers/LoRA 推理可用。验证未读取真实表型。

当前已实测 GPU 可访问：RTX 3090（24 GB，sm_86）、驱动 595.91.07；`atpheno` 的 PyTorch 2.14.1+cu130 使用 CUDA 13.0 运行时。系统另有 `/usr/local/cuda-13.2`，nvcc 13.2 可用；本次扩展构建使用 `.deps/nvidia/cu13` 中的 nvcc 13.0，保持编译器与 PyTorch 运行时匹配，不改动系统 CUDA 或 PyTorch。

教师的 Transformers／PEFT 接口版本参考 [PlantCAD 官方依赖](https://github.com/plantcad/plantcad/blob/main/requirements.txt)。Mamba 选择覆盖 PlantCAD Mamba2 接口的 2.x 版本，避免引入当前未使用的 Mamba3 依赖。扩展源码只调整为 C++20 和 sm_86 编译，并补齐本地 CUDA 编译目录的 `libcudart.so` 链接名；模型和内核数学代码保持原样。源码版本由 `teacher` extra 锁定，编译组件由 `cuda-build` extra 锁定。

编译 wheel、源码 commit、补丁、构建日志和 GPU 数值测试结果存放在本项目 `runs/environment_build/`，不入 Git。迁移机器时须按该 manifest 核对 Python、Torch、CUDA、C++ ABI 与 GPU 架构，重新构建；本机源码编译使用 `--no-build-isolation --no-deps`，复用已有 CUDA 版 PyTorch。固定权重 revision 和教师 manifest 仍待落实，扩展就绪不等于真实 PlantCAD 教师已验收。

GPU 检查使用合成输入，包括卷积与选择性扫描的参考实现对照，以及 600 bp 长度 Mamba2 融合／非融合路径的前向、反向一致性：

```bash
python -m pytest -q tests/test_cuda_extensions.py
```

本机 7 项 GPU 检查均通过，无失败或跳过；其中教师残差 RMSNorm 也与纯 PyTorch 实现作前向和梯度对照。当前结果及 wheel 校验和分别见 `runs/environment_build/tests.json`、`input_manifest.json` 和 `run_status.json`。没有可见 CUDA 设备时这些测试会标为跳过，不能把跳过当作 GPU 验证通过。

## 当前状态

B 阶段已完成第一批数学核心及其合成单元验证，尚未完成全部阶段验收。当前输入预检返回 `MISSING_REQUIRED_INPUTS`；真实处理产物、来源 manifest、原始调用、参考版本、protein 单位、FASTA、ACR BED、教师 manifest／revision，以及既有发现／封存清单的 SoyBase 对应关系和严格隔离接口仍缺失或待核实，详见任务书。真实训练保持停止，`at_pheno` 封存表型继续禁止访问。

后续独立开发先补齐基线与指标、合成机制验收和成本测量；真实数据路线须完成 A/C/D 阶段合同后再进入主比较。此前通过的 GPU 验证本轮未重复运行。

设计保留既有封存集，不另划、不默认为无封存；开发十折只服务发现集的组约束嵌套评价。父仓库配置与 ID 清单来源已定位，但跨物种适用性及隔离实现尚未验收；达到预注册门槛后须另行申请最终一次封存评价。来源与阻塞条件统一见[任务书 §3.2、§6.1](docs/任务书.md)。
