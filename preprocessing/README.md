# 本项目预处理

`genotypes.py` 从官方 Wm82.gnm2 VCF 导出核染色体双等位 SNV。样本、位点保留 VCF 顺序，数组为 **variant × sample**：`dosages.npy` 是 float32 ALT 剂量，任一等位缺失则 NaN；`alleles.npy` 保留两个调用（-1 表示缺失）；`phased.npy` 保留相位分隔符；`called.npy` 表示两等位均有调用。相位有效须同时满足 phased 与 called，未定相调用不能视为已知单倍型。`samples.tsv`、`variants.tsv` 对应数组轴，`gt_audit.json` 汇总包括被排除位点在内的全部原始 GT。导出拒绝覆盖已有正式产物，不进行频率筛选、填补或中心化。

在项目根目录、既有环境及 README 所述依赖路径下运行：

```bash
PYTHONPATH="$PWD:$PWD/src:$PWD/.deps" python -m preprocessing.genotypes --config configs/funcprior.toml
```

`dosages.py` 的 `DosageTransform` 接收 **sample × variant** 剂量，每个实际内外层训练分区独立 `fit`，测试只使用该对象 `transform`。读取导出时先按冻结 ID 选择样本和位点并转置，再拟合训练统计量；不得先用全队列 QC 或均值处理。训练单态列保留，完全无调用列剔除。

路径和数值参数只维护于 [configs/funcprior.toml](../configs/funcprior.toml)。产物在配置指定的 `preprocessing/processed/`，不入 Git。转换不读取表型，也不表示完整训练数据已就绪；protein 观测规则、新队列划分、隔离入口和 ACR 验收见[任务书](../docs/任务书.md)。

`phenotypes.py` 的 `protein_study` 解析 GRIN 描述符 51016 的逐 Study 材料表，验证上下文、列名、行结构及材料身份，保留官方 VALUE、观测说明和库存文本；原始解析不聚合、不填补；`aggregate_studies` 仅在授权的划分前来源重建中按配置作先 Study 内、再 Study 间等权均值，不能在冻结后重放全量读取。方法身份由 `method_document` 核验。来源完整性与观测选择合同尚未通过时，原始记录不能作为开发表型入口。

`regions.py` 从记录完整的源窗口及两组装 SAM 比对结果验收完整零编辑匹配、原位回映射、次要比对歧义、同染色体及独立 FASTA 序列一致性。按 1-based SNP 对 0-based 半开窗口求交，再以配置固定哈希选择不重叠多 SNP 窗。正式面板是完整源中心 600 bp 窗口的参考转移，不在目标参考重新取中点；参数和映射版本只维护在唯一配置。

`splits.py` 用全目录原始调用的一致性构建近重复连通组，覆盖不足的候选对保守连接；不做全队列频率筛选或填补。整组封存、平衡外层折及整组哈希前缀预算／内层折由配置固定，已有正式清单不覆盖。开发标签及无值偏移索引在冻结前隔离导出，冻结后仅通过 `io.load_development_phenotypes` 读取许可开发行。源全量表型及隔离封存标签不再读取。

`regions.audit_window_calls` 在实际训练 ID 上审计原始调用有效性、缺失、多未定相杂合、训练 QC 与多态性，不读取表型、不生成填补序列、不补选窗口。调用有效、至少两个 QC SNP 与至少两个多态 SNP 的窗数分别报告；教师可分辨信号另行验收。
