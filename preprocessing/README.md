# 本项目预处理

`dosages.py` 对已核实的 0/1/2 ALT 剂量与显式 NaN 缺失提供训练侧 QC、均值填补和中心化。每个实际内层／外层训练分区独立调用 `DosageTransform.fit`，测试只调用已拟合对象的 `transform`；阈值来自唯一实验配置。训练单态列保留，完全无调用列剔除。

本模块不读取原始 SoyBase 的 1/2/3 类别，也不证明其调用语义。原始调用恢复、REF/ALT 与参考映射、来源 manifest 和开发表型导出尚未实现；缺少合同不能生成正式处理产物。未来产物统一写入配置指定的 `preprocessing/processed/`，原始 `data/` 保持稳定，产物不入 Git。
