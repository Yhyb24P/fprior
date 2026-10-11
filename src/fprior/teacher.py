"""教师接口协议(规范 3 教师接口)。

协议:对所有参考/单变异/单倍型序列使用相同窗口、方向、tokenizer 与推理配置;
eval() 模式、关闭梯度与 dropout;默认正链;二分类头输出
F(h) = logits[:, positive] - logits[:, negative](直接读 logits,
不由概率反求);按 (teacher_revision, tokenizer_hash, output_definition, dtype,
sequence_sha256) 缓存;同一序列不得因参考/变异角色不同获得不同输出。

缓存键或权重 revision 任一字段未知时 fail-fast,不得声称教师复现完成。

当前为框架占位,未实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TeacherManifest:
    """teacher_manifest.json 的必填字段(规范 2.1);任一字段缺失即不可声称教师复现。"""

    repo_commit: str
    weight_revision: str
    weight_sha256: str
    tokenizer: str
    label_mapping: dict
    sequence_length: int
    dtype: str


class Teacher(Protocol):
    """参数固定的窗口教师:输入编码序列,输出标量 F(h)。"""

    def score(self, sequences: list[str]) -> list[float]:
        """对一批等长(默认 600bp)序列返回 logit 差标量,float32 精度可接受。"""
        ...


def cache_key(manifest: TeacherManifest, sequence: str) -> str:
    """按 (revision, tokenizer_hash, output_definition, dtype, sequence_sha256) 生成缓存键。"""
    raise NotImplementedError


def load_teacher(manifest_path: str) -> Teacher:
    """从 teacher_manifest.json 加载 plantcad ACR 教师;

    权重/分词器/标签映射任一未落实则报错停止功能分支(不用 GWAS 峰临时替代)。
    """
    raise NotImplementedError
