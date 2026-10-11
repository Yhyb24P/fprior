"""Versioned, frozen PlantCAD2 ACR logits with a content-addressed sequence cache."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Protocol


@dataclass(frozen=True)
class TeacherManifest:
    base_revision: str
    adapter_revision: str
    assets_sha256: dict[str, str]
    label_mapping: dict[str, int]
    sequence_length: int
    dtype: str
    output_definition: str
    add_special_tokens: bool
    inference_batch_size: int

    def __post_init__(self):
        if (not re.fullmatch('[0-9a-f]{40}', self.base_revision)
                or not re.fullmatch('[0-9a-f]{40}', self.adapter_revision)
                or not self.assets_sha256
                or any(not re.fullmatch('[0-9a-f]{64}', h) for h in self.assets_sha256.values())
                or self.label_mapping != {'negative': 0, 'positive': 1}
                or self.sequence_length != 600 or self.dtype != 'float32'
                or self.output_definition != 'logit_positive_minus_logit_negative'
                or self.add_special_tokens is not False or self.inference_batch_size != 1):
            raise ValueError('TEACHER_MANIFEST_INVALID')


class Teacher(Protocol):
    def score(self, sequences: list[str]) -> list[float]: ...


def cache_key(manifest: TeacherManifest, sequence: str) -> str:
    if len(sequence) != manifest.sequence_length or set(sequence) - set('ACGT'):
        raise ValueError('TEACHER_SEQUENCE_INVALID')
    identity = dict(asdict(manifest), sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest())
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class PlantCADTeacher:
    def __init__(self, model, tokenizer, manifest: TeacherManifest, device):
        self.model, self.tokenizer, self.manifest, self.device = model, tokenizer, manifest, device
        self._cache = {}

    def raw_logits(self, sequences: list[str]):
        """Uncached inference for acceptance; no softmax or probability inversion."""
        import torch
        if not sequences:
            return torch.empty((0, 2), dtype=torch.float32)
        for sequence in sequences:
            cache_key(self.manifest, sequence)
        tokens = self.tokenizer(sequences, add_special_tokens=False, padding=False,
                                truncation=False, return_tensors='pt')['input_ids']
        if tokens.shape != (len(sequences), self.manifest.sequence_length):
            raise ValueError('TEACHER_TOKEN_LENGTH_MISMATCH')
        self.model.eval()
        outputs = []
        with torch.inference_mode():
            # PEFT has already installed LoRA and the saved score head in this model.
            # Its generic wrapper forwards attention_mask, unsupported by Caduceus.
            # Canonical single-sequence inference avoids batch-shape FP32 drift.
            # This is part of the manifest and cache identity, never role-dependent.
            for row in tokens:
                outputs.append(self.model.get_base_model()(
                    input_ids=row[None, :].to(self.device)).logits)
            logits = torch.cat(outputs)
        if logits.shape != (len(sequences), 2) or not torch.isfinite(logits).all():
            raise ValueError('TEACHER_LOGITS_INVALID')
        return logits.detach().float().cpu()

    def score(self, sequences: list[str]) -> list[float]:
        keys = [cache_key(self.manifest, sequence) for sequence in sequences]
        pending = {}
        for key, sequence in zip(keys, sequences):
            if key not in self._cache:
                pending.setdefault(key, sequence)
        if pending:
            logits = self.raw_logits(list(pending.values()))
            values = logits[:, self.manifest.label_mapping['positive']] - logits[:, self.manifest.label_mapping['negative']]
            self._cache.update(zip(pending, values.tolist()))
        return [self._cache[key] for key in keys]


def load_teacher(manifest_path: str, *, device='cuda') -> PlantCADTeacher:
    """Load only hash-verified, local assets; reject incomplete backbone or adapter."""
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = Path(manifest_path).resolve()
    record = json.loads(path.read_text())
    manifest = TeacherManifest(**record['identity'])
    assets = record['assets']
    if set(assets) != set(manifest.assets_sha256):
        raise ValueError('TEACHER_ASSET_SET_MISMATCH')
    for name, asset in assets.items():
        file = path.parent / asset['path']
        digest = hashlib.sha256()
        with file.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        if digest.hexdigest() != manifest.assets_sha256[name]:
            raise ValueError(f'TEACHER_ASSET_HASH_MISMATCH:{name}')
    base_dir = (path.parent / record['base_directory']).resolve()
    adapter_dir = (path.parent / record['adapter_directory']).resolve()
    tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True)
    base, info = AutoModelForSequenceClassification.from_pretrained(
        base_dir, local_files_only=True, trust_remote_code=True, num_labels=2,
        id2label={0: 'NEGATIVE', 1: 'POSITIVE'}, label2id={'NEGATIVE': 0, 'POSITIVE': 1},
        torch_dtype=torch.float32, output_loading_info=True)
    if info.get('mismatched_keys') or info.get('error_msgs'):
        raise ValueError('TEACHER_BASE_LOAD_MISMATCH')
    if any(not key.startswith('score.') for key in info.get('missing_keys', [])):
        raise ValueError(f"TEACHER_BACKBONE_INCOMPLETE:{info['missing_keys']}")
    if set(info.get('unexpected_keys', [])) - {'lm_head.complement_map', 'lm_head.lm_head.weight'}:
        raise ValueError('TEACHER_UNEXPECTED_BASE_WEIGHTS')
    model = PeftModel.from_pretrained(base, adapter_dir, local_files_only=True,
                                     is_trainable=False)
    expected = load_file(str(adapter_dir/'adapter_model.safetensors'))
    restored = get_peft_model_state_dict(model)
    if set(restored) != set(expected) or any(
            not torch.equal(restored[name].cpu(), tensor) for name, tensor in expected.items()):
        raise ValueError('TEACHER_ADAPTER_RESTORATION_MISMATCH')
    if not any('score' in name and name.endswith('weight') for name in expected):
        raise ValueError('TEACHER_CLASSIFIER_HEAD_MISSING')
    model.requires_grad_(False)
    model.to(device=device, dtype=torch.float32).eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('TEACHER_NOT_FROZEN')
    result = PlantCADTeacher(model, tokenizer, manifest, device)
    result.loading_info = info
    return result
