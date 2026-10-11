from dataclasses import replace

import pytest

from fprior.teacher import PlantCADTeacher, TeacherManifest, cache_key


def identity():
    return TeacherManifest('a'*40, 'b'*40, {'weights': 'c'*64},
                           {'negative': 0, 'positive': 1}, 600, 'float32',
                           'logit_positive_minus_logit_negative', False, 1)


def test_cache_covers_backbone_adapter_assets_and_rejects_invalid_sequences():
    manifest = identity()
    reference = 'A'*600
    key = cache_key(manifest, reference)
    assert key != cache_key(replace(manifest, base_revision='d'*40), reference)
    assert key != cache_key(replace(manifest, adapter_revision='e'*40), reference)
    assert key != cache_key(replace(manifest, assets_sha256={'weights': 'f'*64}), reference)
    with pytest.raises(ValueError, match='SEQUENCE_INVALID'):
        cache_key(manifest, 'A'*599+'N')
    with pytest.raises(ValueError, match='MANIFEST_INVALID'):
        replace(manifest, label_mapping={'negative': 1, 'positive': 0})


def test_score_uses_signed_raw_logits_and_deduplicates_roles():
    import torch
    teacher = PlantCADTeacher(None, None, identity(), 'cpu')
    requests = []

    def infer(sequences):
        requests.append(sequences)
        return torch.tensor([[3., -2.] if s[0] == 'A' else [-4., 2.]
                             for s in sequences])

    teacher.raw_logits = infer
    a, c = 'A'*600, 'C'*600
    assert teacher.score([a, c, a]) == [-5, 6, -5]
    assert requests == [[a, c]]
    assert teacher.score([c, a]) == [6, -5]
    assert len(requests) == 1


def test_raw_inference_uses_canonical_single_sequence_batches():
    import torch
    from types import SimpleNamespace

    class BatchSensitiveModel:
        def __init__(self):
            self.batch_sizes = []

        def eval(self):
            return self

        def get_base_model(self):
            return self

        def __call__(self, input_ids):
            self.batch_sizes.append(len(input_ids))
            value = input_ids[:, 0].float() + len(input_ids)
            return SimpleNamespace(logits=torch.stack([-value, value], dim=1))

    def tokenizer(sequences, **kwargs):
        assert kwargs['add_special_tokens'] is False and kwargs['truncation'] is False
        return {'input_ids': torch.tensor([[0 if s[0] == 'A' else 1]*600
                                          for s in sequences])}

    model = BatchSensitiveModel()
    teacher = PlantCADTeacher(model, tokenizer, identity(), 'cpu')
    a, c = 'A'*600, 'C'*600
    assert teacher.raw_logits([a, c]).tolist() == [[-1, 1], [-2, 2]]
    assert teacher.raw_logits([c]).tolist() == [[-2, 2]]
    assert model.batch_sizes == [1, 1, 1]
