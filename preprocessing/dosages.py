"""Training-only QC and imputation for already verified ALT dosages."""
from dataclasses import dataclass
import numpy as np


def _dosages(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError('DOSAGE_SHAPE_INVALID')
    called = ~np.isnan(array)
    if not np.isin(array[called], [0.0, 1.0, 2.0]).all():
        raise ValueError('DOSAGE_INVALID: expected 0/1/2 or NaN')
    return array, called


@dataclass(frozen=True)
class DosageTransform:
    retained: np.ndarray
    means: np.ndarray
    call_rates: np.ndarray
    maf: np.ndarray
    n_variants: int

    @classmethod
    def fit(cls, train, *, min_call_rate: float, maf_min: float):
        if not 0 <= min_call_rate <= 1 or not 0 <= maf_min <= .5:
            raise ValueError('QC_THRESHOLD_INVALID')
        array, called = _dosages(train)
        counts = called.sum(axis=0)
        means = np.divide(np.nansum(array, axis=0), counts,
                          out=np.zeros(array.shape[1]), where=counts > 0)
        rates = counts / len(array)
        maf = np.minimum(means / 2, 1 - means / 2)
        retained = (counts > 0) & (rates >= min_call_rate) & (maf >= maf_min)
        return cls(retained, means[retained], rates, maf, array.shape[1])

    def transform(self, values) -> tuple[np.ndarray, np.ndarray]:
        """Return Q and X; preserve training-monomorphic columns in testing."""
        array, _ = _dosages(values)
        if array.shape[1] != self.n_variants:
            raise ValueError('VARIANT_ORDER_MISMATCH: wrong number of columns')
        selected = array[:, self.retained]
        q = np.where(np.isnan(selected), self.means, selected)
        return q, q - self.means
