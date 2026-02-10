"""Sample functionality for Ibis backend."""

import warnings
from typing import List

from ibis import Table

from ydata_profiling.config import Settings
from ydata_profiling.model.sample import Sample, get_sample


@get_sample.register
def get_sample_ibis(config: Settings, df: Table) -> List[Sample]:
    """Obtains a sample from the table.

    Args:
        config: Settings object
        df: the ibis Table

    Returns:
        a list of Sample objects
    """
    samples: List[Sample] = []
    if df.count().execute() == 0:
        return samples

    n_head = config.samples.head
    if n_head > 0:
        samples.append(
            Sample(id="head", data=df.head(n_head).to_pandas(), name="First rows")
        )

    n_tail = config.samples.tail
    if n_tail > 0:
        warnings.warn(
            "tail sample not implemented for Ibis. "
            "Set config.samples.n_tail to 0 to disable this warning"
        )

    n_random = config.samples.random
    if n_random > 0:
        warnings.warn(
            "random sample not implemented for Ibis. "
            "Set config.samples.n_random to 0 to disable this warning"
        )

    return samples
