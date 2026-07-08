"""Run independent thunks concurrently with a bounded thread pool.

Judge calls are I/O-bound (an HTTP round-trip to the model API), so threads give
real parallelism despite the GIL — no async rewrite of the metrics needed. A
concurrency of 1 runs inline, identical to a plain loop, and is the default so
nothing changes unless you ask for it.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


def run_concurrent(thunks: Sequence[Callable[[], T]], max_concurrency: int = 1) -> list[T]:
    """Call each thunk and return results in the SAME order as ``thunks``.

    Exceptions propagate (the first one raised by ``.result()`` wins), so a
    failing task surfaces just as it would in a sequential loop.
    """
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")
    if max_concurrency == 1 or len(thunks) <= 1:
        return [thunk() for thunk in thunks]
    with ThreadPoolExecutor(max_workers=min(max_concurrency, len(thunks))) as pool:
        futures = [pool.submit(thunk) for thunk in thunks]
        return [future.result() for future in futures]
