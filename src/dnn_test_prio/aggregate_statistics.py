"""A timed, online calculator of mins, max and stds of equally shaped arrays"""
from typing import List, Tuple

import numpy as np
from welford import Welford

from src.core.timer import Timer

AggStats = Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]


class AggregateStatisticsCollector:
    """A timed, online calculator of mins, max and stds of equally shaped arrays"""

    def __init__(self):
        self.is_initialized = False
        self.done = False

        self.mins: List[np.ndarray] = []
        self.maxs: List[np.ndarray] = []
        self.welfords: List[Welford] = []

        self.min_timer = Timer()
        self.max_timer = Timer()
        self.welford_timer = Timer()

    def _initialize(self, sample: List[np.ndarray]) -> None:
        for layer in sample:
            with self.min_timer:
                self.mins.append(layer)
            with self.max_timer:
                self.maxs.append(layer)
            with self.welford_timer:
                self.welfords.append(Welford(np.expand_dims(layer, 0)))
                self.is_initialized = True

    def track(self, badge: List[np.ndarray]) -> None:
        """Pass the next badge of arrays to be included in aggregate metrics."""
        if self.done:
            raise RuntimeError(
                "`get` has been called. calling it multiple times falsifies timer."
            )

        if not self.is_initialized:
            self._initialize([b[0] for b in badge])
            badge = [b[1:] for b in badge]

        with self.min_timer:
            self.mins = [
                np.minimum(self.mins[i], np.min(badge[i], axis=0))
                for i in range(len(badge))
            ]
        with self.max_timer:
            self.maxs = [
                np.maximum(self.maxs[i], np.max(badge[i], axis=0))
                for i in range(len(badge))
            ]

        with self.welford_timer:
            for i in range(len(badge)):
                self.welfords[i].add_all(badge[i])

    def get(self) -> AggStats:
        """Return the aggregated metrics."""
        with self.welford_timer:
            stds = [np.sqrt(s.var_s) for s in self.welfords]
        return self.mins, self.maxs, stds
