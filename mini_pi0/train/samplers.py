"""Training sampler strategies for dataset-specific access patterns.

This module keeps sample-order policy separate from dataset loading. The main
use case is LeRobot video-backed data: fully random shuffling causes expensive
random seeks into long video files, so a block-shuffle sampler preserves local
read locality while still changing training order each epoch.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence, Sized

from torch.utils.data import Sampler


class BlockShuffleSampler(Sampler[int]):
    """Shuffle contiguous blocks while preserving locality inside each block.

    Args:
        ordered_indices: Dataset indices ordered by storage locality.
        block_size: Number of samples per locality block.
        seed: Base random seed. The effective seed is ``seed + epoch``.
        shuffle_within_block: Whether to shuffle samples inside each local block.
            Keep this disabled for video-backed datasets when throughput is more
            important than per-sample randomization.

    Example:
        >>> sampler = BlockShuffleSampler(range(10), block_size=4, seed=0)
        >>> sorted(list(sampler))
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """

    def __init__(
        self,
        ordered_indices: Sequence[int],
        *,
        block_size: int,
        seed: int,
        shuffle_within_block: bool = False,
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}.")
        self._ordered_indices = tuple(int(index) for index in ordered_indices)
        self.block_size = int(block_size)
        self.seed = int(seed)
        self.shuffle_within_block = bool(shuffle_within_block)
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        """Yield one epoch of locality-aware shuffled indices."""

        rng = random.Random(self.seed + self.epoch)
        starts = list(range(0, len(self._ordered_indices), self.block_size))
        rng.shuffle(starts)
        for start in starts:
            block = list(self._ordered_indices[start : start + self.block_size])
            if self.shuffle_within_block:
                rng.shuffle(block)
            yield from block

    def __len__(self) -> int:
        """Return the number of yielded sample indices."""

        return len(self._ordered_indices)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to seed block shuffling.

        Args:
            epoch: Zero-based training epoch.
        """

        self.epoch = int(epoch)


def dataset_prefers_locality_sampler(dataset: object) -> bool:
    """Return whether a dataset asks for locality-aware sampling.

    Args:
        dataset: Dataset or ``torch.utils.data.Subset``-like wrapper.

    Returns:
        ``True`` when the dataset or an unwrapped base dataset exposes
        ``prefers_locality_sampler=True``.
    """

    current = dataset
    while current is not None:
        if bool(getattr(current, "prefers_locality_sampler", False)):
            return True
        current = getattr(current, "dataset", None)
    return False


def locality_order_for_dataset(dataset: Sized) -> tuple[int, ...]:
    """Build dataloader indices ordered by underlying storage locality.

    Args:
        dataset: Dataset or ``Subset``-like wrapper.

    Returns:
        Dataloader-facing indices sorted by their underlying source index when
        the dataset is a subset; otherwise ``range(len(dataset))``.
    """

    indices = getattr(dataset, "indices", None)
    if indices is None:
        return tuple(range(len(dataset)))
    return tuple(sorted(range(len(indices)), key=lambda position: int(indices[position])))


__all__ = [
    "BlockShuffleSampler",
    "dataset_prefers_locality_sampler",
    "locality_order_for_dataset",
]
