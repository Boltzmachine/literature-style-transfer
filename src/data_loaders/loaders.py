from typing import Iterable, List, Tuple, Sequence, Optional
import random
import math
import allennlp

from allennlp.data.instance import Instance
from allennlp.data.data_loaders import DataLoader
from allennlp.data.samplers import BatchSampler
from allennlp.common.util import lazy_groups_of


@BatchSampler.register("domain-bucket")
class DomainBucketBatchSampler(BatchSampler):
    def __init__(
        self,
        batch_size: int,
        sorting_keys: List[str] = None,
        padding_noise: float = 0.1,
        drop_last: bool = False,
        shuffle: bool = True,
    ) -> None:
        self.sorting_keys = sorting_keys
        self.padding_noise = padding_noise
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def _argsort_by_padding(
        self, instances: Iterable[Instance]
    ) -> Tuple[List[int], List[List[int]]]:
        """
        Argsorts the instances by their padding lengths, using the keys in
        `sorting_keys` (in the order in which they are provided). `sorting_keys`
        is a list of `(field_name, padding_key)` tuples.
        """
        instances_with_lengths = []
        for instance in instances:
            # Make sure instance is indexed before calling .get_padding
            lengths = []
            noisy_lengths = []
            for field_name in self.sorting_keys:  # type: ignore
                lengths.append(len(instance.fields[field_name]))

            instances_with_lengths.append((noisy_lengths, lengths, instance))
        with_indices = [(x, i) for i, x in enumerate(instances_with_lengths)]
        with_indices.sort(key=lambda x: x[0][0])
        return (
            [instance_with_index[-1] for instance_with_index in with_indices],
            [instance_with_index[0][1] for instance_with_index in with_indices],
        )

    def get_batch_indices(self, instances: Sequence[Instance]) -> Iterable[List[int]]:
        indices, _ = self._argsort_by_padding(instances)
        batches = []
        for group in lazy_groups_of(indices, self.batch_size):
            batch_indices = list(group)
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            batches.append(batch_indices)
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def get_num_batches(self, instances: Sequence[Instance]) -> int:
        batch_count_float = len(instances) / self.batch_size
        if self.drop_last:
            return math.floor(batch_count_float)
        else:
            return math.ceil(batch_count_float)

    def get_batch_size(self) -> Optional[int]:
        return self.batch_size
