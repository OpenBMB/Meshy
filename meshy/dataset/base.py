from typing import Any, List

import datasets

from meshy.utils.sample import Sample, SampleBuilder


class Dataset:

    def __init__(self, hf_kwargs: dict[str, Any], batch_size: int, seed: int | None = None):
        self.dataset = datasets.load_dataset(**hf_kwargs)
        self.batch_size = batch_size
        self.index = 0
        if seed is not None:
            self.dataset = self.dataset.shuffle(seed=seed)

    def next_batch(self, builder: SampleBuilder) -> List[Sample]:
        if self.index + self.batch_size >= len(self.dataset):
            return []
        batch_data = self.dataset.select(range(self.index, self.index + self.batch_size))
        self.index += self.batch_size
        return [self.apply_chat_template(data, builder) for data in batch_data]

    def apply_chat_template(self, data: dict, builder: SampleBuilder) -> Sample:
        raise NotImplementedError
