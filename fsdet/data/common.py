# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import random
import torch.utils.data as data

from fsdet.utils.serialize import PicklableWrapper

__all__ = ["MapDataset", "DatasetFromList"]


class MapDataset(data.Dataset):
    """
    Map a function over the elements in a datasets.

    Args:
        dataset: a datasets where map function is applied.
        map_func: a callable which maps the element in datasets. map_func is
            responsible for error handling, when error happens, it needs to
            return None so the MapDataset will randomly use other
            elements from the datasets.
    """

    def __init__(self, dataset, map_func):
        self._dataset = dataset
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work

        self._rng = random.Random(42)
        self._fallback_candidates = set(range(len(dataset)))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)

        while True:
            data = self._map_func(self._dataset[cur_idx])
            if data is not None:
                self._fallback_candidates.add(cur_idx)
                return data

            # _map_func fails for this idx, use a random new index from the pool
            retry_count += 1
            self._fallback_candidates.discard(cur_idx)
            cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]

            if retry_count >= 3:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Failed to apply `_map_func` for idx: {}, retry count: {}".format(
                        idx, retry_count
                    )
                )


class DatasetFromList(data.Dataset):
    """
    Wrap a list to a torch Dataset. It produces elements of the list as data.
    """

    def __init__(self, lst: list, copy: bool = True):
        """
        Args:
            lst (list): a list which contains elements to produce.
            copy (bool): whether to deepcopy the element when producing it,
                so that the result can be modified in place without affecting the
                source in the list.
        """
        self._lst = lst
        self._copy = copy

    def __len__(self):
        return len(self._lst)

    def __getitem__(self, idx):
        if self._copy:
            return copy.deepcopy(self._lst[idx])
        else:
            return self._lst[idx]
