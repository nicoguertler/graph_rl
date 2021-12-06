from typing import Any, Tuple, Union, Optional

import numpy as np

from tianshou.data import ReplayBuffer
from tianshou.data.batch import Batch

class SegmentedReplayBuffer:

    def __init__(self, sizes: dict, ratios = None, **kwargs) -> None:
        self._buffers = {name: ReplayBuffer(size) for name, size in sizes.items()}
        self._len = sum(sizes.values())
        if ratios is not None:
            assert sizes.keys() == ratios.keys()
        self._ratios = ratios

    def add(self,
            name, 
            obs: Union[dict, np.ndarray],
            act: Union[np.ndarray, float],
            rew: float,
            done: bool,
            obs_next: Optional[Union[dict, np.ndarray]] = None,
            info: dict = {},
            policy: Optional[Union[dict, Batch]] = {},
            **kwargs) -> None:
        """Add a batch of data into replay buffer."""
        self._buffers[name].add(obs, act, rew, done, obs_next, info, policy, **kwargs)

    def sample_and_process(self, batch_size: int, process_fn: callable) -> Batch:
        """Get a random sample from buffer with size equal to batch_size.

        :return: Sample data.
        """
        if self._ratios is None:
            n_total = float(self.__len__())
            batch_sizes = {key: int(batch_size*len(buff)/n_total) for key, buff 
                    in self._buffers.items()}
        else:
            batch_sizes = {key: int(batch_size*ratio) for key, ratio in self._ratios.items()}
        # make sure the overall batch size is as desired
        batch_sizes[list(self._buffers.keys())[0]] += batch_size - sum(batch_sizes.values())

        batch = Batch()
        for name, bs in batch_sizes.items():
            sub_batch, sub_indices = self._buffers[name].sample(bs)
            sub_batch = process_fn(sub_batch, self._buffers[name], sub_indices)
            batch.cat(sub_batch)

        return batch

    def __len__(self) -> int:
        return sum(len(b) for b in self._buffers.values())




