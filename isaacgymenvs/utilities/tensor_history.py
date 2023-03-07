""" Log a series of tensors to file """

import h5py
import numpy as np
import torch 
import typing

class TensorHistory:
    def __init__(self, 
        max_len: int, 
        tensor_shape: typing.Tuple[int, ...], 
        dtype: typing.Any = torch.float32,
        device: typing.Any = None
    ):
        if device is None: device = torch.device('cpu')
        self.tensor_history = torch.zeros((max_len,) + tensor_shape, device=device, dtype=dtype)
        self.current_idx = 0

    def update(self, value: torch.Tensor, idx: typing.Optional[int] = None):
        if idx is None:
            idx = self.current_idx
            self.current_idx += 1
        self.tensor_history[idx] = value

    def __len__(self):
        return self.current_idx

    def get_history(self):
        return self.tensor_history[:self.current_idx + 1]

    def clear(self):
        self.tensor_history = torch.zeros_like(self.tensor_history)
        self.current_idx = 0


# WIP: Implement batched tensor history that can clear / reset / get subsets
# Necessary if we ever want to test with early termination
# However, it's unclear how to correctly implement this
class TensorBatchHistory:
    """ Utility class for recording histories of batched tensors """
    def __init__(self, 
        max_len: int, 
        max_batch_size: int, 
        tensor_shape: typing.Tuple[int, ...], 
        dtype: typing.Any = torch.float32,
        device: typing.Any = None
    ):
        if device is None: device = torch.device('cpu')
        self.tensor_history = torch.zeros((max_len, max_batch_size) + tensor_shape, device=device, dtype=dtype)
        self.current_idx = torch.zeros((max_batch_size,))

class TensorIO:
    """ Read/write tensors to disk.
    
    Wrapper around h5py dataset """

    def __init__(self,
        file: h5py.File,
        tensor_shape: typing.Tuple[int, ...],
        dataset_name: str = 'tensor' 
    ):
        self.file = file 
        self.dataset_name = dataset_name
        self.tensor_shape = tensor_shape
        self.curr_size = 0
        self.max_size = 16
        self.dataset = file.create_dataset(dataset_name, (16, *tensor_shape), chunks=True, maxshape=(None, *tensor_shape))

    @staticmethod 
    def new_file(
        filepath: str, 
        append: bool = False,
    ):
        """ Initialize a new file to write to """
        mode = 'a' if append else 'w' 
        return h5py.File(filepath, mode)

    @staticmethod
    def new(
        filepath: str, 
        tensor_shape: typing.Tuple[int, ...],
        dataset_name: str = 'tensor', 
        append: bool = False
    ):
        file = TensorIO.new_file(filepath, append)
        return TensorIO(file, tensor_shape, dataset_name)

    def _is_idx_valid(self, idx: typing.Any) -> bool:
        if isinstance(idx, slice):
            return max(idx) < self.max_size
        elif isinstance(idx, int):
            return idx < self.max_size

    def read(self, idx: typing.Any) -> torch.Tensor:
        return self.dataset[idx]

    def resize(self, new_size: int):
        self.max_size = new_size
        self.dataset.resize((self.max_size, *self.tensor_shape))

    def _get_next_size(self):
        return self.max_size * 2

    def write(self, value: np.ndarray):
        if len(value.shape) > len(self.tensor_shape):
            # Assume first dimension is batch dimension 
            num_rows = value.shape[0]
        else: 
            num_rows = 1

        start_idx = self.curr_size
        end_idx = start_idx + num_rows

        while not self._is_idx_valid(end_idx):
            self.resize(self._get_next_size())
        self.curr_size = end_idx
        self.dataset[start_idx: end_idx] = value
        self.dataset.attrs['size'] = self.curr_size

        self._print_debug_info()

    def _print_debug_info(self):
        print(f"dataset: {self.dataset_name} | curr size: {self.curr_size}")