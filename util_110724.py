from collections.abc import (
    Generator
)
import torch
from typing import Optional

from util_092324 import get_random_reshuffler
from util_110424 import is_ensembled
from util_110424 import get_minibatch as get_array_minibatch


class Embedding(torch.nn.Module):
    """
    Ensemble-ready embedding.

    Arguments
    ---------
    config : `dict`
        Configuration dictionary. Required key-value pairs:
        `"device"` : `str`
            The device to store parameters on.
        `"ensemble_shape"` : `tuple[int]`
            The shape of the ensemble of affine transformations
            the model represents.
        `"float_dtype"` : `torch.dtype`
            The floating point datatype to use for the parameters.
    embedding_dim : `int`
        The number of embedding dimensions.
    vocabulary_size : `int`
        The number of vocabulary entries.

    Calling
    -------
    Instance calls require one positional argument:
    indices : `torch.Tensor`
        The index tensor. It is required to be one of the following shapes:
        1. `ensemble_shape + batch_shape`
        2. `batch_shape`

        Upon a call, the model thinks we're in the first case
        if the first `len(ensemble_shape)` many entries of the
        shape of the input tensor is `ensemble_shape`.
    """
    def __init__(
        self,
        config: dict,
        embedding_dim: int,
        vocabulary_size: int
    ):
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.empty(
            config["ensemble_shape"] + (vocabulary_size, embedding_dim),
            device=config["device"],
            dtype=config["float_dtype"]
        ).normal_())


    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        ensemble_shape = self.embedding.shape[:-2]
        embedding_dim = self.embedding.shape[-1]

        indices = to_ensembled(ensemble_shape, indices)
        indices_shape = indices.shape
        indices = indices.flatten(len(ensemble_shape))
        indices = indices[..., None].expand(
            indices.shape + (embedding_dim,)
        )

        embedding = self.embedding.gather(
            -2,
            indices
        )
        embedding = embedding.reshape(
            indices_shape + (embedding_dim,)
        )

        return embedding


def get_array_sequence_keys(
    dataset: dict,
    indptr_key="indptr"
) -> tuple[tuple[str], tuple[str]]:
    """
    Get the array and sequence keys of the dataset,
    which are defined as follows:

    Given a key-value pair, we say that the key is
    1. an array key, if the value tensor has the dataset size
        at the batch dimension and
    2. a sequence key, if the value tensor has the total number of tokens
        at the batch dimension.

    For the definition of batch dimension and dataset size,
    see `get_dataset_size`.

    Parameters
    ----------
    dataset : `dict`
        The dataset.
    indptr_key : `str`, optional
        The key of the index pointer tensor. Default: `indptr`.

    Returns
    -------
    The pair of
    1. the tuple of array keys and
    2. the tuple of sequence keys.
    """
    if indptr_key in dataset:
        indptr = dataset[indptr_key]
        ensemble_shape = indptr.shape[:-1]

        ensemble_dim = len(ensemble_shape)
        entry_num = indptr.shape[-1] - 1
        token_num = indptr.max()

        if entry_num == token_num:
            raise ValueError(
                f"The number of dataset entries equals the maximum number of tokens per ensemble member: {entry_num}"
            )

        array_keys = tuple((
            key
            for key, value in dataset.items()
            if (
                key != indptr_key
            and value.shape[ensemble_dim] == entry_num
            )
        ))
        sequence_keys = tuple((
            key
            for key, value in dataset.items()
            if (
                (key != indptr_key)
            and (value.shape[ensemble_dim] == token_num)
            )
        ))
    else:
        array_keys = tuple(dataset)
        sequence_keys = ()

    return array_keys, sequence_keys


def get_dataloader_random_reshuffle(
    config: dict,
    dataset: dict,
    indptr_key="indptr",
    minibatch_size: Optional[int] = None
) -> Generator[dict]:
    """
    Given a dataset as a dictionary with tensor values,
    creates a random reshuffling (without replacement) dataloader
    that yields minibatch dictionaries indefinitely.
    Support arbitrary ensemble shapes and sequential data.

    Parameters
    ----------
    config : `dict`
        Configuration dictionary. Required key-value pair:
        ensemble_shape : tuple[int]
            The required ensemble shapes of the outputs.
    dataset : `dict`
        Dataset with `torch.Tensor` values.
    indptr_key : `str`, optional
        If the dataset has sequential entries,
        then this is the key of the index pointer tensor. Default: `indptr`.
    minibatch_size : `int`, optional
        Minibatch size. If not given, it is `config["minibatch_size"]`.

    Returns
    -------
    A generator of minibatch dictionaries.
    Their extra keys `"mask"` map to the mask tensors of
    entries that are not padding entries.
    """
    if minibatch_size is None:
        minibatch_size = config["minibatch_size"]

    dataset = {
        key: to_ensembled(config["ensemble_shape"], value)
        for key, value in dataset.items()
    }
    
    dataset_size = get_dataset_size(config, dataset, indptr_key=indptr_key)

    random_reshuffler = get_random_reshuffler(
        dataset_size,
        minibatch_size,
        device=config["device"],
        ensemble_shape=config["ensemble_shape"]
    )

    for indices in random_reshuffler:
        yield get_minibatch(dataset, indices, indptr_key=indptr_key)


def get_dataset_size(
    config: dict,
    dataset: dict,
    indptr_key="indptr"
) -> int:
    """
    Get the size of the potentially ensembled dataset,
    which is defined as follows:

    Let us call *batch dimension* of a tensor the dimension
        at the entry of its shape
        that comes after the ensemble shape entries
    1. If the dataset has an index pointer tensor,
        then its size is the batch the index pointer tensor minus 1.
    2. Otherwise, we take any value of the dataset.
        Then the dataset size is the batch dimension of the value tensor.

    Parameters
    ----------
    config : `dict`
        Configuration dictionary. Requires key-value pair:
        `"ensemble_shape"` : `tuple[int]`
            Ensemble shape.
    dataset : `dict`
        The dataset.
    indptr_key : `str`, optional
        The key of the index pointer tensor. Default: `indptr`.

    Returns
    -------
    The dataset size.
    """
    if indptr_key in dataset:
        dataset_size = dataset[indptr_key].shape[-1] - 1
    else:
        ensemble_dim = len(config["ensemble_shape"])
        dataset_size = next(iter(dataset.values())).shape[ensemble_dim]

    return dataset_size


def get_minibatch(
    dataset: dict,
    indices: torch.Tensor,
    indptr_key="indptr",
) -> dict:
    """
    Returns the minibatch of an ensembled dataset
    given by ensembled indices.

    Parameters
    ----------
    dataset: `dict`
        A dataset given as a dictionary with `torch.Tensor` values.
    indices: `torch.Tensor`
        An index tensor for the dataset.
        Each value in the dataset should have shape prefixed by
        the shape of the index tensor.
    indptr_key : `str`, optional
        If the dataset has sequential entries,
        then this is the key of the index pointer tensor. Default: `indptr`.

    Returns
    -------
    The minibatch given by the dataset and the index tensor.
    The extra key `"mask"` is mapped to the mask tensor of
    entries that are not padding entries.
    """
    minibatch = {}

    dense_keys, sequence_keys = get_array_sequence_keys(
        dataset, indptr_key=indptr_key
    )

    if len(dense_keys) > 0:
        minibatch.update(get_array_minibatch(
            {
                key: dataset[key]
                for key in dense_keys
            },
            indices
        ))

    if len(sequence_keys) > 0:
        indptr_left, indptr_right = (
            dataset[indptr_key].gather(
                -1,
                i
            )
            for i in (indices, indices + 1)
        )
        
        sizes = indptr_right - indptr_left
        sizes_max = sizes.max()
        
        sequence_indices = (
            indptr_left[..., None]
          + torch.arange(sizes_max, device=indices.device)
        )

        mask: torch.Tensor = sequence_indices < indptr_right[..., None]
        minibatch["mask"] = mask
        sequence_indices[~mask] = 0

        minibatch_shape = indices.shape
        for key in sequence_keys:
            data_raw: torch.Tensor = dataset[key]
            data = data_raw.gather(
                len(minibatch_shape) - 1,
                sequence_indices.reshape(
                    minibatch_shape[:-1]
                  + (minibatch_shape[-1] * sizes_max,)
                  + data_raw.shape[len(minibatch_shape):]
                )
            ).reshape(
                minibatch_shape
              + (sizes_max,)
              + data_raw.shape[len(minibatch_shape):]
            )
            minibatch[key] = data

    return minibatch


def to_ensembled(
    ensemble_shape: tuple[int],
    tensor: torch.Tensor
) -> torch.Tensor:
    """
    We say that a tensor is *ensembled*,
    if its shape starts by the ensemble shape.

    This function converts a tensor to an ensembled tensor.
    """
    if is_ensembled(ensemble_shape, tensor):
        return tensor
    
    return tensor.broadcast_to(
        ensemble_shape + tensor.shape
    )