import torch

def get_minibatch(
    dataset: dict,
    indices: torch.Tensor
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

    Returns
    -------
    The minibatch given by the dataset and the index tensor.
    """
    # print(indices.shape, indices.min(), indices.max())
    # for key, value in dataset.items():
    #     print(key, value.shape)
    return {
        key: value.gather(
            len(indices.shape) - 1,
            indices.reshape(
                indices.shape + (1,) * (
                    len(value.shape) - len(indices.shape)
                )
            ).expand(
                indices.shape + value.shape[len(indices.shape):]
            )
        )
        for key, value in dataset.items()
    }


def is_ensembled(
    ensemble_shape: tuple[int],
    tensor: torch.Tensor
) -> bool:
    """
    We view `tensor` as *ensembled* if it is prefixed by `ensemble_shape`,
    that is its slice of the first `len(ensemble_shape)` entries
    is `ensemble_shape`.

    This function checks this condition.
    """
    return tensor.shape[:len(ensemble_shape)] == ensemble_shape