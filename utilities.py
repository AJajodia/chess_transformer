from collections.abc import Generator
import torch

def get_shuffled_indices(
    dataset_size: int,
    device="cpu",
    ensemble_shape=(),
) -> torch.Tensor:
    """
    Get a tensor of a batch of shuffles of indices `0,...,dataset_size - 1`.

    Parameters
    ----------
    dataset_size : int
        The size of the dataset the indices of which to shuffle
    device : int | str | torch.device, optional
        The device to store the resulting tensor on. Default: "cpu"
    ensemble_shape : tuple[int], optional
        The batch shape of the shuffled index tensors. Default: ()
    """
    total_shape = ensemble_shape + (dataset_size,)
    uniform = torch.rand(
        total_shape,
        device=device
    )
    indices = uniform.argsort(dim=-1)

    return indices

def get_random_reshuffler(
    dataset_size: int,
    minibatch_size: int,
    device="cpu",
    ensemble_shape=()
) -> Generator[torch.Tensor]:
    """
    Generate minibatch indices for a random shuffling dataloader.
    Supports arbitrary ensemble shapes.

    Parameters
    ----------
    dataset_size : int
        The size of the dataset to yield batches of minibatch indices for.
    minibatch_size : int
        The minibatch size.
    device : int | str | torch.device, optional
        The device to store the index tensors on. Default: "cpu"
    ensemble_shape : tuple[int], optional
        The ensemble shape of the minibatch indices. Default: ()
    """
    q, r = divmod(dataset_size, minibatch_size)
    minibatch_num = q + min(1, r)
    minibatch_index = minibatch_num
    while True:
        if minibatch_index == minibatch_num:
            minibatch_index = 0
            shuffled_indices = get_shuffled_indices(
                dataset_size,
                device=device,
                ensemble_shape=ensemble_shape
            )

        yield shuffled_indices[
            ...,
            minibatch_index * minibatch_size
        :(minibatch_index + 1) * minibatch_size
        ]

        minibatch_index += 1

def get_dataloader_random_reshuffle(
    config: dict,
    features: torch.Tensor,
    labels: torch.Tensor
) -> Generator[tuple[torch.Tensor, torch.Tensor]]:
    """
    Given a feature and a label tensor,
    creates a random reshuffling (without replacement) dataloader
    that yields pairs `minibatch_features, minibatch_labels` indefinitely.
    Support arbitrary ensemble shapes.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Required keys:
        ensemble_shape : tuple[int]
            The required ensemble shapes of the outputs.
        minibatch_size : int
            The size of the minibatches.
    features : torch.Tensor
        Tensor of dataset features.
        We assume that the first dimension is the batch dimension
    labels : torch.Tensor
        Tensor of dataset labels.

    Returns
    -------
    A generator of tuples `minibatch_features, minibatch_labels`.
    """
    for indices in get_random_reshuffler(
        len(labels),
        config["minibatch_size"],
        device=config["device"],
        ensemble_shape=config["ensemble_shape"]
    ):
        yield features[indices], labels[indices]