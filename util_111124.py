from collections.abc import (
    Callable
)
import math
import torch
import torch.nn.functional as F
import tqdm

from util_110724 import (
    get_dataset_size,
    get_minibatch,
    to_ensembled
)


    

class DictReLU(torch.nn.Module):
    """
    Applies ReLU elementwise to appropriate values in a dictionary.

    Calling
    -------
    Instance calls require one positional argument:
    batch : `dict`
        The input data dictionary. ReLU is applied to the following
        required and optional keys:
        `"features"` : `torch.Tensor`
            Tensor of element-level features, of shape
            `batch_shape + (sequence_dim, in_features)` or
            `ensemble_shape + batch_shape + (sequence_dim, in_features)`
        `"extra"` : `torch.Tensor`, optional
            If given, we view it as the tensor of set-level features,
            of shape `batch_shape + (extra_features,)` or
            `ensemble_shape + batch_shape + (extra_features,)`.
    """
    def forward(self, batch: dict) -> dict:
        if "extra" in batch:
            batch = batch | {"extra": F.relu(batch["extra"])}

        return batch | {"features": F.relu(batch["features"])}


def evaluate_model(
    config: dict,
    dataset: dict,
    get_metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    model: torch.nn.Module,
    indptr_key="indptr",
    target_key="target"
) -> torch.Tensor:
    """
    Evaluate a model on a supervised dataset.

    Parameters
    ----------
    config : `dict`
        Configuration dictionary. Required key-value pair:
        `"minibatch_size_eval"` : `int`
            Size of consecutive minibatches to take from the dataset.
            To be set according to RAM or GPU memory capacity.
    dataset : `dict`
        The dataset to evaluate the model on.
    get_metric : `Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`
        Function to get the metric from a pair of
        predicted and target value tensors.
    model : `torch.nn.Module`
        The model to evaluate.
    indptr_key : `str`, optional
        If the dataset has sequential entries,
        then this is the key of the index pointer tensor.
        Default: `"indptr"`.
    target_key : `str`, optional
        The key mapped to the target value tensor in the dataset.
        Default: `"target"`
    """
    ensemble_shape = config["ensemble_shape"]
    dataset = {
        key: to_ensembled(ensemble_shape, value)
        for key, value in dataset.items()
    }

    dataset_size = get_dataset_size(config, dataset, indptr_key=indptr_key)
    minibatch_size = config["minibatch_size_eval"]

    minibatch_num = math.ceil(dataset_size / minibatch_size)
    metric = 0
    progress_bar = tqdm.trange(minibatch_num)

    with torch.no_grad():
        for i in progress_bar:
        # for i in range(minibatch_num):
            minibatch_indices = torch.arange(
                i * minibatch_size,
                min((i + 1) * minibatch_size, dataset_size),
                device=config["device"]
            )
            minibatch_indices = minibatch_indices.broadcast_to(
                ensemble_shape + (len(minibatch_indices),)
            )
            minibatch = get_minibatch(
                dataset,
                minibatch_indices,
                indptr_key=indptr_key
            )
            minibatch_predict = model(minibatch)
            minibatch_metric = get_metric(
                minibatch_predict,
                minibatch[target_key]
            )

            metric += minibatch_metric * minibatch_indices.shape[-1]

    progress_bar.close()

    return metric / dataset_size


def get_binary_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    Get the binary accuracy between a label and a logit tensor.
    It can handle arbitrary ensemble shapes.

    Parameters
    ----------
    logits : torch.Tensor
        The logit tensor. We assume it has shape
        `ensemble_shape + (dataset_size, 1)`.
    labels : torch.Tensor
        The tensor of true labels. We assume it has shape
        `(dataset_size, 1)` or `ensemble_shape + (dataset_size, 1)`.

    Returns
    -------
    The tensor of binary accuracies per ensemble member
    of shape `ensemble_shape`.
    """

    logits = logits["features"]
    predict_positives = logits[..., 0] > 0
    true_positives = labels.broadcast_to(
        predict_positives.shape
    ).to(torch.bool)

    return (
        predict_positives == true_positives
    ).to(torch.float32).mean(dim=-1)


def get_binary_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    Get the binary cross-entropy between a label and a logit tensor.
    It can handle arbitrary ensemble shapes.

    Parameters
    ----------
    logits : torch.Tensor
        The logit tensor. We assume it has shape
        `ensemble_shape + (dataset_size,)`.
    labels : torch.Tensor
        The tensor of true labels. We assume it has shape
        `(dataset_size, 1)` or `ensemble_shape + (dataset_size, 1)`.

    Returns
    -------
    The tensor of binary cross-entropies per ensemble member
    of shape `ensemble_shape`.
    """

    logits = logits["features"]

    return F.binary_cross_entropy_with_logits(
        logits[..., 0],
        labels.broadcast_to(logits.shape[:-1]),
        reduction="none"
    ).mean(dim=-1)


def normalize_features(
    train_features: torch.Tensor,
    additional_features=(),
    verbose=False
):
    """
    Normalize feature tensors by
    1. subtracting the total mean of the training features, then
    2. dividing by the total std of the offset training features.

    Optionally, apply the same transformation to additional feature tensors,
    eg. validation and test feature tensors.

    Parameters
    ----------
    train_features : `torch.Tensor`
        Training feature tensor.
    additional_features : `Iterable[torch.Tensor]`, optional
        Iterable of additional features to apply the transformation to.
        Default: `()`.
    verbose : `bool`, optional
        Whether to print the total mean and std
        gotten for the transformation.
    """
    sample_mean = train_features.mean()
    train_features -= sample_mean
    for features in additional_features:
        features -= sample_mean

    sample_std = train_features.std()
    train_features /= sample_std
    for features in additional_features:
        features /= sample_std

    if verbose:
        print(
            "Training feature tensor statistics before normalization:",
            f"mean {sample_mean.cpu().item():.4f}",
            f"std {sample_std.cpu().item():.4f}",
            flush=True
        )