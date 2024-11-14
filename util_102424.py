from collections import defaultdict
from collections.abc import Callable
import math
import matplotlib.pyplot as plt
import scipy
import torch
import torch.nn.functional as F
import tqdm

from util_091924 import load_preprocessed_dataset
from util_092324 import (
    get_dataloader_random_reshuffle, 
    line_plot_confidence_band
)
from util_102124 import get_mlp

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


def get_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Given logits output by a classification model, calculate the accuracy.
    Supports model ensembles of arbitrary ensemble shape.

    Parameters
    ----------
    logits : torch.Tensor
        Logit tensor of shape
        `ensemble_shape + (dataset_size, label_num)`.
    labels : torch.Tensor
        Label tensor of shape 
        `(dataset_size,)` or
        `ensemble_shape + (dataset_size,)`.

    Returns
    -------
    The tensor of accuracies of shape `ensemble_shape`.
    """
    labels_predict = logits.argmax(dim=-1)
    accuracy = (labels == labels_predict).to(torch.float32).mean(dim=-1)

    return accuracy


def get_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Given logits output by a classification model, 
    calculate the cross-entropy.
    Supports model ensembles of arbitrary ensemble shape.

    Parameters
    ----------
    logits : torch.Tensor
        Logit tensor of shape
        `ensemble_shape + (dataset_size, label_num)`.
    labels : torch.Tensor
        Label tensor of shape 
        `(dataset_size,)` or
        `ensemble_shape + (dataset_size,)`.

    Returns
    -------
    The tensor of accuracies of shape `ensemble_shape`.
    """
    return F.cross_entropy(
        logits.movedim((-2, -1), (0, 1)),
        labels.broadcast_to(logits.shape[:-1]).movedim(-1, 0),
        reduction="none"
    ).mean(dim=0)

def evaluate_model(
    config: dict,
    features: torch.Tensor,
    get_metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    model: torch.nn.Module,
    values: torch.Tensor
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
    features : `torch.Tensor`
        Feature tensor.
    get_metric : `Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`
        Function to get the metric from a pair of
        predicted and target value tensors.
    model : `torch.nn.Module`
        The model to evaluate.
    values : `torch.Tensor`
        Target value tensor.
    """
    dataset_size = len(features)
    minibatch_num = math.ceil(dataset_size / config["minibatch_size_eval"])
    metric = 0
    with torch.no_grad():
        for i in range(minibatch_num):
            minibatch_features, minibatch_values = (
                t[
                    i * config["minibatch_size_eval"]
                   :(i + 1) * config["minibatch_size_eval"]
                ]
                for t in (features, values)
            )
            minibatch_predict = model(minibatch_features)
            minibatch_metric = get_metric(
                minibatch_predict,
                minibatch_values
            )
            minibatch_size = len(minibatch_features)

            metric += minibatch_metric * minibatch_size

    return metric / dataset_size


def welch_one_sided(
    source: torch.Tensor,
    target: torch.Tensor,
    confidence_level=.95
) -> torch.Tensor:
    """
    Performs Welch's t-test with null hypothesis: the expected value
    of the random variable the target tensor collects samples of
    is larger then the expected value
    of the random variable the source tensor collects samples of.

    In the tensors, dimensions after the first 
    are considered batch dimensions.

    Parameters
    ----------
    source : `torch.Tensor`
        Source sample, of shape `(sample_size,) + batch_shape`.
    target : `torch.Tensor`
        Target sample, of shape `(sample_size,) + batch_shape`.
    confidence_level : `float`, optional
        Confidence level of the test. Default: `.95`.
    Returns
    -------
    A Boolean tensor of shape `batch_shape` that is `False`
    where the null hypothesis is rejected.
    """
    sample_num = len(source)
    source_sample_mean, target_sample_mean = (
        t.mean(dim=0)
        for t in (source, target)
    )
    source_sample_var, target_sample_var = (
        t.var(dim=0)
        for t in (source, target)
    )
    var_sum = source_sample_var + target_sample_var

    t = (
        (target_sample_mean - source_sample_mean)
      * (sample_num / var_sum).sqrt()
    )

    nu = (
        var_sum.square()
      * (sample_num - 1)
      / (source_sample_var ** 2 + target_sample_var ** 2)
    )

    p = scipy.stats.t(
        nu.cpu().numpy()
    ).cdf(
        t.cpu().numpy()
    )

    return torch.asarray(
        p > confidence_level,
        device=source.device
    )
