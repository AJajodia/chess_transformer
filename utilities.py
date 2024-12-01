import torch
from collections.abc import Generator, Sequence, Iterable, Callable
from collections import defaultdict
import scipy
import matplotlib.pyplot as plt
import torch.nn.functional as F
import datasets
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import tqdm



import os
from PIL import Image as PILImage
import torch
from typing import Optional
import math
import itertools
from abc import (
    ABC,
    abstractmethod
)

def add_column_of_1s(
    matrix: torch.Tensor
) -> torch.Tensor:
    """
    Adds a column of 1s to a matrix.

    Parameters
    ----------
    matrix : torch.Tensor
        A 2-dimensional tensor, that is a matrix.

    Returns
    -------
    A new tensor that is the matrix
    augmented by a column of 1s on the right.
    """
    matrix_aug = torch.concatenate(
        [
            matrix,
            torch.ones_like(matrix[:, :1])
        ],
        dim=1
    )

    return matrix_aug

def load_preprocessed_dataset(
    config: dict
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor]
]:
    """
    Loads a dataset that was saved with `torch.save`.
    We expect that the object that was saved is a dictionary with keys
    `train_features`, `train_labels`
    `valid_features`, `valid_labels`,
    `test_features`, `test_labels`
    storing the appropriate data in tensors.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Required keys:  
        dataset_preprocessed_path : str
            The path where the preprocessed dataset was saved to.
        device : torch.device | int | str
            The device to map the tensors to.
        features_dtype : torch.dtype
            The datatype to convert feature tensors to.

    Returns
    -------
    The triple of pairs
    `(train_features, train_labels),
    (valid_feautres, valid_labels),
    (test_features, test_labels)`
    """
    loaded = torch.load(
        config["dataset_preprocessed_path"],
        weights_only=True
    )
    (
        train_features,
        train_labels,
        valid_features,
        valid_labels,
        test_features,
        test_labels
    ) = (
        loaded[key].to(config["device"])
        for key in [
            "train_features",
            "train_labels",
            "valid_features",
            "valid_labels",
            "test_features",
            "test_labels"
        ]
    )
    train_features, valid_features, test_features = (
        features.to(config["features_dtype"])
        for features in (train_features, valid_features, test_features)
    )

    return (
        (train_features, train_labels),
        (valid_features, valid_labels),
        (test_features, test_labels)
    )


def flatten_images(
    images: torch.Tensor,
    dtype=torch.float32,
    scale=1/255
) -> torch.Tensor:
    """
    Given as input a batch of images of shape
    `(batch_size, channel_num, height, width)`
    flatten it and output a tensor of shape `(batch_size, feature_dim)`.

    Moreover:
    1. transform the tensor to `dtype` and
    2. multiply it by `scale`.

    Parameters
    ----------
    images : torch.Tensor
        The images in `torch.Tensor` format.
    dtype : torch.dtype, optional
        The dtype to transform the tensor to. Default: `torch.float32`.
    scale : float, optional
        The value to scale the tensor with. Default: `1 / 255`.
    """
    batch_size, channel_num, height, width = images.shape
    feature_dim = channel_num * height * width

    images = (
        images
       .reshape(batch_size, feature_dim)
       .to(dtype)
      * scale
    )

    return images

def load_preprocessed_dataset(
    config: dict
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor]
]:
    """
    Loads a dataset that was saved with `torch.save`.
    We expect that the object that was saved is a dictionary with keys
    `train_features`, `train_labels`
    `valid_features`, `valid_labels`,
    `test_features`, `test_labels`
    storing the appropriate data in tensors.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Required keys:  
        dataset_preprocessed_path : str
            The path where the preprocessed dataset was saved to.
        device : torch.device | int | str
            The device to map the tensors to.
        features_dtype : torch.dtype
            The datatype to convert feature tensors to.

    Returns
    -------
    The triple of pairs
    `(train_features, train_labels),
    (valid_feautres, valid_labels),
    (test_features, test_labels)`
    """
    loaded = torch.load(
        config["dataset_preprocessed_path"],
        weights_only=True
    )
    (
        train_features,
        train_labels,
        valid_features,
        valid_labels,
        test_features,
        test_labels
    ) = (
        loaded[key].to(config["device"])
        for key in [
            "train_features",
            "train_labels",
            "valid_features",
            "valid_labels",
            "test_features",
            "test_labels"
        ]
    )
    train_features, valid_features, test_features = (
        features.to(config["features_dtype"])
        for features in (train_features, valid_features, test_features)
    )

    return (
        (train_features, train_labels),
        (valid_features, valid_labels),
        (test_features, test_labels)
    )

def get_accuracy(
    labels: torch.Tensor,
    logits: torch.Tensor
) -> torch.Tensor:
    """
    Given logits output by a classification model, calculate the accuracy.
    Supports model ensembles of arbitrary ensemble shape.

    Parameters
    ----------
    labels : torch.Tensor
        Label tensor of shape 
        `(dataset_size,)` or
        `ensemble_shape + (dataset_size,)`.
    logits : torch.Tensor
        Logit tensor of shape
        `ensemble_shape + (dataset_size, label_num)`.

    Returns
    -------
    The tensor of accuracies of shape `ensemble_shape`.
    """
    labels_predict = logits.argmax(dim=-1)
    accuracy = (labels == labels_predict).to(torch.float32).mean(dim=-1)

    return accuracy


def get_cross_entropy(
    labels: torch.Tensor,
    logits: torch.Tensor
) -> torch.Tensor:
    """
    Given logits output by a classification model, 
    calculate the cross-entropy.
    Supports model ensembles of arbitrary ensemble shape.

    Parameters
    ----------
    labels : torch.Tensor
        Label tensor of shape 
        `(dataset_size,)` or
        `ensemble_shape + (dataset_size,)`.
    logits : torch.Tensor
        Logit tensor of shape
        `ensemble_shape + (dataset_size, label_num)`.

    Returns
    -------
    The tensor of accuracies of shape `ensemble_shape`.
    """
    return F.cross_entropy(
        logits.movedim((-2, -1), (0, 1)),
        labels.broadcast_to(logits.shape[:-1]).movedim(-1, 0),
        reduction="none"
    ).mean(dim=0)


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


def line_plot_confidence_band(
    x: Sequence,
    y: torch.Tensor,
    color=None,
    confidence_level=.95,
    label="",
    opacity=.2
):
    """
    Plot training curves from an ensemble with a pointwise confidence band.

    Parameters
    ----------
    x : Sequence
        The sequence of time indicators (eg. number of train steps)
        when the measurements took place.
    y : torch.Tensor
        The tensor of measurements of shape `(len(x), ensemble_num)`.
    color : str | tuple[float] | None, optional
        The color of the plot. Default: `None`
    confidence_level : float, optional
        The confidence level of the confidence band. Default: 0.95
    label : str, optional
        The label of the plot. Default: ""
    opacity : float, optional
        The opacity of the confidence band, to be set via the
        `alpha` keyword argument of `plt.fill_between`. Default: 0.2
    """
    sample_size = y.shape[1]
    student_coefficient = -scipy.stats.t(sample_size - 1).ppf(
        2 * (1 - confidence_level)
    )
    y_mean = y.mean(dim=-1)
    y_std = y.std(dim=-1)
    
    interval_half_length = student_coefficient * y_std / sample_size ** .5
    y_low = y_mean - interval_half_length
    y_high = y_mean + interval_half_length

    plt.fill_between(x, y_low, y_high, alpha=opacity, color=color)
    plt.plot(x, y_mean, color=color, label=label)


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
   
def get_binary_accuracy(
    labels: torch.Tensor,
    logits: torch.Tensor
) -> torch.Tensor:
    """
    Get the binary accuracy between a label and a logit tensor.
    It can handle arbitrary ensemble shapes.

    Parameters
    ----------
    labels : torch.Tensor
        The tensor of true labels. We assume it has shape
        `(dataset_size,)` or `ensemble_shape + (dataset_size,)`.
    logits : torch.Tensor
        The logit tensor. We assume it has shape
        `ensemble_shape + (dataset_size, 1)`.

    Returns
    -------
    The tensor of binary accuracies per ensemble member
    of shape `ensemble_shape`.
    """
    predict_positives = logits[..., 0] > 0
    true_positives = labels.broadcast_to(
        predict_positives.shape
    ).to(torch.bool)

    return (
        predict_positives == true_positives
    ).to(torch.float32).mean(dim=-1)


def get_seed(
    upper=1 << 31
) -> int:
    """
    Generates a random integer by the `torch` PRNG,
    to be used as seed in a stochastic function.

    Parameters
    ----------
    upper : int, optional
        Exclusive upper bound of the interval to generate integers from.
        Default: 1 << 31.

    Returns
    -------
    A random integer.
    """
    return int(torch.randint(upper, size=()))

def lsa(
    config: dict,
    training_dataset: datasets.Dataset,
    validation_datasets: Iterable[datasets.Dataset] = ()
) -> Generator[tuple[torch.Tensor, torch.Tensor]]:
    """
    Fit a composite of a `TfidfVectorizer` and a `TruncatedSVD`
    on the corpus at the `"text"` key of the training dataset.
    Then use this composite to transform the training corpus
    and the optional validation corpora to feature matrices.
    Also returns the labels in the datasets as tensors.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Required keys:
        "device" : torch.device
            The device to store feature matrices and label vectors on.
        "features_dtype" : torch.dtype
            The datatype of feature matrices.
        "labels_dtype" : torch.dtype
            The datatype of label vectors.
        "n_components": int
            The number of dimensions to reduce the feature dimensions to
            with truncated SVD.
    training_dataset : datasets.Dataset
        The training dataset. Required keys:
        "text" : Iterable[str]
            The dataset corpus
        "label" : Iterable[int]
            The dataset labels
    validation_datasets : Iterable[datasets.Dataset], optional
        An iterable of additional datasets,
        of the same structure as `training_dataset`.
        Default: `()`.

    Returns
    -------
    A generator of pairs of feature matrices and label vectors.
    The first pair is the training data.
    Then the optional validation data follows.
    """
    tf_idf = TfidfVectorizer()
    train_features = tf_idf.fit_transform(training_dataset["text"])

    truncated_svd = TruncatedSVD(
        n_components=config["n_components"],
        random_state=get_seed()
    )
    train_features = truncated_svd.fit_transform(train_features)

    train_features = torch.asarray(
        train_features,
        device=config["device"],
        dtype=config["features_dtype"]
    )
    train_labels = training_dataset.with_format(
        "torch",
        device=config["device"]
    )["label"].to(config["labels_dtype"])
    
    yield train_features, train_labels

    for validation_dataset in validation_datasets:
        valid_features = tf_idf.transform(validation_dataset["text"])
        valid_features = truncated_svd.transform(valid_features)
        valid_features = torch.asarray(
            valid_features,
            device=config["device"],
            dtype=config["features_dtype"]
        )
        
        valid_labels = validation_dataset.with_format(
            "torch",
            device=config["device"]
        )["label"].to(config["labels_dtype"])

        yield (valid_features, valid_labels)


def train_logistic_regression(
    config: dict,
    get_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    get_metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    label_num: int,
    train_dataloader: Generator[tuple[torch.Tensor, torch.Tensor]],
    valid_features: torch.Tensor,
    valid_labels: torch.Tensor,
    loss_name="loss",
    metric_name="metric"
) -> dict:
    """
    Train a logistic regression model on a classification task.
    Support model ensembles of arbitrary shape.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Required keys:
        ensemble_shape : tuple[int]
            The shape of the model ensemble.
        improvement_threshold : float
            Making the best validation score this much better
            counts as an improvement.
        learning_rate : float | torch.Tensor
            The learning rate of the SGD optimization.
            If a tensor, then it should have shape
            broadcastable to `ensemble_shape`.
            In that case, the members of the ensemble are trained with
            different learning rates.
        steps_num : int
            The maximum number of training steps to take.
        steps_without_improvement : int
            The maximum number of training steps without improvement to take.
        valid_interval : int
            The frequency of evaluations,
            measured in the number of train steps.
    get_loss : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Function that calculates the loss values of an ensemble
        from a label tensor and a logit tensor.
    get_metric : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Function that calculates the metric values of an ensemble
        from a label tensor and a logit tensor.
    label_num : int
        The number of distinct labels in the classification task.
    train_dataloader : Generator[tuple[torch.Tensor, torch.Tensor]]
        A training minibatch dataloader, that yields pairs of
        feature and label tensors indefinitely.
        We assume that these have shape
        `ensemble_shape + (minibatch_size, feature_dim)`
        and `ensemble_shape + (minibatch_size,)`
        respectively.
    valid_features : torch.Tensor
        Validation feature matrix.
    valid_labels : torch.Tensor
        Validation label vector.
    loss_name : str, optional
        The name of the loss values in the output dictionary.
        Default: "loss"
    metric_name : str, optional
        The name of the metric values in the output dictionary.
        Default: "metric"

    Returns
    -------
    An output dictionary with the following keys:
        best scores : torch.Tensor
            The best validation accuracy per each ensemble member
        best weights : torch.Tensor
            The logistic regression weights
            that were the best per each ensemble member.
        training {metric_name} : torch.Tensor
            The tensor of training metric values, of shape
            `(evaluation_num,) + ensemble_shape`.
        training {loss_name} : torch.Tensor
            The tensor of training loss values, of shape
            `(evaluation_num,) + ensemble_shape`.
        training steps : list[int]
            The list of the number of training steps at each evaluation.
        validation {metric_name} : torch.Tensor
            The tensor of validation metric values, of shape
            `(evaluation_num,) + ensemble_shape`.
        validation {loss_name} : torch.Tensor
            The tensor of validation loss values, of shape
            `(evaluation_num,) + ensemble_shape`.
    """
    device = valid_features.device
    features_dtype = valid_features.dtype
    output = defaultdict(list)

    best_scores = torch.zeros(
        config["ensemble_shape"],
        device=device,
        dtype=features_dtype
    ).log()
    steps_without_improvement = 0

    if isinstance(config["learning_rate"], torch.Tensor):
        learning_rate = config["learning_rate"][..., None, None]
    else:
        learning_rate = config["learning_rate"]

    train_accuracies_step = torch.zeros(
        config["ensemble_shape"],
        device=device,
        dtype=features_dtype
    )
    train_entries = 0
    train_losses_step = torch.zeros(
        config["ensemble_shape"],
        device=device,
        dtype=features_dtype
    )

    progress_bar = tqdm.trange(config["steps_num"])
    step_id = 0
    weights = torch.zeros(
        config["ensemble_shape"] + (valid_features.shape[1], label_num),
        device=device,
        dtype=features_dtype,
        requires_grad=True
    )

    best_weights = torch.empty_like(weights, requires_grad=False)

    for minibatch_features, minibatch_labels in train_dataloader:
        minibatch_size = minibatch_labels.shape[-1]
        weights.grad = None
        logits = minibatch_features @ weights
        train_accuracies_step += get_metric(
            minibatch_labels,
            logits.detach()
        ) * minibatch_size
        loss = get_loss(
            minibatch_labels,
            logits
        )
        loss.sum().backward()
        with torch.no_grad():
            weights -= learning_rate * weights.grad

        train_losses_step += loss.detach() * minibatch_size
        train_entries += minibatch_size

        progress_bar.update()
        step_id += 1
        if step_id % config["valid_interval"] == 0:
            with torch.no_grad():
                logits = valid_features @ weights

            valid_accuracy = get_metric(
                valid_labels,
                logits
            )

            valid_loss = get_loss(
                valid_labels,
                logits
            )

            output[f"training {metric_name}"].append(
                (train_accuracies_step / train_entries)
            )
            output[f"training {loss_name}"].append(
                (train_losses_step / train_entries)
            )
            output["training steps"].append(step_id)
            output[f"validation {metric_name}"].append(valid_accuracy)
            output[f"validation {loss_name}"].append(valid_loss)

            train_accuracies_step.zero_()
            train_entries = 0
            train_losses_step.zero_()

            improvement = valid_accuracy - best_scores
            improvement_mask = improvement > config["improvement_threshold"]

            if improvement_mask.any():
                best_scores[improvement_mask] \
                    = valid_accuracy[improvement_mask]
                best_weights[improvement_mask] = weights[improvement_mask]
                steps_without_improvement = 0
            else:
                steps_without_improvement += config["valid_interval"]

            if (
                step_id >= config["steps_num"]
             or (
                    steps_without_improvement
                 >= config["steps_without_improvement"]
                )  
            ):
                for key in (
                    f"training {metric_name}",
                    f"training {loss_name}",
                    f"validation {metric_name}",
                    f"validation {loss_name}"
                ):
                    output[key] = torch.stack(output[key]).cpu()

                output["best scores"] = best_scores
                output["best weights"] = best_weights
                progress_bar.close()

                return output
            
def get_binary_cross_entropy(
    labels: torch.Tensor,
    logits: torch.Tensor
) -> torch.Tensor:
    """
    Get the binary cross-entropy between a label and a logit tensor.
    It can handle arbitrary ensemble shapes.

    Parameters
    ----------
    labels : torch.Tensor
        The tensor of true labels. We assume it has shape
        `(dataset_size,)` or `ensemble_shape + (dataset_size, 1)`.
    logits : torch.Tensor
        The logit tensor. We assume it has shape
        `ensemble_shape + (dataset_size,)`.

    Returns
    -------
    The tensor of binary cross-entropies per ensemble member
    of shape `ensemble_shape`.
    """

    return F.binary_cross_entropy_with_logits(
        logits[..., 0],
        labels.broadcast_to(logits.shape[:-1]),
        reduction="none"
    ).mean(dim=-1)

import torch
from collections.abc import Generator, Sequence, Iterable, Callable
import scipy
import matplotlib.pyplot as plt
import torch.nn.functional as F
import datasets
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional

def add_column_of_1s(
    matrix: torch.Tensor
) -> torch.Tensor:
    """
    Adds a column of 1s to a matrix.

    Parameters
    ----------
    matrix : torch.Tensor
        A 2-dimensional tensor, that is a matrix.

    Returns
    -------
    A new tensor that is the matrix
    augmented by a column of 1s on the right.
    """
    matrix_aug = torch.concatenate(
        [
            matrix,
            torch.ones_like(matrix[:, :1])
        ],
        dim=1
    )

    return matrix_aug

def load_preprocessed_dataset(
    config: dict
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor]
]:
    """
    Loads a dataset that was saved with `torch.save`.
    We expect that the object that was saved is a dictionary with keys
    `train_features`, `train_labels`
    `valid_features`, `valid_labels`,
    `test_features`, `test_labels`
    storing the appropriate data in tensors.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Required keys:  
        dataset_preprocessed_path : str
            The path where the preprocessed dataset was saved to.
        device : torch.device | int | str
            The device to map the tensors to.
        features_dtype : torch.dtype
            The datatype to convert feature tensors to.

    Returns
    -------
    The triple of pairs
    `(train_features, train_labels),
    (valid_feautres, valid_labels),
    (test_features, test_labels)`
    """
    loaded = torch.load(
        config["dataset_preprocessed_path"],
        weights_only=True
    )
    (
        train_features,
        train_labels,
        valid_features,
        valid_labels,
        test_features,
        test_labels
    ) = (
        loaded[key].to(config["device"])
        for key in [
            "train_features",
            "train_labels",
            "valid_features",
            "valid_labels",
            "test_features",
            "test_labels"
        ]
    )
    train_features, valid_features, test_features = (
        features.to(config["features_dtype"])
        for features in (train_features, valid_features, test_features)
    )

    return (
        (train_features, train_labels),
        (valid_features, valid_labels),
        (test_features, test_labels)
    )


def flatten_images(
    images: torch.Tensor,
    dtype=torch.float32,
    scale=1/255
) -> torch.Tensor:
    """
    Given as input a batch of images of shape
    `(batch_size, channel_num, height, width)`
    flatten it and output a tensor of shape `(batch_size, feature_dim)`.

    Moreover:
    1. transform the tensor to `dtype` and
    2. multiply it by `scale`.

    Parameters
    ----------
    images : torch.Tensor
        The images in `torch.Tensor` format.
    dtype : torch.dtype, optional
        The dtype to transform the tensor to. Default: `torch.float32`.
    scale : float, optional
        The value to scale the tensor with. Default: `1 / 255`.
    """
    batch_size, channel_num, height, width = images.shape
    feature_dim = channel_num * height * width

    images = (
        images
       .reshape(batch_size, feature_dim)
       .to(dtype)
      * scale
    )

    return images

def load_preprocessed_dataset(
    config: dict
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor]
]:
    """
    Loads a dataset that was saved with `torch.save`.
    We expect that the object that was saved is a dictionary with keys
    `train_features`, `train_labels`
    `valid_features`, `valid_labels`,
    `test_features`, `test_labels`
    storing the appropriate data in tensors.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Required keys:  
        dataset_preprocessed_path : str
            The path where the preprocessed dataset was saved to.
        device : torch.device | int | str
            The device to map the tensors to.
        features_dtype : torch.dtype
            The datatype to convert feature tensors to.

    Returns
    -------
    The triple of pairs
    `(train_features, train_labels),
    (valid_feautres, valid_labels),
    (test_features, test_labels)`
    """
    loaded = torch.load(
        config["dataset_preprocessed_path"],
        weights_only=True
    )
    (
        train_features,
        train_labels,
        valid_features,
        valid_labels,
        test_features,
        test_labels
    ) = (
        loaded[key].to(config["device"])
        for key in [
            "train_features",
            "train_labels",
            "valid_features",
            "valid_labels",
            "test_features",
            "test_labels"
        ]
    )
    train_features, valid_features, test_features = (
        features.to(config["features_dtype"])
        for features in (train_features, valid_features, test_features)
    )

    return (
        (train_features, train_labels),
        (valid_features, valid_labels),
        (test_features, test_labels)
    )

def get_accuracy(
    labels: torch.Tensor,
    logits: torch.Tensor
) -> torch.Tensor:
    """
    Given logits output by a classification model, calculate the accuracy.
    Supports model ensembles of arbitrary ensemble shape.

    Parameters
    ----------
    labels : torch.Tensor
        Label tensor of shape 
        `(dataset_size,)` or
        `ensemble_shape + (dataset_size,)`.
    logits : torch.Tensor
        Logit tensor of shape
        `ensemble_shape + (dataset_size, label_num)`.

    Returns
    -------
    The tensor of accuracies of shape `ensemble_shape`.
    """
    labels_predict = logits.argmax(dim=-1)
    accuracy = (labels == labels_predict).to(torch.float32).mean(dim=-1)

    return accuracy


def get_cross_entropy(
    labels: torch.Tensor,
    logits: torch.Tensor
) -> torch.Tensor:
    """
    Given logits output by a classification model, 
    calculate the cross-entropy.
    Supports model ensembles of arbitrary ensemble shape.

    Parameters
    ----------
    labels : torch.Tensor
        Label tensor of shape 
        `(dataset_size,)` or
        `ensemble_shape + (dataset_size,)`.
    logits : torch.Tensor
        Logit tensor of shape
        `ensemble_shape + (dataset_size, label_num)`.

    Returns
    -------
    The tensor of accuracies of shape `ensemble_shape`.
    """
    return F.cross_entropy(
        logits.movedim((-2, -1), (0, 1)),
        labels.broadcast_to(logits.shape[:-1]).movedim(-1, 0),
        reduction="none"
    ).mean(dim=0)


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


def line_plot_confidence_band(
    x: Sequence,
    y: torch.Tensor,
    color=None,
    confidence_level=.95,
    label="",
    opacity=.2
):
    """
    Plot training curves from an ensemble with a pointwise confidence band.

    Parameters
    ----------
    x : Sequence
        The sequence of time indicators (eg. number of train steps)
        when the measurements took place.
    y : torch.Tensor
        The tensor of measurements of shape `(len(x), ensemble_num)`.
    color : str | tuple[float] | None, optional
        The color of the plot. Default: `None`
    confidence_level : float, optional
        The confidence level of the confidence band. Default: 0.95
    label : str, optional
        The label of the plot. Default: ""
    opacity : float, optional
        The opacity of the confidence band, to be set via the
        `alpha` keyword argument of `plt.fill_between`. Default: 0.2
    """
    sample_size = y.shape[1]
    student_coefficient = -scipy.stats.t(sample_size - 1).ppf(
        2 * (1 - confidence_level)
    )
    y_mean = y.mean(dim=-1)
    y_std = y.std(dim=-1)
    
    interval_half_length = student_coefficient * y_std / sample_size ** .5
    y_low = y_mean - interval_half_length
    y_high = y_mean + interval_half_length

    plt.fill_between(x, y_low, y_high, alpha=opacity, color=color)
    plt.plot(x, y_mean, color=color, label=label)


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
   
def get_binary_accuracy(
    labels: torch.Tensor,
    logits: torch.Tensor
) -> torch.Tensor:
    """
    Get the binary accuracy between a label and a logit tensor.
    It can handle arbitrary ensemble shapes.

    Parameters
    ----------
    labels : torch.Tensor
        The tensor of true labels. We assume it has shape
        `(dataset_size,)` or `ensemble_shape + (dataset_size,)`.
    logits : torch.Tensor
        The logit tensor. We assume it has shape
        `ensemble_shape + (dataset_size, 1)`.

    Returns
    -------
    The tensor of binary accuracies per ensemble member
    of shape `ensemble_shape`.
    """
    predict_positives = logits[..., 0] > 0
    true_positives = labels.broadcast_to(
        predict_positives.shape
    ).to(torch.bool)

    return (
        predict_positives == true_positives
    ).to(torch.float32).mean(dim=-1)


def get_seed(
    upper=1 << 31
) -> int:
    """
    Generates a random integer by the `torch` PRNG,
    to be used as seed in a stochastic function.

    Parameters
    ----------
    upper : int, optional
        Exclusive upper bound of the interval to generate integers from.
        Default: 1 << 31.

    Returns
    -------
    A random integer.
    """
    return int(torch.randint(upper, size=()))

def lsa(
    config: dict,
    training_dataset: datasets.Dataset,
    validation_datasets: Iterable[datasets.Dataset] = ()
) -> Generator[tuple[torch.Tensor, torch.Tensor]]:
    """
    Fit a composite of a `TfidfVectorizer` and a `TruncatedSVD`
    on the corpus at the `"text"` key of the training dataset.
    Then use this composite to transform the training corpus
    and the optional validation corpora to feature matrices.
    Also returns the labels in the datasets as tensors.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Required keys:
        "device" : torch.device
            The device to store feature matrices and label vectors on.
        "features_dtype" : torch.dtype
            The datatype of feature matrices.
        "labels_dtype" : torch.dtype
            The datatype of label vectors.
        "n_components": int
            The number of dimensions to reduce the feature dimensions to
            with truncated SVD.
    training_dataset : datasets.Dataset
        The training dataset. Required keys:
        "text" : Iterable[str]
            The dataset corpus
        "label" : Iterable[int]
            The dataset labels
    validation_datasets : Iterable[datasets.Dataset], optional
        An iterable of additional datasets,
        of the same structure as `training_dataset`.
        Default: `()`.

    Returns
    -------
    A generator of pairs of feature matrices and label vectors.
    The first pair is the training data.
    Then the optional validation data follows.
    """
    tf_idf = TfidfVectorizer()
    train_features = tf_idf.fit_transform(training_dataset["text"])

    truncated_svd = TruncatedSVD(
        n_components=config["n_components"],
        random_state=get_seed()
    )
    train_features = truncated_svd.fit_transform(train_features)

    train_features = torch.asarray(
        train_features,
        device=config["device"],
        dtype=config["features_dtype"]
    )
    train_labels = training_dataset.with_format(
        "torch",
        device=config["device"]
    )["label"].to(config["labels_dtype"])
    
    yield train_features, train_labels

    for validation_dataset in validation_datasets:
        valid_features = tf_idf.transform(validation_dataset["text"])
        valid_features = truncated_svd.transform(valid_features)
        valid_features = torch.asarray(
            valid_features,
            device=config["device"],
            dtype=config["features_dtype"]
        )
        
        valid_labels = validation_dataset.with_format(
            "torch",
            device=config["device"]
        )["label"].to(config["labels_dtype"])

        yield (valid_features, valid_labels)


def train_logistic_regression(
    config: dict,
    get_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    get_metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    label_num: int,
    train_dataloader: Generator[tuple[torch.Tensor, torch.Tensor]],
    valid_features: torch.Tensor,
    valid_labels: torch.Tensor,
    loss_name="loss",
    metric_name="metric"
) -> dict:
    """
    Train a logistic regression model on a classification task.
    Support model ensembles of arbitrary shape.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Required keys:
        ensemble_shape : tuple[int]
            The shape of the model ensemble.
        improvement_threshold : float
            Making the best validation score this much better
            counts as an improvement.
        learning_rate : float | torch.Tensor
            The learning rate of the SGD optimization.
            If a tensor, then it should have shape
            broadcastable to `ensemble_shape`.
            In that case, the members of the ensemble are trained with
            different learning rates.
        steps_num : int
            The maximum number of training steps to take.
        steps_without_improvement : int
            The maximum number of training steps without improvement to take.
        valid_interval : int
            The frequency of evaluations,
            measured in the number of train steps.
    get_loss : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Function that calculates the loss values of an ensemble
        from a label tensor and a logit tensor.
    get_metric : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Function that calculates the metric values of an ensemble
        from a label tensor and a logit tensor.
    label_num : int
        The number of distinct labels in the classification task.
    train_dataloader : Generator[tuple[torch.Tensor, torch.Tensor]]
        A training minibatch dataloader, that yields pairs of
        feature and label tensors indefinitely.
        We assume that these have shape
        `ensemble_shape + (minibatch_size, feature_dim)`
        and `ensemble_shape + (minibatch_size,)`
        respectively.
    valid_features : torch.Tensor
        Validation feature matrix.
    valid_labels : torch.Tensor
        Validation label vector.
    loss_name : str, optional
        The name of the loss values in the output dictionary.
        Default: "loss"
    metric_name : str, optional
        The name of the metric values in the output dictionary.
        Default: "metric"

    Returns
    -------
    An output dictionary with the following keys:
        best scores : torch.Tensor
            The best validation accuracy per each ensemble member
        best weights : torch.Tensor
            The logistic regression weights
            that were the best per each ensemble member.
        training {metric_name} : torch.Tensor
            The tensor of training metric values, of shape
            `(evaluation_num,) + ensemble_shape`.
        training {loss_name} : torch.Tensor
            The tensor of training loss values, of shape
            `(evaluation_num,) + ensemble_shape`.
        training steps : list[int]
            The list of the number of training steps at each evaluation.
        validation {metric_name} : torch.Tensor
            The tensor of validation metric values, of shape
            `(evaluation_num,) + ensemble_shape`.
        validation {loss_name} : torch.Tensor
            The tensor of validation loss values, of shape
            `(evaluation_num,) + ensemble_shape`.
    """
    device = valid_features.device
    features_dtype = valid_features.dtype
    output = defaultdict(list)

    best_scores = torch.zeros(
        config["ensemble_shape"],
        device=device,
        dtype=features_dtype
    ).log()
    steps_without_improvement = 0

    if isinstance(config["learning_rate"], torch.Tensor):
        learning_rate = config["learning_rate"][..., None, None]
    else:
        learning_rate = config["learning_rate"]

    train_accuracies_step = torch.zeros(
        config["ensemble_shape"],
        device=device,
        dtype=features_dtype
    )
    train_entries = 0
    train_losses_step = torch.zeros(
        config["ensemble_shape"],
        device=device,
        dtype=features_dtype
    )

    progress_bar = tqdm.trange(config["steps_num"])
    step_id = 0
    weights = torch.zeros(
        config["ensemble_shape"] + (valid_features.shape[1], label_num),
        device=device,
        dtype=features_dtype,
        requires_grad=True
    )

    best_weights = torch.empty_like(weights, requires_grad=False)

    for minibatch_features, minibatch_labels in train_dataloader:
        minibatch_size = minibatch_labels.shape[-1]
        weights.grad = None
        logits = minibatch_features @ weights
        train_accuracies_step += get_metric(
            minibatch_labels,
            logits.detach()
        ) * minibatch_size
        loss = get_loss(
            minibatch_labels,
            logits
        )
        loss.sum().backward()
        with torch.no_grad():
            weights -= learning_rate * weights.grad

        train_losses_step += loss.detach() * minibatch_size
        train_entries += minibatch_size

        progress_bar.update()
        step_id += 1
        if step_id % config["valid_interval"] == 0:
            with torch.no_grad():
                logits = valid_features @ weights

            valid_accuracy = get_metric(
                valid_labels,
                logits
            )

            valid_loss = get_loss(
                valid_labels,
                logits
            )

            output[f"training {metric_name}"].append(
                (train_accuracies_step / train_entries)
            )
            output[f"training {loss_name}"].append(
                (train_losses_step / train_entries)
            )
            output["training steps"].append(step_id)
            output[f"validation {metric_name}"].append(valid_accuracy)
            output[f"validation {loss_name}"].append(valid_loss)

            train_accuracies_step.zero_()
            train_entries = 0
            train_losses_step.zero_()

            improvement = valid_accuracy - best_scores
            improvement_mask = improvement > config["improvement_threshold"]

            if improvement_mask.any():
                best_scores[improvement_mask] \
                    = valid_accuracy[improvement_mask]
                best_weights[improvement_mask] = weights[improvement_mask]
                steps_without_improvement = 0
            else:
                steps_without_improvement += config["valid_interval"]

            if (
                step_id >= config["steps_num"]
             or (
                    steps_without_improvement
                 >= config["steps_without_improvement"]
                )  
            ):
                for key in (
                    f"training {metric_name}",
                    f"training {loss_name}",
                    f"validation {metric_name}",
                    f"validation {loss_name}"
                ):
                    output[key] = torch.stack(output[key]).cpu()

                output["best scores"] = best_scores
                output["best weights"] = best_weights
                progress_bar.close()

                return output
            
def get_binary_cross_entropy(
    labels: torch.Tensor,
    logits: torch.Tensor
) -> torch.Tensor:
    """
    Get the binary cross-entropy between a label and a logit tensor.
    It can handle arbitrary ensemble shapes.

    Parameters
    ----------
    labels : torch.Tensor
        The tensor of true labels. We assume it has shape
        `(dataset_size,)` or `ensemble_shape + (dataset_size, 1)`.
    logits : torch.Tensor
        The logit tensor. We assume it has shape
        `ensemble_shape + (dataset_size,)`.

    Returns
    -------
    The tensor of binary cross-entropies per ensemble member
    of shape `ensemble_shape`.
    """

    return F.binary_cross_entropy_with_logits(
        logits[..., 0],
        labels.broadcast_to(logits.shape[:-1]),
        reduction="none"
    ).mean(dim=-1)

class Linear(torch.nn.Module):
    """
    Ensemble-ready affine transformation `y = x^T W + b`.

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
    in_features : `int`
        The number of input features
    out_features : `int`
        The number of output features.
    bias : `bool`, optional
        Whether the model should include bias. Default: `True`.
    feature_dim_index: `int`, optional
        The index of the feature dimension. Default: `-1`,
    init_multiplier : `float`, optional
        The weight parameter values are initialized following
        a normal distribution with center 0 and std
        `in_features ** -.5` times this value. Default: `1.`

    Calling
    -------
    Instance calls require one positional argument:
    batch : `dict` | `torch.Tensor`.
        The input data dictionary. 
        1. If it is a `dict`, then there is one required key:
            `"features"` : `torch.Tensor`
                Tensor of features. The feature dimension is determined by
                `feature_dim_index`.

            In this case, the module returns another `dict`,
            with the tensor at `"features"` modified by the
            affine transformation.
        2. If it is a `torch.Tensor`, then the image tensor
            is returned directly.
    """
    def __init__(
        self,
        config: dict,
        in_features: int,
        out_features: int,
        bias=True,
        feature_dim_index=-1,
        init_multiplier=1.
    ):
        super().__init__()

        self.feature_dim_index = feature_dim_index

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(
                config["ensemble_shape"] + (out_features,),
                device=config["device"],
                dtype=config["float_dtype"]
            ))
        else:
            self.bias = None

        self.weight = torch.nn.Parameter(torch.empty(
            config["ensemble_shape"] + (in_features, out_features),
            device=config["device"],
            dtype=config["float_dtype"]
        ).normal_(std=out_features ** -.5) * init_multiplier)


    def forward(
        self,
        batch: dict | torch.Tensor
    ) -> dict | torch.Tensor:
        input_tensor = isinstance(batch, torch.Tensor)
        if input_tensor:
            features = batch
        else:
            features: torch.Tensor = batch["features"]

        features = features.movedim(
            self.feature_dim_index,
            -1
        )

        ensemble_shape = self.weight.shape[:-2]
        ensemble_dim = len(ensemble_shape)
        ensemble_input = features.shape[:ensemble_dim] == ensemble_shape
        batch_dim = len(features.shape) - 1 - ensemble_dim * ensemble_input
        
        # (*e, *b, i) @ (*e, *b[:-1], i, o)
        weight = self.weight.reshape(
            ensemble_shape
          + (1,) * (batch_dim - 1)
          + self.weight.shape[-2:]
        )
        features = features @ weight

        if self.bias is not None:
            # (*e, *b, o) + (*e, *b, o)
            bias = self.bias.reshape(
                ensemble_shape
            + (1,) * batch_dim
            + self.bias.shape[-1:]
            )
            features = features + bias

            features = features.movedim(
                -1,
                self.feature_dim_index
            )

        if input_tensor:
            return features

        return batch | {"features": features}


def get_mlp(
    config: dict,
    in_features: int,
    out_features: int,
    hidden_layer_num: Optional[int] = None,
    hidden_layer_size: Optional[int] = None,
    hidden_layer_sizes: Optional[Iterable[int]] = None,
) -> torch.nn.Sequential:
    """
    Creates an MLP with ReLU activation functions.
    Can create a model ensemble.

    config : `dict`
        Configuration dictionary. Required key-value pairs:
        `"device"` : `str`
            The device to store parameters on.
        `"ensemble_shape"` : `tuple[int]`
            The shape of the ensemble of affine transformations
            the model represents.
        `"float_dtype"` : `torch.dtype`
            The floating point datatype to use for the parameters.
    in_features : `int`
        The number of input features
    out_features : `int`
        The number of output features.
    hidden_layer_num : `int`, optional
        If `hidden_layer_sizes` is not given, we create an MLP with
        `hidden_layer_num` hidden layers of
        `hidden_layer_size` dimensions.
    hidden_layer_size : `int`, optional
        If `hidden_layer_sizes` is not given, we create an MLP with
        `hidden_layer_num` hidden layers of
        `hidden_layer_size` dimensions.
    hidden_layer_sizes: `Iterable[int]`, optional
        If given, each entry gives a hidden layer with the given size.
    """
    if hidden_layer_sizes is None:
        hidden_layer_sizes = (hidden_layer_size,) * hidden_layer_num

    layers = []
    layer_in_size = in_features
    for layer_out_size in hidden_layer_sizes:
        layers.extend([
            Linear(
                config,
                layer_in_size,
                layer_out_size,
                init_multiplier=2 ** .5
            ),
            torch.nn.ReLU()
        ])
        layer_in_size = layer_out_size
    
    layers.append(Linear(
        config,
        layer_in_size,
        out_features
    ))

    return torch.nn.Sequential(*layers)

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

    with torch.no_grad():
        for i in range(minibatch_num):
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

    return metric / dataset_size

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

def load_vectors(config: dict) -> tuple[list[str], dict, torch.Tensor]:
    """
    Load the word vectors from a file.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Required keys:
        device : int | str | torch.device
            The device to store the word tensor on.
        features_dtype : torch.dtype
            The floating point datatype the word tensor should use.
        word_vectors_path : str
            The path to the word vector file.
            We assume the file is a text file with the following structure:
            1. The first line of the file contains the number of words
                in the vocabulary and the size of the vectors.
            2. Each following line contains a word followed by
                its vector components,
                like in the default fastText text format.
                Each component is space separated.

    Returns
    -------
    A triple of the following:
    1. A list of the words in the order they appear in the file.
    2. A dictionary mapping words to their index in the list.
    3. A tensor of the stacked word vectors.
    """
    import io

    id2token = []
    id2vector = []
    token2id = dict()

    fin = io.open(
        config["word_vectors_path"],
        'r',
        encoding='utf-8',
        newline='\n',
        errors='ignore'
    )
    n, d = map(int, fin.readline().split())
    progress_bar = tqdm.tqdm(fin, total=n)
    for line in progress_bar:
        word, *components = line.rstrip().split(' ')
        token2id[word] = len(id2token)
        id2token.append(word)
        id2vector.append(torch.tensor(
            [float(component) for component in components],
            device=config["device"],
            dtype=config["features_dtype"]
        ))

    progress_bar.close()

    id2vector = torch.stack(id2vector)

    if id2vector.shape != (n, d):
        raise ValueError(
            f"The shape of the word tensor should be {(n, d)} but it is {id2vector.shape}"
        )

    return id2token, token2id, id2vector

class Optimizer(ABC):
    """
    Optimizer base class.
    Can optimize model ensembles
    with training defined by hyperparameter ensembles.

    Arguments
    ---------
    parameters : `Iterable[torch.nn.Parameter]`
        An iterable of `torch.nn.Parameter` to track.
        In a simple case of optimizing a single `model: torch.nn.Module`,
        this can be `model.parameters()`.
    config : `dict`, optional
        If given, the `update_config` method is called on it
        to initialize hyperparameters. Default: `None`.

    Class attributes
    ----------------
    keys : `tuple[str]`
        The collection of the hyperparameter keys to track
        in the configuration dictionary.

        We expect the hyperparameter values to be either
        `float` or `torch.Tensor`. In the latter case,
        we expect the shape to be a prefix of the shape of the parameters.
        The hyperparameter shapes are regarded as ensemble shapes.

        Required keys:
        `"learning_rate"`
        `"weight_decay"`

    Instance attributes
    -------------------
    config : `dict`
        The hyperparameter dictionary.
    parameters : `list[torch.nn.Parameter]`
        The list of tracked parameters.
    step_id : `int`
        Train step counter.
    """
    keys=(
        "learning_rate",
        "weight_decay"
    )
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        config=None
    ):
        self.config = dict()
        self.parameters = list(parameters)
        self.step_id = 0

        if config is not None:
            self.update_config(config)


    def get_hyperparameter(
        self,
        key: str,
        parameter: torch.Tensor
    ) -> torch.Tensor:
        """
        Take the hyperparameter with name `key`,
        transform it to `torch.Tensor` with the same
        `device` and `dtype` as `parameter`
        and reshape it to be broadcastable
        to `parameter` by postfixing to its shape
        an appropriate number of dimensions of 1.
        """        
        hyperparameter = torch.asarray(
            self.config[key],
            device=parameter.device,
            dtype=parameter.dtype
        )

        return hyperparameter.reshape(
            hyperparameter.shape
            + (
                len(parameter.shape)
                - len(hyperparameter.shape)
            )
            * (1,)
        )
    

    def get_parameters(self) -> Iterable[torch.Tensor]:
        return iter(self.parameters)


    def step(self):
        """
        Update optimizer state, then apply parameter updates in-place.
        Assumes that backpropagation has already occurred by
        a call to the `backward` method of the loss tensor.
        """
        self.step_id += 1
        with torch.no_grad():
            for i, parameter in enumerate(self.parameters):
                self._update_parameter(parameter, i)


    def update_config(self, config: dict):
        """
        Update hyperparameters by the values in `config: dict`.
        """
        for key in self.keys:
            self.config[key] = config[key]


    def zero_grad(self):
        """
        Make the `grad` attribute of each tracked parameter `None`.
        """
        for parameter in self.parameters:
            parameter.grad = None


    def _apply_parameter_update(
        self,
        parameter: torch.nn.Parameter,
        parameter_update: torch.Tensor
    ):
        parameter += parameter_update


    @abstractmethod
    def _get_parameter_update(
        self,
        parameter: torch.nn.Parameter,
        parameter_id: int
    ) -> torch.Tensor:
        if self.config["weight_decay"] is None:
            return torch.zeros_like(parameter)
        
        return -(
            self.get_hyperparameter("learning_rate", parameter)
          * self.get_hyperparameter("weight_decay", parameter)
          * parameter
        )


    def _update_state(
        self,
        parameter: torch.nn.Parameter,
        parameter_id: int
    ):
        pass


    def _update_parameter(
        self,
        parameter: torch.nn.Parameter,
        parameter_id: int
    ):
        self._update_state(parameter, parameter_id)
        parameter_update = self._get_parameter_update(
            parameter,
            parameter_id
        )
        self._apply_parameter_update(
            parameter,
            parameter_update
        )


class AdamW(Optimizer):
    """
    Adam optimizer with optionally weight decay.
    Can optimize model ensembles
    with training defined by hyperparameter ensembles.

    Arguments
    ---------
    parameters : `Iterable[torch.nn.Parameter]`
        An iterable of `torch.nn.Parameter` to track.
        In a simple case of optimizing a single `model: torch.nn.Module`,
        this can be `model.parameters()`.
    config : `dict`, optional
        If given, the `update_config` method is called on it
        to initialize hyperparameters. Default: `None`.

    Class attributes
    ----------------
    keys : `tuple[str]`
        The collection of the hyperparameter keys to track
        in the configuration dictionary.

        We expect the hyperparameter values to be either
        `float` or `torch.Tensor`. In the latter case,
        we expect the shape to be a prefix of the shape of the parameters.
        The hyperparameter shapes are regarded as ensemble shapes.

        Required keys:
        `"epsilon"`,
        `"first_moment_decay"`,
        `"learning_rate"`
        `"second_moment_decay"`,
        `"weight_decay"`
    """
    keys = (
        "epsilon",
        "first_moment_decay",
        "learning_rate",
        "second_moment_decay",
        "weight_decay"
    )
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        config=None
    ):
        super().__init__(parameters, config)
        self.first_moments = [
            torch.zeros_like(parameter)
            for parameter in self.parameters
        ]
        self.second_moments = [
            torch.zeros_like(parameter)
            for parameter in self.parameters
        ]


    def _get_parameter_update(
        self,
        parameter: torch.nn.Parameter,
        parameter_id: int
    ) -> torch.Tensor:
        parameter_update = super()._get_parameter_update(
            parameter,
            parameter_id
        )

        epsilon = self.get_hyperparameter(
            "epsilon",
            parameter
        )
        first_moment = self.first_moments[parameter_id]
        first_moment_decay = self.get_hyperparameter(
            "first_moment_decay",
            parameter
        )
        learning_rate = self.get_hyperparameter(
            "learning_rate",
            parameter
        )
        second_moment = self.second_moments[parameter_id]
        second_moment_decay = self.get_hyperparameter(
            "second_moment_decay",
            parameter
        )

        first_moment_debiased = (
            first_moment
          / (1 - first_moment_decay ** self.step_id)
        )
        second_moment_debiased = (
            second_moment
          / (1 - second_moment_decay ** self.step_id)
        )        

        parameter_update -= (
            learning_rate
          * first_moment_debiased
          / (
                second_moment_debiased.sqrt()
              + epsilon
            )
        )

        return parameter_update


    def _update_state(
        self,
        parameter: torch.nn.Parameter,
        parameter_id: int
    ):
        if parameter.grad is None:
            return

        first_moment = self.first_moments[parameter_id]
        first_moment_decay = self.get_hyperparameter(
            "first_moment_decay",
            parameter
        )
        second_moment = self.second_moments[parameter_id]
        second_moment_decay = self.get_hyperparameter(
            "second_moment_decay",
            parameter
        )

        first_moment[:] = (
            first_moment_decay
          * first_moment
          + (1 - first_moment_decay)
          * parameter.grad
        )
        second_moment[:] = (
            second_moment_decay
          * second_moment
          + (1 - second_moment_decay)
          * parameter.grad.square()
        )


    def get_parameters(self) -> Iterable[torch.Tensor]:
        return itertools.chain(
            self.parameters,
            self.first_moments,
            self.second_moments
        )

def pbt_init(
    config: dict,
    log: dict
):
    """
    Initializes Population Based Training.

    Parameters
    ----------
    config : `dict`
        Configuration dictionary. Required key-value pairs:
        `"ensemble_shape"` : tuple[int]
            Ensemble shape. We assume this is a 1-dimensional tuple
            with dimensions the population size.
        `"hyperparameter_raw_init_distributions"` : `dict`
            Dictionary that maps tuned hyperparameter names
            to `torch.distributions.Distribution` of raw hyperparameter values.
        `"hyperparameter_transforms"` : `dict`
            Dictionary that maps tuned hyperparameter names
            to transformations of raw hyperparameter values.
    log : `defauldict(list)`
        Training log dictionary.

    Updates
    -------
    For each `key in config["hyperparameter_raw_init_distributions"]`:
    1. It samples raw hyperparameter values
        and updates `config[key + "_raw"]` by them.
    2. It applies `config["hyperparameter_transforms"][key]`
        to the raw hyperparameter values and
        1. updates `config[key]` by them and
        2. appends them to `log[key]`.
    """
    for name, distribution in config[
        "hyperparameter_raw_init_distributions"
    ].items():
        value_raw = distribution.sample(config["ensemble_shape"])
        config[name + "_raw"] = value_raw
        value = config[
            "hyperparameter_transforms"
        ][name](value_raw)
        config[name] = value
        log[name].append(value)


def pbt_update(
    config: dict,
    evaluations: torch.Tensor,
    log: dict,
    parameters: Iterable[torch.nn.Parameter]
):
    """
    Performs a Population Based Training update
    with exploitation determined by one-sided Welch's t-tests.

    Parameters
    ----------
    config : `dict`
        Configuration dictionary. Required key-value pairs:
        `"ensemble_shape"` : tuple[int]
            Ensemble shape. We assume this is a 1-dimensional tuple
            with dimensions the population size.
        `"hyperparameter_raw_perturbs"` : `dict`
            Dictionary that maps tuned hyperparameter names
            to `torch.distributions.Distribution` of additive noise.
        `"hyperparameter_transforms"` : `dict`
            Dictionary that maps tuned hyperparameter names
            to transformations of raw hyperparameter values.
        `"welch_confidence_level"` : `float`
            The confidence level in Welch's t-test
            that is used in determining if a population member
            is to be replaced by another member with perturbed hyperparameters.
        `"welch_sample_size"` : `int`
            The last this many validation metrics are used
            in Welch's t-test.
    evaluations : `torch.Tensor`
        Tensor of evaluations. We assume it has shape
        `(welch_sample_size, population_size)`.
    log : `defauldict(list)`
        Training log dictionary.
    parameters : `Iterable[torch.nn.Parameter]`
        Iterable of parameters to update.

    Updates
    -------
    1. For each population entry, a target index is drawn,
        the index of another population entry to compare evaluations with.
    2. The entries are then compared with the target entries
        via a one-sided Welch's t-test.
    3. We get a mask of population entries
        such that the hypothesis that the corresponding entry at the target
        index has better expected evaluations cannot be rejected.
    4. The indices and masks are appended to the
        `"source mask"` and `"target indices"` lists of `log`.

    5. For each tuned hyperparameter, name `key`:
        we replace the masked entries
        by perturbed corresponding target values:
        to the appropriate values at `config[key + "_raw"]`,
        we add noise sampled from
        `config["hyperparameter_raw_perturbs"][key]`,
        then transform them by
        `config["hyperparameter_transforms"][key]`.

        We update the appropriate values of
        `config[key]` and `config[key + "_raw"]`
        and append the new hyperparameter values to `log[key]`.

    6. For each parameter in `parameters`:
        We replace the masked subtensors by the
        correponding entries at the target indices.
    """
    population_size = config["ensemble_shape"][0]

    target_indices = torch.randint(
        device=evaluations.device,
        high=population_size,
        size=(population_size,)
    )
    source_mask = welch_one_sided(
        evaluations,
        evaluations[:, target_indices],
        confidence_level=config["welch_confidence_level"]
    )
    log["source mask"].append(source_mask)
    log["target indices"].append(target_indices)

    if source_mask.any():
        for parameter in parameters:
            parameter[source_mask] = parameter[
                target_indices[source_mask]
            ]

        for name, transform in config[
            "hyperparameter_transforms"
        ].items():
            value_raw: torch.Tensor = config[
                name + "_raw"
            ]

            additive_noise = config[
                "hyperparameter_raw_perturb"
            ][name].sample(
                (source_mask.sum(),)
            )
            perturbed_values = value_raw[
                target_indices
            ][source_mask] + additive_noise
            value_raw[source_mask] = perturbed_values
            value = transform(value_raw)
            config[name] = value
            log[name].append(value)
            
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

class Dropout(torch.nn.Module):
    """
    Ensemble-ready dropout layer.

    Arguments
    ---------
    config : `dict`
        Configuration dictionary. Required key-value pairs:
        `"dropout_p"` : `torch.Tensor`
            Dropout probability tensor, of shape `ensemble_shape`.
        `"ensemble_shape"` : `tuple[int]`
            The shape of the ensemble of affine transformations
            the model represents.

    Calling
    -------
    Instance calls require one positional argument:
    batch : `dict`
        The input data dictionary. Required key:
        `"features"` : `torch.Tensor`
            Tensor of features.
    """

    def __init__(self, config: dict):
        super().__init__()

        self.config = config


    def forward(self, batch: dict) -> dict:
        if not self.training:
            return batch
        
        ensemble_shape = self.config["ensemble_shape"]
        ensemble_dim = len(ensemble_shape)
        features = batch["features"]
        
        features = to_ensembled(self.config["ensemble_shape"], features)
        dropout_p = self.config["dropout_p"].unflatten(
            -1,
            ensemble_shape + (1,) * (len(features.shape) - ensemble_dim)
        )

        features = features / (1 - dropout_p + 1e-4)

        sample = torch.rand(features.shape, device=features.device)
        mask = sample > dropout_p

        features = features * mask


        return batch | {"features": features}

class LayerNorm(torch.nn.Module):
    """
    Ensemble-ready layer normalization layer

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
    normalized_shape : `int | tuple[int]`
        The part of the shape of the incoming tensors
        that are to be normalized together with batch dimensions.
        We view the following as batch dimensions:
        ```
        range(
            len(ensemble_shape),
            -len(normalized_shape) - normalized_offset
        )
        ```
        If an integer, we view it as a single-element tuple.
    bias : `bool`, optional
        If `elementwise_affine`, whether to include offset
        in the learned transformation. Default: `True`.
    elementwise_affine : `bool`, optional
        Whether to include learnable scale. If this and `bias`,
        then we also include learnable offset. These will be tensors
        of shape `ensemble_shape + normalized_shape` that are
        broadcast to the incoming feature tensors appropriately.
        Default: `True`.
    epsilon : `float`, optional
        Small positive value, to be included in the divisor when we
        divide by the variance, for numerical stability. Default: `1e-5`.
    normalized_offset : `int`, optional
        We get `normalized_shape` out of an incoming feature tensor
        at dimensions
        ```
        range(
            -len(normalized_shape) - normalized_offset,
            -normalized_offset
        )
        ```
        Default: `0`.

    Calling
    -------
    Instance calls require one positional argument:
    batch : `dict`
        The input data dictionary. Required key:
        `"features"` : `torch.Tensor`
            Tensor of features.
    """
    def __init__(
        self,
        config: dict,
        normalized_shape: int | tuple[int],
        bias=True,
        elementwise_affine=True,
        epsilon=1e-5,
        normalized_offset=0
    ):
        super().__init__()

        if hasattr(normalized_shape, "__int__"):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = normalized_shape

        self.ensemble_shape = config["ensemble_shape"]
        self.epsilon = epsilon
        self.normalized_offset = normalized_offset

        if elementwise_affine:
            self.scale = torch.nn.Parameter(torch.ones(
                self.ensemble_shape + self.normalized_shape + (1,) * normalized_offset,
                device=config["device"],
                dtype=config["float_dtype"]
            ))
            if bias:
                self.bias = torch.nn.Parameter(torch.zeros_like(self.scale))
            else:
                self.bias = None

        else:
            self.bias, self.scale = None, None


    def forward(self, batch: dict) -> dict:
        features: torch.Tensor = batch["features"]

        ensemble_dim = len(self.ensemble_shape)
        features = to_ensembled(self.ensemble_shape, features)

        normalized_dim = len(self.normalized_shape)

        batch_dim = len(features.shape) - ensemble_dim - normalized_dim - self.normalized_offset
        normalized_range = tuple(range(
            ensemble_dim,
            ensemble_dim + batch_dim
        )) + tuple(range(
            -normalized_dim - self.normalized_offset,
            -self.normalized_offset
        ))

        features = features - features.mean(dim=normalized_range, keepdim=True)
        features = features / features.std(dim=normalized_range, keepdim=True)

        if self.scale is not None:
            scale = self.scale.unflatten(
                ensemble_dim,
                (1,) * batch_dim + self.normalized_shape[:1]
            )

            features = features * scale

            if self.bias is not None:
                bias = self.bias.unflatten(
                    ensemble_dim,
                    (1,) * batch_dim + self.normalized_shape[:1]
                )
                features = features + bias

        return batch | {"features": features}
    
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
    
def update_model(
    config: dict,
    model: torch.nn.Module
):
    """
    Update the configuration dictionary of a model.
    We iterate over its submodules and whichever has a `config` attribute,
    we update it by the included `config` dictionary.

    Parameters
    ----------
    config : `dict`
        The updated configuration dictionary.
    model : `torch.nn.Module`
        The model to update.
    """
    for module in model.modules():
        if hasattr(module, "config"):
            module.config.update(config)
            
class AdamW(Optimizer):
    """
    Adam optimizer with optionally weight decay.
    Can optimize model ensembles
    with training defined by hyperparameter ensembles.

    Arguments
    ---------
    parameters : `Iterable[torch.nn.Parameter]`
        An iterable of `torch.nn.Parameter` to track.
        In a simple case of optimizing a single `model: torch.nn.Module`,
        this can be `model.parameters()`.
    config : `dict`, optional
        If given, the `update_config` method is called on it
        to initialize hyperparameters. Default: `None`.

    Class attributes
    ----------------
    keys : `tuple[str]`
        The collection of the hyperparameter keys to track
        in the configuration dictionary.

        We expect the hyperparameter values to be either
        `float` or `torch.Tensor`. In the latter case,
        we expect the shape to be a prefix of the shape of the parameters.
        The hyperparameter shapes are regarded as ensemble shapes.

        Required keys:
        `"epsilon"`,
        `"first_moment_decay"`,
        `"learning_rate"`
        `"second_moment_decay"`,
        `"weight_decay"`
    """
    keys = (
        "epsilon",
        "first_moment_decay",
        "learning_rate",
        "second_moment_decay",
        "weight_decay"
    )
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        config=None
    ):
        super().__init__(parameters, config)
        self.first_moments = [
            torch.zeros_like(parameter)
            for parameter in self.parameters
        ]
        self.second_moments = [
            torch.zeros_like(parameter)
            for parameter in self.parameters
        ]


    def _get_parameter_update(
        self,
        parameter: torch.nn.Parameter,
        parameter_id: int
    ) -> torch.Tensor:
        parameter_update = super()._get_parameter_update(
            parameter,
            parameter_id
        )

        epsilon = self.get_hyperparameter(
            "epsilon",
            parameter
        )
        first_moment = self.first_moments[parameter_id]
        first_moment_decay = self.get_hyperparameter(
            "first_moment_decay",
            parameter
        )
        learning_rate = self.get_hyperparameter(
            "learning_rate",
            parameter
        )
        second_moment = self.second_moments[parameter_id]
        second_moment_decay = self.get_hyperparameter(
            "second_moment_decay",
            parameter
        )

        first_moment_debiased = (
            first_moment
          / (1 - first_moment_decay ** self.step_id)
        )
        second_moment_debiased = (
            second_moment
          / (1 - second_moment_decay ** self.step_id)
        )        

        parameter_update -= (
            learning_rate
          * first_moment_debiased
          / (
                second_moment_debiased.sqrt()
              + epsilon
            )
        )

        return parameter_update


    def _update_state(
        self,
        parameter: torch.nn.Parameter,
        parameter_id: int
    ):
        if parameter.grad is None:
            return

        first_moment = self.first_moments[parameter_id]
        first_moment_decay = self.get_hyperparameter(
            "first_moment_decay",
            parameter
        )
        second_moment = self.second_moments[parameter_id]
        second_moment_decay = self.get_hyperparameter(
            "second_moment_decay",
            parameter
        )

        first_moment[:] = (
            first_moment_decay
          * first_moment
          + (1 - first_moment_decay)
          * parameter.grad
        )
        second_moment[:] = (
            second_moment_decay
          * second_moment
          + (1 - second_moment_decay)
          * parameter.grad.square()
        )


    def get_parameters(self) -> Iterable[torch.Tensor]:
        return itertools.chain(
            self.parameters,
            self.first_moments,
            self.second_moments
        )

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

def get_array_minibatch(
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