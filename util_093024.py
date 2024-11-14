from collections import defaultdict
from collections.abc import Callable, Generator, Iterable
import datasets
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn.functional as F
import tqdm


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