from collections.abc import Generator
import torch


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