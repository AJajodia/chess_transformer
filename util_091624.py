import datasets
import torch

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


def preprocess_dataset(
    dataset: datasets.Dataset,
    bias=True,
    device="cpu",
    features_dtype=torch.float32,
    features_scale=1 / 255,
    images_name="image",
    labels_name="label"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given an image classification dataset, outputs as `torch.Tensor`s:
    - the image data:
        * flattened to a matrix of shape `(dataset_size, feature_num)`,
        * with an added column of 1s if `bias`,
        * transformed to `features_dtype` and
        * multiplied by `features_scale` and
    - the label data:
        * as a vector of shape `(dataset_size)` and
        * of dtype `torch.int64`.

    Parameters
    ----------
    dataset : datasets.Dataset
        The dataset to preprocess. We assume that the image and label
        feature names are `"image"` and `"label"`, respectively.
    bias : bool, optional
        Whether to add a column of 1s to the feature matrix
        which can be used as feature for the bias. Default: `True`.
    device : torch.device | int | str, optional
        The device to put data tensors to. Default: "cpu"
    features_dtype : torch.dtype, optional
        The floating point datatype to transform the feature matrix to.
        Default: `torch.float32`.
    feature_scale : float, optional
        The value to multiply the feature matrix with.
        Default: `1 / 255`.
    images_name : str, optional
        The name of the dataset entry storing the images. Default: `"image"`
    labels_name : str, optional
        The name of the dataset entry storing the labels. Default: `"label"`

    Returns
    -------
    The pair `feature_matrix, labels`.
    """
    if device is None:
        device = "cpu"
    if images_name is None:
        images_name = "image"
    if labels_name is None:
        labels_name = "label"

    dataset = dataset.with_format("torch", device=device)
    feature_matrix = flatten_images(
        dataset[images_name],
        dtype=features_dtype,
        scale=features_scale
    )
    if bias:
        feature_matrix = add_column_of_1s(feature_matrix)

    labels = dataset[labels_name]

    return feature_matrix, labels