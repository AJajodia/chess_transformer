import torch
import torch.nn.functional as F
from typing import Optional

class Conv(torch.nn.Module):
    """
    Ensemble-ready, multi-dimensional convolution layer

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
    in_channels : `int`
        The number of input channels.
    kernel_shape : `tuple[int]`
        The kernel shape. The number of entries implicitly
        determines the number of sequence dimensions.
    out_channels : `int`
        The number of output channels.
    bias : `bool`, optional
        Whether to include bias along the output channels.
    init_multiplier : `float`, optional
        We initialize linear maps with Glorot normal initialization,
        that is using the centered normal distribution
        with standard deviation `out_channels ** -.5` times this value.
        Default: `1.`.
    stride : `int | tuple[int]`, optional
        The stride in all directions or per direction.
        Default: `1`.

    Calling
    -------
    Instance calls require one positional argument:
    batch : `dict`
        The input data dictionary. Required key:
        `"features"` : `torch.Tensor`
            Tensor of features, of shape
            `batch_shape + (in_channels,) + sequence_shape` or
            `ensemble_shape + batch_shape + (in_channels,) + sequence_shape`
    """
    def __init__(
        self,
        config: dict,
        in_channels: int,
        kernel_shape: tuple[int],
        out_channels: int,
        bias=True,
        init_multiplier=1.,
        stride=1
    ):
        super().__init__()

        self.ensemble_shape = config["ensemble_shape"]
        self.kernel_shape = kernel_shape
        self.out_channels = out_channels

        kernel_dim = len(kernel_shape)
        self.stride = (stride,) * kernel_dim if hasattr(stride, "__int__") else stride

        self.weight = torch.nn.Parameter(torch.empty(
            self.ensemble_shape + (in_channels, out_channels) + kernel_shape,
            device=config["device"],
            dtype=config["float_dtype"]
        ).normal_(std=out_channels ** -.5) * init_multiplier)

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(
                self.ensemble_shape + (out_channels,),
                device=config["device"],
                dtype=config["float_dtype"]
            ))
        else:
            self.bias = None


    def forward(self, batch: dict) -> dict:
        features: torch.Tensor = batch["features"]
        ensemble_dim = len(self.ensemble_shape)
        kernel_dim = len(self.kernel_shape)

        for kernel_size, stride in zip(
            self.kernel_shape,
            self.stride
        ):
            features = features.unfold(
                -kernel_dim,
                kernel_size,
                stride
            )

        ensemble_prefix = "abcdefgh"[:ensemble_dim]
        kernel_postfix = "ABCDEFGH"[:kernel_dim]
        sequence_infix = "wxyz"[:kernel_dim]
        einsum_formula = f"{ensemble_prefix}...i{sequence_infix}{kernel_postfix},{ensemble_prefix}ij{kernel_postfix}->{ensemble_prefix}...j{sequence_infix}"

        features = torch.einsum(
            einsum_formula,
            features,
            self.weight
        )

        if self.bias is not None:
            bias = self.bias.unflatten(
                -1,
                (1,) * (len(features.shape) - len(self.bias.shape) - kernel_dim) + (self.out_channels,) + (1,) * kernel_dim
            )

            features = features + bias

        return batch | {"features": features}
    

def get_accuracy(
    batch: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Given logits output by a classification model, calculate the accuracy.
    Supports model ensembles of arbitrary ensemble shape.

    Parameters
    ----------
    batch : dict
        The output of a model. It should have a `"features"` key
        mapping to a logit tensor of shape
        `ensemble_shape + (dataset_size, label_num)`.
    labels : torch.Tensor
        Label tensor of shape 
        `(dataset_size,)` or
        `ensemble_shape + (dataset_size,)`.

    Returns
    -------
    The tensor of accuracies of shape `ensemble_shape`.
    """
    batch = batch["features"]

    labels_predict = batch.argmax(dim=-1)
    accuracy = (labels == labels_predict).to(torch.float32).mean(dim=-1)

    return accuracy


def get_cross_entropy(
    batch: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Given logits output by a classification model, 
    calculate the cross-entropy.
    Supports model ensembles of arbitrary ensemble shape.

    Parameters
    ----------
    batch : dict
        The output of a model. It should have a `"features"` key
        mapping to a logit tensor of shape
        `ensemble_shape + (dataset_size, label_num)`.
    labels : torch.Tensor
        Label tensor of shape 
        `(dataset_size,)` or
        `ensemble_shape + (dataset_size,)`.

    Returns
    -------
    The tensor of accuracies of shape `ensemble_shape`.
    """
    batch = batch["features"]

    return F.cross_entropy(
        batch.movedim((-2, -1), (0, 1)),
        labels.broadcast_to(batch.shape[:-1]).movedim(-1, 0),
        reduction="none"
    ).mean(dim=0)
    

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
    batch : `dict`
        The input data dictionary. Required key:
        `"features"` : `torch.Tensor`
            Tensor of features. The feature dimension is determined by
            `feature_dim_index`.
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
        batch: dict
    ) -> dict:
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

        if self.bias is None:
            return features
        
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

        return batch | {"features": features}
    

class Pool(torch.nn.Module):
    """
    Ensemble-ready mean pool operation

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
    kernel_shape : `tuple[int]`, optional
        The kernel shape. The number of entries implicitly
        determines the number of sequence dimensions.
        If given, we pool along the kernel displacements.
    sequence_dim_num: `int`, optional
        If `kernel_shape` is not given, we pool
        along full spatial dimensions of this number.
    stride : `int | tuple[int]`, optional
        The stride in all directions or per direction.
        It is used if `kernel_shape` is given.
        Default: `1`.

    Calling
    -------
    Instance calls require one positional argument:
    batch : `dict`
        The input data dictionary. Required key:
        `"features"` : `torch.Tensor`
            Tensor of features, of shape
            `batch_shape + (in_channels,) + sequence_shape` or
            `ensemble_shape + batch_shape + (in_channels,) + sequence_shape`
    """
    def __init__(
        self,
        config: dict,
        kernel_shape: Optional[tuple[int]] = None,
        sequence_dim_num: Optional[int] = None,
        stride=1
    ):
        super().__init__()

        self.ensemble_shape = config["ensemble_shape"]

        self.kernel_shape = kernel_shape
        self.sequence_dim_num = len(kernel_shape) if sequence_dim_num is None else sequence_dim_num

        self.stride = (stride,) * self.sequence_dim_num if hasattr(stride, "__int__") else stride


    def forward(self, batch: dict) -> dict:
        features: torch.Tensor = batch["features"]

        if self.kernel_shape is not None:
            for kernel_size, stride in zip(
                self.kernel_shape,
                self.stride
            ):
                features = features.unfold(
                    -self.sequence_dim_num,
                    kernel_size,
                    stride
                )

        features = features.mean(
            dim=tuple(range(-self.sequence_dim_num,0))
        )

        return batch | {"features": features}