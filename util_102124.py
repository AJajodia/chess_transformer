from collections.abc import (
    Callable,
    Iterable
)
import torch
import torch.nn.functional as F
import tqdm
from typing import Optional

from util_092324 import get_dataloader_random_reshuffle


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


def get_mse(
    predict: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the MSE between two tensors. Compatible with ensembles.

    Parameters
    ----------
    predict : `torch.Tensor`
        Predicted values. The expected shape is
        `ensemble_shape + (batch_size, values_dim)`.
    target : `torch.Tensor`
        Target values. The expected shape is either
        `ensemble_shape + (batch_size, values_dim)` or
        `(batch_size, values_dim)`.

    Returns
    -------
    The tensor of MSE values, of shape `ensemble_shape`.
    """
    # print(predict.shape, target.shape)
    target = target.broadcast_to(predict.shape)
    # print(target.shape)
    mse = F.mse_loss(predict, target, reduction="none")

    return mse.sum(dim=-1).mean(dim=-1)


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
    init_multiplier : `float`, optional
        The weight parameter values are initialized following
        a normal distribution with center 0 and std
        `in_features ** -.5` times this value. Default: `1.`

    Calling
    -------
    Instance calls require one positional argument:
    features : `torch.Tensor`
        The input tensor. It is required to be one of the following shapes:
        1. `ensemble_shape + batch_shape + (in_features,)`
        2. `batch_shape + (in_features,)

        Upon a call, the model thinks we're in the first case
        if the first `len(ensemble_shape)` many entries of the
        shape of the input tensor is `ensemble_shape`.
    """
    def __init__(
        self,
        config: dict,
        in_features: int,
        out_features: int,
        bias=True,
        init_multiplier=1.
    ):
        super().__init__()

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
        features: torch.Tensor
    ) -> torch.Tensor:
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

        return features
    

def train_supervised(
    config: dict,
    get_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    model: torch.nn.Module,
    train_features: torch.Tensor,
    train_values: torch.Tensor,
    valid_features: torch.Tensor,
    valid_values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Train a model on a supervised dataset. Supports ensembles.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Required keys:
        ensemble_shape : tuple[int]
            The shape of the model ensemble.
        learning_rate : float | torch.Tensor
            The learning rate of the SGD optimization.
            If a tensor, then it should have shape
            broadcastable to `ensemble_shape`.
            In that case, the members of the ensemble are trained with
            different learning rates.
        minibatch_size : int
            The minibatch size of the reshuffling dataloader to use.
        steps_num : int
            The number of training steps to take.
        valid_interval : int
            The frequency of evaluations,
            measured in the number of train steps.
    get_loss : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Function that calculates the loss values of an ensemble
        from a predicted and a target value tensor.
    model : torch.nn.Module
        The model to train.
    train_features : torch.Tensor
        Training feature matrix.
    train_values : torch.Tensor
        Training value vector.
    valid_features : torch.Tensor
        Validation feature matrix.
    valid_values : torch.Tensor
        Validation value vector.
    """
    learning_rate = torch.asarray(
        config["learning_rate"],
        device=valid_features.device,
        dtype=valid_features.dtype
    )
    progress_bar = tqdm.trange(config["steps_num"])
    train_dataloader = get_dataloader_random_reshuffle(
        config,
        train_features,
        train_values
    )
    train_losses = []
    valid_losses = []

    for step_id in progress_bar:
        minibatch_features, minibatch_values = next(train_dataloader)
        for parameter in model.parameters():
            parameter.grad = None

        predict = model(minibatch_features)
        loss = get_loss(predict, minibatch_values).sum()
        loss.backward()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter -= learning_rate.reshape(
                    learning_rate.shape
                  + (len(parameter.shape) - len(learning_rate.shape))
                  * (1,)
                ) * parameter.grad
        
        if step_id % config["valid_interval"] == 0:
            with torch.no_grad():
                for features, values, losses in (
                    (train_features, train_values, train_losses),
                    (valid_features, valid_values, valid_losses)
                ):
                    predict = model(features)
                    loss = get_loss(predict, values)
                    losses.append(loss)

    return tuple((
        torch.stack(losses)
        for losses in (train_losses, valid_losses)
    ))