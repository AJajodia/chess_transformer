from collections.abc import Iterable
import gymnasium as gym
import moviepy.editor as mpy
import os
import torch
from typing import Optional

from util_102424 import welch_one_sided
from util_093024 import get_seed


def get_discounted_returns(
    config: dict,
    rewards: torch.Tensor
) -> torch.Tensor:
    """
    Given a reward vector `(r_1, r_2, ..., r_T)`
    or a batch of such vectors,
    output the corresponding discounted return vector
    `(g_0, g_1, ..., g_{T-1})` or the batch of such vectors.

    Parameters
    ----------
    config : `dict`
        Configuration dictionary. Required key-value pairs:
        `"device"` : `torch.device | int | str`
            The device the policy model is stored on.
        `"discount"` : `float`
            Discount value.
        `"float_dtype"` : `torch.dtype`
            Floating point datatype 
            used by the parameters of the policy model.
    rewards : torch.Tensor
        Reward tensor.
        The last dimension is viewed as the sequence dimension.

    Returns
    -------
    The discounted return tensor.
    """
    step_num = rewards.shape[-1]

    arange = torch.arange(
        step_num,
        device=config["device"],
        dtype=config["float_dtype"]
    )
    discounts = (config["discount"] ** (arange[:, None] - arange)).tril()

    discounted_returns = rewards @ discounts

    return discounted_returns


def get_episode_data(
    config: dict,
    env: gym.vector.VectorEnv,
    policy: torch.nn.Module,
    gumbel: Optional[torch.distributions.Gumbel] = None
) -> dict:
    """
    Given a vectorized POMDP and a policy model,
    run one episode per environment in parallel
    and output actions, observations and rewards.

    Parameters
    ----------
    config : `dict`
        Configuration dictionary. Required key-value pairs:
        `"device"` : `torch.device | int | str`
            The device the policy model is stored on.
        `"ensemble_shape"` : `tuple[int]`
            Ensemble shape of the policy model.
        `"float_dtype"` : `torch.dtype`
            Floating point datatype 
            used by the parameters of the policy model.
    env : `gym.vector.VectorEnv`
        The vectorized environment.
    policy : `torch.nn.Module`
        The policy model.
    gumbel : `torch.distributions.Gumbel`, optional
        If given, actions are sampled as per their logits,
        using the Gumbel-Max trick. Otherwise, the actions
        with the largest logits are taken.

    Returns
    -------
    A dictionary with keys `"actions"`, `"observations"` and `"rewards"`
    that stores episode data as tensors:
    1. Their first two dimensions are
        1. the number of environments and
        2. the maximum number of steps in the episodes.
    2. The reward where an episode has already ended is 0.
        1. The actions and observations at these positions are irrelevant.
    """
    env_num = env.observation_space.shape[0]

    step_observations = torch.asarray(
        env.reset(seed=get_seed())[0],
        device=config["device"],
        dtype=config["float_dtype"]
    )
    step_ongoing = torch.ones(
        env_num,
        device=config["device"],
        dtype=torch.bool
    )

    episode_actions = []
    episode_observations = [step_observations]
    episode_rewards = []    
    
    while step_ongoing.any():
        logits = get_logits(config, step_observations, policy)
        if gumbel is not None:
            logits += gumbel.sample(logits.shape)

        step_actions = logits.argmax(dim=-1)
        step_observations, step_rewards, truncated, terminal = (
            torch.asarray(
                t,
                device=config["device"],
                dtype=dtype
            )
            for t, dtype in zip(
                env.step(step_actions.cpu().numpy())[:4],
                (config["float_dtype"],) * 2 + (torch.bool,) * 2
            )
        )
        
        for t, l in zip(
            (
                step_actions, step_observations, step_rewards * step_ongoing
            ),
            (
                episode_actions,
                episode_observations,
                episode_rewards,
            )
        ):
            l.append(t)
        
        step_ongoing = step_ongoing & ~(truncated | terminal)
    
    return {
        key: torch.stack(collection, dim=1)
        for key, collection in zip(
            ("actions", "observations", "rewards"),
            (
                episode_actions,
                episode_observations,
                episode_rewards
            )
        )
    }


def get_logits(
    config: dict,
    observation: torch.Tensor,
    policy: torch.nn.Module
) -> torch.Tensor:
    """
    In the context of a POMDP
    with continuous observation space and finite action space,
    given an observation tensor and a policy model,
    output the unnormalized next action logits as per the model.

    Parameters
    ----------
    config : `dict`
        Configuration dictionary. Required key-value pair:
        `"ensemble_shape"` : `tuple[int]`
            Ensemble shape.
    observation : `torch.Tensor`
        A tensor of a single or multiple observations of shape
        `batch_shape + (observation_dim,)` or
        `ensemble_shape + batch_shape + (observation_dim,)`
    policy : `torch.nn.Module`
        A policy model that outputs unnormalized next action logits.

    Returns
    -------
    The tensor of unnormalized next action logits.
    """
    ensemble_shape = config["ensemble_shape"]
    ensemble_dim = len(ensemble_shape)

    ensembled_input = observation.shape[:ensemble_dim] == ensemble_shape
    batch_dim = len(observation.shape) - ensembled_input * ensemble_dim - 1

    if batch_dim == 0:
        observation = observation[..., None, :]

    logits = policy(observation)
    if batch_dim == 0:
        logits = logits[..., 0, :]

    return logits


def make_video(
    config: dict,
    policy: torch.nn.Module,
    ensemble_id: Optional[int] = 0,
    fps: Optional[int] = None,
    video_name="test.mp4",
) -> tuple[float, float, str]:
    """
    Given a POMDP with continuous observation space and finite action space
    and an ensemble of policies as models that output
    unnormalized action logits,
    make a video of an episode following an ensemble member.

    Here, we follow the policy deterministically,
    that is we choose the action with the highest logit.

    Parameters
    ----------
    config : `dict`
        Configuration dictionary. Required key-value pairs:
        `"device"` : `torch.device | int | str`
            The device the policy model is stored on.
        `"discount"` : `float`
            Discount to use when calculating the discounted return.
        `"ensemble_shape"` : `tuple[int]`
            Ensemble shape of the policy model.
        `"env_id"` : `str`
            The ID of the environment in the `gym` registry.
        `"env_kwargs"` : `dict`
            Additional arguments of the environment.
        `"float_dtype"` : `torch.dtype`
            Floating point datatype 
            used by the parameters of the policy model.
        `"videos_directory"` : `str`
            Path to the directory to save the video to.
    policy : `torch.nn.Module`
        The policy model.
    ensemble_id : `int`, optional
        The ID of the ensemble member to follow. Default: 0
    fps : `int`, optional
        Frames per second in the video.
        If not given, we use the default given in the environment,
        at the `"render_fps"` key of its `metadata` attribute.
    video_name : `str`
        The of the video to create.
        Its extension determines the video format. Default: "test.mp4"

    Returns
    -------
    The triple of:
    1. The discounted return.
    2. The undiscounted return.
    3. The path to the video.
    """
    env = gym.make(
        config["env_id"],
        render_mode = "rgb_array",
        **config["env_kwargs"]
    )

    discounted_return = 0
    if fps is None:
        fps = env.metadata["render_fps"]

    frames = []
    observation, _ = env.reset(seed=get_seed())
    step_id = 0
    undiscounted_return = 0

    frames.append(env.render())
    while True:
        logits = get_logits(
            config,
            torch.asarray(
                observation,
                device=config["device"],
                dtype=config["float_dtype"]
            ),
            policy
        )
        action = logits[ensemble_id].argmax().cpu().numpy()

        observation, reward, truncated, terminal, _ = env.step(action)
        discounted_return += config["discount"] ** step_id * reward
        frames.append(env.render())
        undiscounted_return += reward
        if truncated or terminal:
            break

        step_id += 1

    env.close()
    
    # https://stackoverflow.com/a/64796174
    clip = mpy.ImageSequenceClip(
        frames,
        fps=fps
    )
    gif_path = os.path.join(config["videos_directory"], video_name)
    clip.write_videofile(
        gif_path,
        fps=fps
    )

    return discounted_return, undiscounted_return, gif_path


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