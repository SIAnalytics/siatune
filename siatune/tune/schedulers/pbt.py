# Copyright (c) SI-Analytics. All rights reserved.
import copy
import random
from typing import Callable, Dict, Optional

from ray.tune.experiment import Trial
from ray.tune.schedulers.pbt import \
    PopulationBasedTraining as _PopulationBasedTraining
from ray.tune.search.sample import Domain

from siatune.tune.spaces import build_space
from siatune.utils import ImmutableContainer
from .builder import TRIAL_SCHEDULERS


def explore(
    config: Dict,
    mutations: Dict,
    resample_probability: float,
    custom_explore_fn: Optional[Callable],
) -> Dict:
    """Return a config perturbed as specified.

    Args:
        config: Original hyperparameter configuration.
        mutations: Specification of mutations to perform as documented
            in the PopulationBasedTraining scheduler.
        resample_probability: Probability of allowing resampling of a
            particular variable.
        custom_explore_fn: Custom explore fn applied after built-in
            config perturbations are.
    """
    new_config = copy.deepcopy(config)
    for key, distribution in mutations.items():
        assert isinstance(distribution, Domain)
        if random.random() < resample_probability:
            new_config[key] = ImmutableContainer.decouple(
                distribution.sample(None))

        try:
            new_config[key] = config[key] * 1.2 if random.random(
            ) > 0.5 else config[key] * 0.8
            if isinstance(config[key], int):
                new_config[key] = int(new_config[key])
        except Exception:
            new_config[key] = config[key]
    if custom_explore_fn:
        new_config = custom_explore_fn(new_config)
        assert new_config is not None
    return new_config


@TRIAL_SCHEDULERS.register_module(force=True)
class PopulationBasedTraining(_PopulationBasedTraining):

    def __init__(self, *args, **kwargs) -> None:
        hyperparam_mutations = kwargs.get('hyperparam_mutations',
                                          dict()).copy()
        kwargs.update(hyperparam_mutations=build_space(hyperparam_mutations))
        super().__init__(*args, **kwargs)

    def _get_new_config(self, trial: Trial, trial_to_clone: Trial) -> Trial:
        """Gets new config for trial by exploring trial_to_clone's config.

        Argss:
            trial: The trial to chenge.
            trial_to_clone: The trial to reference.

        Returns:
            Trail: Changed Trial
        """
        return explore(
            trial_to_clone.config,
            self._hyperparam_mutations,
            self._resample_probability,
            self._custom_explore_fn,
        )
