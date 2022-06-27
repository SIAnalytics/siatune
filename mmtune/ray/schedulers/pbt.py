from mmtun.ray.scheduler import SCHEDULERS
from ray.tune.schedulers.pbt import \
    PopulationBasedTraining as _PopulationBasedTraining

from mmtune.ray.spaces import build_space


@SCHEDULERS.register_module(force=True)
class PopulationBasedTraining(_PopulationBasedTraining):

    def __init__(self, *args, **kwargs) -> None:
        hyperparam_mutations = kwargs.get('hyperparam_mutations',
                                          dict()).copy()
        kwargs.update(hyperparam_mutations=build_space(hyperparam_mutations))
