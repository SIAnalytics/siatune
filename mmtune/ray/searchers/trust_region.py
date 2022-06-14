import math
import operator
import pickle
import random
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional

import gpytorch
import numpy as np
import torch
from gpytorch.constraints.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from ray.tune.sample import Categorical, Domain, Integer, Normal, Quantized
from ray.tune.suggest import Searcher
from ray.tune.suggest.suggestion import (UNDEFINED_METRIC_MODE,
                                         UNDEFINED_SEARCH_SPACE)
from ray.tune.suggest.variant_generator import parse_spec_vars
from ray.tune.utils.util import flatten_dict, unflatten_dict
from torch.quasirandom import SobolEngine

from .builder import SEARCHERS


class _Optimizer:
    """A wrapper class for gp based trust region optimization."""

    @dataclass(frozen=True)
    class ParamMeta:
        """A data class responsible for mapping parameter information into a
        vector space.

        idx (int): The index of the parameter. category (Optional[List]): The
        category of the parameter. lower (float): The lower bound of the
        parameter. upper (float): The upper bound of the parameter. is_int
        (bool): Whether the parameter is an integer.
        """
        idx: int
        category: Optional[List] = None
        lower: Optional[float] = None
        upper: Optional[float] = None
        is_int: bool = False
        # TODO: support quantization

    class ConstraintsGaussianProcess(ExactGP):
        """Gaussian Processes with Constraints."""

        def __init__(self, train_inputs: torch.Tensor,
                     train_targets: torch.Tensor, likelihood: torch.nn.Module,
                     lengthscale_constraint: tuple,
                     outputscale_constraint: tuple,
                     covar_base_kernel_nu: float):
            """Initialize the Gaussian Process.

            Args:
                train_inputs (torch.Tensor): The training inputs.
                train_targets (torch.Tensor): The training targets.
                likelihood (torch.nn.Module): The likelihood.
                lengthscale_constraint (tuple): The lengthscale constraint.
                outputscale_constraint (tuple): The outputscale constraint.
                covar_base_kernel_nu (float): The nu of the base kernel.
            """
            for constraint in [lengthscale_constraint, outputscale_constraint]:

                assert len(constraint) == 2 and constraint[0] <= constraint[1]
            super().__init__(train_inputs, train_targets, likelihood)
            self.mean = ConstantMean()
            self.covar = ScaleKernel(
                MaternKernel(
                    lengthscale_constraint=Interval(*lengthscale_constraint),
                    ard_num_dims=train_inputs.shape[1],
                    nu=covar_base_kernel_nu),
                outputscale_constraint=Interval(*outputscale_constraint))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            """Forward the Gaussian Process.

            Args:
                inputs (torch.Tensor): The inputs.
            Returns:
                torch.Tensor: The outputs.
            """
            return MultivariateNormal(self.mean(inputs), self.covar(inputs))

        def get_weights(self) -> torch.Tensor:
            """Get the weights of the Gaussian Process.

            Returns:
                torch.Tensor: The weights.
            """
            return self.covar.base_kernel.lengthscale.detach().numpy().ravel()

    @staticmethod
    def stable_softmax(inputs: np.array) -> np.array:
        """Stable softmax for numpy.

        Args:
            inputs (np.array): The inputs.
        Returns:
            np.array: The outputs.
        """
        max_value = np.max(inputs)
        return np.exp(inputs - max_value) / np.sum(np.exp(inputs - max_value))

    def __init__(
            self,
            metas: dict,
            mode: str = min,
            min_num_cands: int = 4096,
            likelihood_noise_constraint: tuple = (5e-4, 2e-1),
            lengthscale_constraint: tuple = (5e-3, 2.0),
            outputscale_constraint: tuple = (5e-2, 2e1),
            covar_base_kernel_nu: float = 2.5,
            gp_model_init_cfg={
                'covar.outputscale': 1.0,
                'covar.base_kernel.lengthscale': 0.5,
                'likelihood.noise': 0.005
            },
            lr: float = 0.1,
            num_training_steps: int = 64,
            max_num_successes: int = 3,
            max_num_fails: int = 4,
            trust_region_expand_rate: float = 2.0,
            max_trust_region: float = 1.6,
            trust_region_shrink_rate: float = 0.5,
            min_trust_region: float = 0.5**7,
            init_trust_region: float = 0.8,
            suc_bound: float = 0.,
            max_cholesky_size: int = 2048) -> None:  # noqa E129
        """Initialize the optimizer.

        Args:
            metas (dict): The metas of the parameters.
            mode (str): The mode of the optimizer.
            min_num_cands (int): The minimum number of candidates.
            likelihood_noise_constraint (tuple):
                The likelihood noise constraint.
            lengthscale_constraint (tuple): The lengthscale constraint.
            outputscale_constraint (tuple): The outputscale constraint.
            covar_base_kernel_nu (float): The nu of the base kernel.
            gp_model_init_cfg (dict):
                The initial configuration of the Gaussian Process.
            lr (float): The learning rate.
            num_training_steps (int): The number of training steps.
            max_num_successes (int): The maximum number of successes.
            max_num_fails (int): The maximum number of fails.
            trust_region_expand_rate (float):
                The expand rate of the trust region.
            max_trust_region (float): The maximum trust region.
            trust_region_shrink_rate (float):
                The shrink rate of the trust region.
            min_trust_region (float): The minimum trust region.
            suc_bound (float): The bound of the success rate.
            max_cholesky_size (int): The maximum size of the cholesky factor.
        """

        self.metas = metas
        self._vector_dims = len(self.metas)
        self.history = dict(x=np.zeros((0, self.vector_dims)), y=[])
        assert mode in ['min', 'max']
        self.mode = mode
        self.num_cands = min(100 * self.vector_dims, min_num_cands)
        self.likelihood_noise_constraint = likelihood_noise_constraint
        self.lengthscale_constraint = lengthscale_constraint
        self.outputscale_constraint = outputscale_constraint
        self.covar_base_kernel_nu = covar_base_kernel_nu
        self.gp_model_init_cfg = gp_model_init_cfg
        self.num_training_steps = num_training_steps
        self.lr = lr
        self.num_successes = 0
        self.num_fails = 0
        self.max_num_successes = max_num_successes
        self.trust_region_expand_rate = trust_region_expand_rate
        self.max_num_fails = np.max([max_num_fails, self.vector_dims])
        self.trust_region_shrink_rate = trust_region_shrink_rate
        self.min_trust_region = min_trust_region
        self.max_trust_region = max_trust_region
        self.init_trust_region = init_trust_region
        self.suc_bound = suc_bound
        self.trust_region = init_trust_region
        self.max_cholesky_size = max_cholesky_size

    @property
    def num_evals(self) -> int:
        """Get the number of evaluations.

        Returns:
            int: The number of evaluations.
        """
        return self.history.get('x').shape[0]

    @property
    def vector_dims(self) -> int:
        """Get the number of vector dimensions.

        Returns:
            int: The number of vector dimensions.
        """
        return self._vector_dims

    def _build(self, train_x: torch.Tensor, train_y: torch.Tensor) -> None:
        """Build the optimizer.

        Args:
            train_x (torch.Tensor): The training inputs.
            train_y (torch.Tensor): The training outputs.
        """
        self.train_x = train_x
        self.train_y = train_y
        self.likelihood = GaussianLikelihood(
            noise_constraint=Interval(*self.likelihood_noise_constraint))
        self.gp_model = _Optimizer.ConstraintsGaussianProcess(
            train_x, train_y, self.likelihood, self.lengthscale_constraint,
            self.outputscale_constraint, self.covar_base_kernel_nu).double()
        self.gp_model.initialize(**self.gp_model_init_cfg)
        self.optimizer = torch.optim.Adam(
            self.gp_model.parameters(), lr=self.lr)
        self.loss = ExactMarginalLogLikelihood(self.likelihood, self.gp_model)
        return

    def _train(self):
        """Fit the Gaussian Process."""
        self.gp_model.train()
        self.likelihood.train()
        for _ in range(self.num_training_steps):
            self.optimizer.zero_grad()
            (-self.loss(self.gp_model(self.train_x), self.train_y)).backward()
            self.optimizer.step()
        self.gp_model.eval()
        self.likelihood.eval()

    def _encode(self, inputs: dict) -> np.ndarray:
        """Encode the inputs. project inputs to vector space.

        Args:
            inputs (dict): The inputs.
        Returns:
            np.ndarray: The encoded inputs.
        """
        results = [0.] * self.vector_dims
        for key, meta in self.metas.items():
            if meta.category is None:
                results[meta.idx] = (inputs[key] - meta.lower) / (
                    meta.upper - meta.lower)
            else:
                ct_len = len(meta.category)
                ct_idx = meta.category.index(inputs[key])
                results[meta.idx] = random.uniform(1 / ct_len * ct_idx,
                                                   1 / ct_len * (ct_idx + 1))
        return np.array(results)

    def _decode(self, inputs: np.array) -> dict:
        """Decode the inputs. project inputs to original space.

        Args:
            inputs (np.array): The inputs.
        Returns:
            dict: The decoded inputs.
        """
        result = dict()
        for key, meta in self.metas.items():
            if meta.category is None:
                denorm = inputs[meta.idx] * (meta.upper -
                                             meta.lower) + meta.lower
                result[key] = int(denorm) if meta.is_int else denorm
            else:
                result[key] = meta.category[int(inputs[meta.idx] *
                                                len(meta.category))]
        return result

    def _del(self):
        """Delete gpytorch module."""
        del self.gp_model, self.likelihood, self.train_x, self.train_y, self.optimizer, self.loss  # noqa E501

    def _adjust_trust_region(self, y: float):
        """Adjust the trust region. shrink the trust region if the success rate
        is low. expand the trust region if the success rate is high.

        Args:
            y (float): The value of function.
        """
        y_best = np.min(self.history['y']) if self.mode == 'min' else np.max(
            self.history['y'])
        comp_op = partial(
            operator.ge if self.mode == 'min' else operator.le,
            y_best - math.fabs(y_best) * self.suc_bound if self.mode == 'min'
            else y_best + math.fabs(y_best) * self.suc_bound)
        if comp_op(y):
            self.num_successes += 1
            self.num_fails = 0
        else:
            self.num_successes = 0
            self.num_fails += 1
        if self.num_successes == self.max_num_successes:
            self.trust_region = min([
                self.trust_region_expand_rate * self.trust_region,
                self.max_trust_region
            ])
            self.num_successes = 0
        elif self.num_fails == self.max_num_fails:
            self.trust_region = max([
                self.trust_region_shrink_rate * self.trust_region,
                self.min_trust_region
            ])
            self.num_fails = 0

    def ask(self, seed: Optional[int] = None) -> Dict:
        """Ask for a new point.

        Args:
            seed (Optional[int]): The seed.
        Returns:
            Dict: The new point.
        """
        if self.num_evals == 0:
            return self._decode(np.random.rand(self.vector_dims))

        x = self.history.get('x')
        y = self.history.get('y')
        y_mean, y_std = np.mean(y), np.std(y)
        y_std = 1.0 if y_std < 1e-6 else y_std
        y = (y.copy() - y_mean) / y_std

        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            self._build(torch.tensor(x), torch.tensor(np.array(y)))
            self._train()

        x_cent = np.expand_dims(
            x[y.argmin() if self.mode == 'min' else y.argmax()].copy(), axis=0)
        weights = self.gp_model.get_weights()
        weights /= weights.mean()
        weights /= np.prod(np.power(weights, 1.0 / len(weights)))
        lower = np.clip(x_cent - weights * self.trust_region / 2., 0., 1.)
        upper = np.clip(x_cent + weights * self.trust_region / 2., 0., 1.)

        seed = np.random.randint(2**31) if seed is None else seed
        pert = SobolEngine(
            self.vector_dims, scramble=True,
            seed=seed).draw(self.num_cands).detach().numpy()
        pert = lower + (upper - lower) * pert
        pert_prob = min(20. / self.vector_dims, 1.)

        mask = np.random.rand(self.num_cands, self.vector_dims) <= pert_prob
        mask_idx = np.where(np.sum(mask, axis=1) == 0)[0]
        mask[mask_idx,
             np.random.randint(0, self.vector_dims -
                               1, size=len(mask_idx))] = 1

        x_cand = x_cent * np.ones((self.num_cands, self.vector_dims))
        x_cand[mask] = pert[mask]

        with torch.no_grad(), gpytorch.settings.max_cholesky_size(
                self.max_cholesky_size):
            y_cand = self.gp_model.likelihood(
                self.gp_model(torch.tensor(x_cand))).sample(torch.Size(
                    [1])).t().detach().numpy()

        self._del()

        y_cand = y_mean + y_std * y_cand
        x_next = x_cand[np.argmin(y_cand[..., 0]) if self.mode ==
                        'min' else np.argmax(y_cand[..., 0])].copy()
        return self._decode(x_next)

    def tell(self, x: dict, y: float) -> None:
        """Tell the point and the value.

        Args:
            x (dict): The point.
            y (float): The value.
        """
        x_vector = self._encode(x)
        if self.num_evals != 0:
            self._adjust_trust_region(y)
        self.history['x'] = np.vstack((self.history.get('x'), x_vector.copy()))
        self.history['y'].append(y)
        return


@SEARCHERS.register_module()
class TrustRegionSearcher(Searcher):
    """Trust region searcher."""

    def __init__(
        self,
        space: Optional[Dict] = None,
        metric: Optional[str] = None,
        mode: Optional[str] = None,
    ):
        """Initialize the searcher.

        Args:
            space (Optional[Dict]): The space where the searcher is running.
            metric (Optional[str]): The metric to optimize.
            mode (Optional[str]): The mode to optimize. min or max.
        """

        self._space = self.convert_search_space(space) if (
            isinstance(space, dict) and space) else dict()
        self._metric = metric
        self._mode = mode
        self._optimizer = self._setup_optimizer()
        self._live_trial_mapping = {}

    def _setup_optimizer(self) -> Optional[_Optimizer]:
        """Build the optimizer.

        Returns:
            Optional[_Optimizer]: The optimizer.
        """
        if self._space and self._mode:
            return _Optimizer(self._space, self._mode)
        return None

    def set_search_properties(self, metric: Optional[str], mode: Optional[str],
                              config: Dict, **spec) -> bool:
        """Set the search properties.

        Args:
            metric (Optional[str]): The metric to optimize.
            mode (Optional[str]): The mode to optimize. min or max.
            config (Dict): The config. the space is defined.
        Returns:
            bool: Whether the search properties are set.
        """
        if self._optimizer and self._space:
            return False

        self._space = self.convert_search_space(config)
        if metric:
            self._metric = metric
        if mode:
            self._mode = mode

        self._optimizer = self._setup_optimizer()
        return True

    def suggest(self, trial_id: str) -> Optional[Dict]:
        """Suggest a new trial.

        Args:
            trial_id (str): The trial id.
        Returns:
            Optional[Dict]: The new trial.
        """
        if not self._optimizer:
            raise RuntimeError(
                UNDEFINED_SEARCH_SPACE.format(
                    cls=self.__class__.__name__, space='space'))
        if not self._metric or not self._mode:
            raise RuntimeError(
                UNDEFINED_METRIC_MODE.format(
                    cls=self.__class__.__name__,
                    metric=self._metric,
                    mode=self._mode))

        suggested_config = self._optimizer.ask()

        self._live_trial_mapping[trial_id] = suggested_config
        return unflatten_dict(suggested_config)

    def on_trial_complete(self,
                          trial_id: str,
                          result: Optional[Dict] = None,
                          error: bool = False) -> None:
        """Handle the trial complete event.

        Args:
            trial_id (str): The trial id.
            result (Optional[Dict]): The result.
            error (bool): Whether the trial is error.
        """
        if result:
            self._process_result(trial_id, result)

        self._live_trial_mapping.pop(trial_id)

    def _process_result(self, trial_id: str, result: Dict) -> None:
        """Process the result. Tell the result to the optimizer.

        Args:
            trial_id (str): The trial id.
            result (Dict): The result.
        """
        self._optimizer.tell(self._live_trial_mapping[trial_id],
                             result[self._metric])

    def save(self, checkpoint_path: str) -> None:
        """Save the searcher.

        Args:
            checkpoint_path (str): The path to save the searcher.
        """
        save_object = self.__dict__
        with open(checkpoint_path, 'wb') as outputFile:
            pickle.dump(save_object, outputFile)

    def restore(self, checkpoint_path: str) -> None:
        """Restore the searcher.

        Args:
            checkpoint_path (str): The path to restore the searcher.
        """
        with open(checkpoint_path, 'rb') as inputFile:
            save_object = pickle.load(inputFile)
        self.__dict__.update(save_object)

    def convert_search_space(self, spec: Dict) -> Dict:
        """Convert the search space to the format of the optimizer.

        Args:
            spec (Dict): The search space.
        Returns:
            Dict: The converted search space.
        """
        _, domain_vars, grid_vars = parse_spec_vars(spec)

        if grid_vars:
            raise ValueError(
                'Grid search parameters cannot be automatically converted '
                'to a trust region search space.')

        # Flatten and resolve again after checking for grid search.
        spec = flatten_dict(spec, prevent_delimiter=True)
        _, domain_vars, grid_vars = parse_spec_vars(spec)

        def resolve_value(domain: Domain,
                          vector_idx: int = 0) -> _Optimizer.ParamMeta:
            if isinstance(domain, Categorical):
                param_meta = _Optimizer.ParamMeta(
                    idx=vector_idx, category=domain.categories)
            else:
                sampler = domain.get_sampler()
                assert not isinstance(sampler, Quantized)
                # apply 6-sigma
                lower, upper = (sampler.mean - 3 * sampler.sd,
                                sampler.mean + 3 * sampler.sd) if isinstance(
                                    sampler, Normal) else (domain.lower,
                                                           domain.upper)
                param_meta = _Optimizer.ParamMeta(
                    idx=vector_idx,
                    lower=lower,
                    upper=upper,
                    is_int=True if isinstance(domain, Integer) else False)
            return param_meta

        metas = dict()
        for vector_idx, (path, domain) in enumerate(domain_vars):
            param_meta = resolve_value(domain, vector_idx)
            metas['/'.join(path)] = param_meta

        return metas
