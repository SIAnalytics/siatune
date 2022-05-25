import math
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

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

    @dataclass(frozen=True)
    class ParamMeta:
        idx: Union[List[int], int]
        category: Optional[List] = None
        lower: Optional[float] = None
        upper: Optional[float] = None
        is_int: bool = False
        # TODO: support quantization

    class ConstraintsGaussianProcess(ExactGP):

        def __init__(self, train_inputs: torch.Tensor,
                     train_targets: torch.Tensor, likelihood,
                     lengthscale_constraint: tuple,
                     outputscale_constraint: tuple,
                     covar_base_kernel_nu: float):
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
            return MultivariateNormal(self.mean(inputs), self.covar(inputs))

        def get_weights(self):
            return self.covar.base_kernel.lengthscale.detach().numpy().ravel()

    @staticmethod
    def stable_softmax(inputs: np.array) -> np.array:
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
        suc_bound: float = 1e-3,
        max_cholesky_size: int = 2048):  # noqa E129

        self.metas = metas
        self._vector_dims = max(
            max(meta.idx) if isinstance(meta.idx, list) else meta.idx
            for meta in self.metas.values()) + 1
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
        self.trust_region = init_trust_region
        self.suc_bound = suc_bound
        self.max_cholesky_size = max_cholesky_size

    @property
    def num_evals(self):
        return self.hisrory.get('x').shape[0]

    @property
    def vector_dims(self):
        return self._vector_dims

    def _build(self, train_x: torch.Tensor, train_y: torch.Tensor):
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
        self.gp_model.train()
        self.likelihood.train()
        for _ in range(self.num_training_steps):
            self.optimizer.zero_grad()
            (-self.loss(self.gp_model(self.train_x), self.train_y)).backward()
            self.optimizer.step()
        self.gp_model.eval()
        self.likelihood.eval()

    def _encode(self, inputs: dict) -> np.ndarray:
        results = {}
        for key, meta in self.metas.items():
            if meta.category is None:
                results[meta.idx] = (inputs[key] - meta.lower) / (
                    meta.upper - meta.lower)
            else:
                cat_vector = [0.] * len(meta.category)
                cat_vector[meta.category.index(inputs[key])] = 1.
                for cat_idx, idx in enumerate(meta.idx):
                    results[idx] = cat_vector[cat_idx]
        return np.array(
            [v for _, v in sorted([(int(k), v) for k, v in results.items()])])

    def _decode(self, inputs: np.array) -> dict:
        result = dict()
        for key, meta in self.metas.items():
            if meta.category is None:
                denorm = inputs[meta.idx] * (meta.upper -
                                             meta.lower) + meta.lower
                result[key] = int(denorm) if meta.is_int else denorm
            else:
                sampling_pmf = self.stable_softmax(inputs[meta.idx].copy())
                result[key] = np.random.choice(meta.category, p=sampling_pmf)
        return result

    def _del(self):
        del self.gp_model, self.likelihood, self.train_x, self.train_y, self.optimizer, self.loss  # noqa E501

    def _adjust_trust_region(self, y: float, eps: float = 1e-4):
        best_y = np.min(self.history['y']) if self.mode == 'min' else np.max(
            self.history['y'])
        if math.fabs((y - best_y) / (best_y + eps)) < self.suc_bound:
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

    def tell(self, x: dict, y: float):
        x_vector = self._encode(x)
        if self.num_evals != 0:
            self._adjust_trust_region(y)
        self.history['x'] = np.vstack((self.history.get('x'), x_vector.copy()))
        self.history['y'].append(y)
        return


@SEARCHERS.register_module()
class TrustRegionSearcher(Searcher):

    def __init__(
        self,
        space: Optional[Union[Dict]] = None,
        metric: Optional[str] = None,
        mode: Optional[str] = None,
    ):

        self._space = self.convert_search_space(space) if (
            isinstance(space, dict) and space) else dict()
        self._metric = metric
        self._mode = mode
        self._optimizer = self._setup_optimizer()
        self._live_trial_mapping = {}

    def _setup_optimizer(self) -> Optional[_Optimizer]:
        if self._space and self._mode:
            return _Optimizer(self._space, self._mode)
        return None

    def set_search_properties(self, metric: Optional[str], mode: Optional[str],
                              config: Dict, **spec) -> bool:
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
                          error: bool = False):
        if result:
            self._process_result(trial_id, result)

        self._live_trial_mapping.pop(trial_id)

    def _process_result(self, trial_id: str, result: Dict):
        self._optimizer.tell(self._live_trial_mapping[trial_id],
                             result[self._metric])

    def save(self, checkpoint_path: str):
        save_object = self.__dict__
        with open(checkpoint_path, 'wb') as outputFile:
            pickle.dump(save_object, outputFile)

    def restore(self, checkpoint_path: str):
        with open(checkpoint_path, 'rb') as inputFile:
            save_object = pickle.load(inputFile)
        self.__dict__.update(save_object)

    def convert_search_space(self, spec: Dict) -> Dict:
        resolved_vars, domain_vars, grid_vars = parse_spec_vars(spec)

        if grid_vars:
            raise ValueError(
                'Grid search parameters cannot be automatically converted '
                'to a trust region search space.')

        # Flatten and resolve again after checking for grid search.
        spec = flatten_dict(spec, prevent_delimiter=True)
        resolved_vars, domain_vars, grid_vars = parse_spec_vars(spec)

        def resolve_value(
                domain: Domain,
                vector_idx: int = 0) -> Tuple[_Optimizer.ParamMeta, int]:
            if isinstance(domain, Categorical):
                param_meta = _Optimizer.ParamMeta(
                    idx=list(
                        range(vector_idx,
                              vector_idx + len(domain.categories))),
                    category=domain.categories)
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
            vector_idx_dev = len(domain.categories) if isinstance(
                domain, Categorical) else 1
            return param_meta, vector_idx_dev

        metas = dict()
        vector_idx = 0
        for path, domain in domain_vars:
            param_meta, vector_idx_dev = resolve_value(domain, vector_idx)
            metas['/'.join(path)] = param_meta
            vector_idx += vector_idx_dev

        return metas
