# Copyright (c) SI-Analytics. All rights reserved.
from typing import List

from ray.tune.experiment import Trial
from ray.tune.integration.mlflow import \
    MLflowLoggerCallback as _MLflowLoggerCallback
from ray.tune.integration.mlflow import logger
from ray.tune.utils.util import is_nan_or_inf

from .builder import CALLBACKS


@CALLBACKS.register_module()
class MLflowLoggerCallback(_MLflowLoggerCallback):
    """Custom MLflow Logger to automatically log Tune results and config to
    MLflow. The main differences from the original MLflow Logger are:

        1. Bind multiple runs into a parent run in the form of nested run.
        2. Log artifacts of the best trial to the parent run.

    Refer to https://github.com/ray-project/ray/blob/ray-1.9.1/python/ray/tune/integration/mlflow.py for details.  # noqa E501

    Args:
        metric (str): Key for trial info to order on.
        mode (str): One of [min, max]. Defaults to ``self.default_mode``.
        scope (str): One of [all, last, avg, last-5-avg, last-10-avg].
            If `scope=last`, only look at each trial's final step for
            `metric`, and compare across trials based on `mode=[min,max]`.
            If `scope=avg`, consider the simple average over all steps
            for `metric` and compare across trials based on
            `mode=[min,max]`. If `scope=last-5-avg` or `scope=last-10-avg`,
            consider the simple average over the last 5 or 10 steps for
            `metric` and compare across trials based on `mode=[min,max]`.
            If `scope=all`, find each trial's min/max score for `metric`
            based on `mode`, and compare trials based on `mode=[min,max]`.
        filter_nan_and_inf (bool): If True, NaN or infinite values
            are disregarded and these trials are never selected as
            the best trial. Default: True.
        **kwargs: kwargs for original ``MLflowLoggerCallback``
    """

    def __init__(self,
                 metric: str = None,
                 mode: str = None,
                 scope: str = 'last',
                 filter_nan_and_inf: bool = True,
                 **kwargs):
        super(MLflowLoggerCallback, self).__init__(**kwargs)
        self.metric = metric
        if mode and mode not in ['min', 'max']:
            raise ValueError('`mode` has to be None or one of [min, max]')
        self.mode = mode
        if scope not in ['all', 'last', 'avg', 'last-5-avg', 'last-10-avg']:
            raise ValueError(
                'ExperimentAnalysis: attempting to get best trial for '
                "metric {} for scope {} not in [\"all\", \"last\", \"avg\", "
                "\"last-5-avg\", \"last-10-avg\"]. "
                "If you didn't pass a `metric` parameter to `tune.run()`, "
                'you have to pass one when fetching the best trial.'.format(
                    self.metric, scope))
        self.scope = scope if scope != 'all' else mode
        self.filter_nan_and_inf = filter_nan_and_inf

    def setup(self, *args, **kwargs):
        """In addition to create `mlflow` experiment, create a parent run to
        bind multiple trial runs."""
        super().setup(*args, **kwargs)

        self.client = self.mlflow_util._get_client()
        self.experiment_id = self.mlflow_util.experiment_id
        self.parent_run = self.client.create_run(
            experiment_id=self.experiment_id, tags=self.tags)

    def log_trial_start(self, trial: 'Trial'):
        """Overrides `log_trial_start` of original `MLflowLoggerCallback` to
        set the parent run ID.

        Args:
            trial (Trial): :class:`ray.tune.experiment.trial.Trial`
        """
        # Create run if not already exists.
        if trial not in self._trial_runs:

            # Set trial name in tags
            tags = self.tags.copy()
            tags['trial_name'] = str(trial)
            tags['mlflow.parentRunId'] = self.parent_run.info.run_id

            run = self.client.create_run(
                experiment_id=self.experiment_id, tags=tags)
            self._trial_runs[trial] = run.info.run_id

        run_id = self._trial_runs[trial]

        # Log the config parameters.
        config = trial.config

        for key, value in config.items():
            self.client.log_param(run_id=run_id, key=key, value=value)

    def on_experiment_end(self, trials: List['Trial'], **info):
        """Overrides `Callback` of `Callback` to copy a best trial to parent
        run. Called after experiment is over and all trials have concluded.

        Args:
            trials (List[Trial]): List of trials.
            **info: Kwargs dict for forward compatibility.
        """
        if not self.metric or not self.mode:
            return

        best_trial, best_score = None, None
        for trial in trials:
            if self.metric not in trial.metric_analysis:
                continue

            score = trial.metric_analysis[self.metric][self.scope]
            if self.filter_nan_and_inf and is_nan_or_inf(score):
                continue

            best_score = best_score or score
            if self.mode == 'max' and score >= best_score or (
                    self.mode == 'min' and score <= best_score):
                best_trial, best_score = trial, score

        if best_trial is None:
            logger.warning(
                'Could not find best trial. Did you pass the correct `metric` '
                'parameter?')
            return

        if best_trial not in self._trial_runs:
            return

        # Copy the run of best trial to parent run.
        run_id = self._trial_runs[best_trial]
        run = self.client.get_run(run_id)
        parent_run_id = self.parent_run.info.run_id

        for key, value in run.data.params.items():
            self.client.log_param(run_id=parent_run_id, key=key, value=value)

        for key, value in run.data.metrics.items():
            self.client.log_metric(run_id=parent_run_id, key=key, value=value)

        if self.should_save_artifact:
            self.client.log_artifacts(
                parent_run_id, local_dir=best_trial.logdir)

        self.client.set_terminated(run_id=parent_run_id, status='FINISHED')
