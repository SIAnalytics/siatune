import glob
import re
import shutil
import tempfile
import threading
from os import path as osp
from typing import Dict, List, Optional

from ray.tune.integration.mlflow import \
    MLflowLoggerCallback as _MLflowLoggerCallback
from ray.tune.integration.mlflow import logger
from ray.tune.result import TIMESTEPS_TOTAL, TRAINING_ITERATION
from ray.tune.trial import Trial
from ray.tune.utils.util import is_nan_or_inf

from .builder import CALLBACKS


def _create_temporary_copy(path, temp_file_name):
    temp_dir = tempfile.gettempdir()
    temp_path = osp.join(temp_dir, temp_file_name)
    shutil.copy2(path, temp_path)
    return temp_path


@CALLBACKS.register_module()
class MLflowLoggerCallback(_MLflowLoggerCallback):

    TRIAL_LIMIT = 5

    def __init__(self,
                 work_dir: Optional[str],
                 metric: str = None,
                 mode: str = None,
                 scope: str = 'last',
                 filter_nan_and_inf: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.work_dir = work_dir
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
        self.thrs = []

    def setup(self, *args, **kwargs):
        cp_trial_runs = getattr(self, '_trial_runs', dict()).copy()
        super().setup(*args, **kwargs)
        self._trial_runs = cp_trial_runs
        self.parent_run = self.client.create_run(
            experiment_id=self.experiment_id, tags=self.tags)

    def log_trial_start(self, trial: 'Trial'):
        # Create run if not already exists.
        if trial not in self._trial_runs:

            # Set trial name in tags.
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
            key = re.sub(r'[^a-zA-Z0-9_=./\s]', '', key)
            self.client.log_param(run_id=run_id, key=key, value=value)

    def log_trial_result(self, iteration: int, trial: 'Trial', result: Dict):
        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
        run_id = self._trial_runs[trial]
        for key, value in result.items():
            key = re.sub(r'[^a-zA-Z0-9_=./\s]', '', key)
            try:
                value = float(value)
            except (ValueError, TypeError):
                logger.debug('Cannot log key {} with value {} since the '
                             'value cannot be converted to float.'.format(
                                 key, value))
                continue
            for idx in range(MLflowLoggerCallback.TRIAL_LIMIT):
                try:
                    self.client.log_metric(
                        run_id=run_id, key=key, value=value, step=step)
                except Exception as ex:
                    print(ex)
                    print(f'Retrying ... : {idx+1}')

    def log_trial_end(self, trial: 'Trial', failed: bool = False):

        def log_artifacts(run_id,
                          path,
                          trial_limit=MLflowLoggerCallback.TRIAL_LIMIT):
            for idx in range(trial_limit):
                try:
                    self.client.log_artifact(
                        run_id, local_path=path, artifact_path='checkpoint')
                except Exception as ex:
                    print(ex)
                    print(f'Retrying ... : {idx+1}')

        run_id = self._trial_runs[trial]

        if self.save_artifact:
            trial_id = trial.trial_id
            work_dir = osp.join(self.work_dir, trial_id)
            checkpoints = glob.glob(osp.join(work_dir, '*.pth'))
            if checkpoints:
                pth = _create_temporary_copy(
                    max(checkpoints, key=osp.getctime), 'model_final.pth')
                th = threading.Thread(target=log_artifacts, args=(run_id, pth))
                self.thrs.append(th)
                th.start()

            cfg = _create_temporary_copy(
                glob.glob(osp.join(work_dir, '*.py'))[0], 'model_config.py')
            if cfg:
                th = threading.Thread(target=log_artifacts, args=(run_id, cfg))
                self.thrs.append(th)
                th.start()

        # Stop the run once trial finishes.
        status = 'FINISHED' if not failed else 'FAILED'
        self.client.set_terminated(run_id=run_id, status=status)

    def on_experiment_end(self, trials: List['Trial'], **info):
        for th in self.thrs:
            th.join()

        def cp_artifacts(src_run_id,
                         dst_run_id,
                         tmp_dir,
                         trial_limit=MLflowLoggerCallback.TRIAL_LIMIT):
            for idx in range(trial_limit):
                try:
                    self.client.download_artifacts(
                        run_id=src_run_id, path='checkpoint', dst_path=tmp_dir)
                    self.client.log_artifacts(
                        run_id=dst_run_id,
                        local_dir=osp.join(tmp_dir, 'checkpoint'),
                        artifact_path='checkpoint')
                except Exception as ex:
                    print(ex)
                    print(f'Retrying ... : {idx+1}')

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

        run_id = self._trial_runs[best_trial]
        run = self.client.get_run(run_id)
        parent_run_id = self.parent_run.info.run_id
        for key, value in run.data.params.items():
            self.client.log_param(run_id=parent_run_id, key=key, value=value)
        for key, value in run.data.metrics.items():
            self.client.log_metric(run_id=parent_run_id, key=key, value=value)

        if self.save_artifact:
            tmp_dir = tempfile.gettempdir()
            th = threading.Thread(
                target=cp_artifacts, args=(run_id, parent_run_id, tmp_dir))
            th.start()
            th.join()

        self.client.set_terminated(run_id=parent_run_id, status='FINISHED')
