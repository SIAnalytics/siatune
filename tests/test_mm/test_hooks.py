from unittest.mock import MagicMock

from mmtune.mm.hooks import RayCheckpointHook


def test_hook():
    hook = RayCheckpointHook(
        interval=1,
        by_epoch=True,
        out_dir='/tmp/ray_checkpoint',
        mode='min',
        metric_name='loss',
        max_concurrent=1,
        checkpoint_metric=True,
        checkpoint_at_end=True,
    )
    mock_runner = MagicMock()
    mock_runner.inner_iter = 3
    mock_runner.iter = 5

    cur_iter = hook.get_iter(mock_runner, False)
    assert cur_iter == 6
    cur_iter = hook.get_iter(mock_runner, True)
    assert cur_iter == 4

    mock_runner.model = MagicMock()

    hook._save_checkpoint(mock_runner)
