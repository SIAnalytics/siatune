from mmtune.ray.schedulers import SCHEDULERS, build_scheduler


def test_build_schedulers():

    @SCHEDULERS.register_module()
    class TestScheduler:
        pass

    assert isinstance(
        build_scheduler(dict(type='TestScheduler')), TestScheduler)
