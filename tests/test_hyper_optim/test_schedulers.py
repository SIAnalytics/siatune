from siatune.hyper_optim.schedulers import TRIAL_SCHEDULERS, build_scheduler


def test_build_schedulers():

    @TRIAL_SCHEDULERS.register_module()
    class TestScheduler:
        pass

    assert isinstance(
        build_scheduler(dict(type='TestScheduler')), TestScheduler)
