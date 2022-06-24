from ray import tune

from mmtune.ray.searchers import (SEARCHERS, NevergradSearch,
                                  TrustRegionSearcher, build_searcher)


def test_build_searcher():

    @SEARCHERS.register_module()
    class TestSearcher:
        pass

    assert isinstance(build_searcher({'type': 'TestSearcher'}), TestSearcher)


def test_nevergradsearch():
    budget = 2
    config = {
        'steps': 10,
        'width': tune.uniform(0, 20),
        'height': tune.uniform(-100, 100),
    }

    def _objective(config):
        width, height = config['width'], config['height']
        for step in range(config['steps']):
            intermediate_score = (0.1 +
                                  width * step / 100)**(-1) + height * 0.1
            tune.report(iterations=step, mean_loss=intermediate_score)

    tune.run(
        _objective,
        metric='mean_loss',
        mode='min',
        search_alg=NevergradSearch(optimizer='PSO', budget=budget),
        num_samples=budget,
        config=config)


def test_trust_region_searcher():
    budget = 2
    config = {
        'steps': 10,
        'width': tune.uniform(0, 20),
        'height': tune.uniform(-100, 100),
    }

    def _objective(config):
        width, height = config['width'], config['height']
        for step in range(config['steps']):
            intermediate_score = (0.1 +
                                  width * step / 100)**(-1) + height * 0.1
            tune.report(iterations=step, mean_loss=intermediate_score)

    tune.run(
        _objective,
        metric='mean_loss',
        mode='min',
        search_alg=TrustRegionSearcher(),
        num_samples=budget,
        config=config)
