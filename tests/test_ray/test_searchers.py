from ray import tune

from mmtune.ray.searchers import SEARCHERS, NevergradSearch, build_searcher


def test_build_searcher():

    @SEARCHERS.register_module()
    class TestSearcher:
        pass

    assert isinstance(build_searcher({'type': 'TestSearcher'}), TestSearcher)


def _objective(config):
    width, height = config['width'], config['height']
    for step in range(config['step']):
        intermediate_score = (0.1 + width * step / 100)**(-1) + height * 0.1
        tune.report(iterations=step, mean_loss=intermediate_score)


def test_nevergradsearch():
    optimizers = [
        'OnePlusOne', 'PSO', 'DE', 'CMA', 'SQP', 'Cobyla', 'Powell', 'BO',
        'Shiwa', 'NGO'
    ]
    budget = 10
    config = {
        'steps': 100,
        'width': tune.uniform(0, 20),
        'height': tune.uniform(-100, 100),
    }

    for optimizer in optimizers:
        search = NevergradSearch(optimizer=optimizer, budget=budget)
        tune.run(
            _objective,
            metric='mean_loss',
            mode='min',
            search_alg=search,
            num_samples=budget,
            config=config)
