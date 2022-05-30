_base_ = ['./_base_/context/blackbox.py', './_base_/searcher/trust_region.py']

metric = 'result'
mode = 'min'

space = {
    f'_variable{idx}':
    dict(type='Choice', categories=[0, 1], alias=['ON', 'OFF'])
    for idx in range(8)
}

task = dict(type='DiscreteTestFunction')

num_samples = 512
