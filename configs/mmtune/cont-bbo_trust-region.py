_base_ = ['./_base_/context/blackbox.py', './_base_/searcher/trust_region.py']

metric = 'result'
mode = 'min'

space = {
    f'_variable{idx}': dict(type='Uniform', lower=-1.0, upper=1.0)
    for idx in range(8)
}

task = dict(type='ContinuousTestFunction')

num_samples = 512
