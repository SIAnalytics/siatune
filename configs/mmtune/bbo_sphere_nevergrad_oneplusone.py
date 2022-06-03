_base_ = [
    './_base_/context/blackbox.py', './_base_/searcher/nevergrad_oneplusone.py'
]

task = dict(type='Sphere')

metric = 'result'
mode = 'min'

space = dict(
    _variable0=dict(type='Uniform', lower=-2.0, upper=2.0),
    _variable1=dict(type='Uniform', lower=-2.0, upper=2.0),
    _variable2=dict(type='Uniform', lower=-2.0, upper=2.0),
    _variable3=dict(type='Uniform', lower=-2.0, upper=2.0),
    _variable4=dict(type='Uniform', lower=-2.0, upper=2.0),
    _variable5=dict(type='Uniform', lower=-2.0, upper=2.0),
    _variable6=dict(type='Uniform', lower=-2.0, upper=2.0),
    _variable7=dict(type='Uniform', lower=-2.0, upper=2.0),
    _variable8=dict(type='Uniform', lower=-2.0, upper=2.0),
    _variable9=dict(type='Uniform', lower=-2.0, upper=2.0),
    _variable10=dict(type='Uniform', lower=-2.0, upper=2.0),
    _variable11=dict(type='Uniform', lower=-2.0, upper=2.0),
    _variable12=dict(type='Uniform', lower=-2.0, upper=2.0),
    _variable13=dict(type='Uniform', lower=-2.0, upper=2.0),
    _variable14=dict(type='Uniform', lower=-2.0, upper=2.0),
    _variable15=dict(type='Uniform', lower=-2.0, upper=2.0),
)

num_samples = 256
