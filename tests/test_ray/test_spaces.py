from mmtune.ray.spaces import Choice, Constant

def test_choice():
    choice = Choice([dict(model='A'), 
                     dict(model='B'), 
                     dict(model='C')])

def test_constant():
    constant = Constant(value = 3, alias='three')
