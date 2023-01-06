import ray
from siatune.core.launch import DistTorchLauncher

def test_dist_torch_launcher():
    ray.init(num_cpus=1)
    ret = [0] * 2
    def func():
        import os
        ret[int(os.environ["RANK"])] = 1
    launcher = DistTorchLauncher(num_cpus_per_worker=0.5, num_workers=2)    
    launcher.launch(func)
    assert sum(ret) == 2
    ray.shutdown()

