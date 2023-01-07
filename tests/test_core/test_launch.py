import ray
from ray.util.queue import Queue

from siatune.core.launch import DistributedTorchLauncher


def test_dist_torch_launcher():
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=1)
    queue = Queue(2)

    def func():
        queue.put(1)

    launcher = DistributedTorchLauncher(num_cpus_per_worker=0.5, num_workers=2)
    launcher.launch(func)

    ret = 0
    while not queue.empty():
        ret += queue.get(block=True)

    assert ret == 2
    ray.shutdown()
