# Build SIATune

## Dockerfile (RECOMMENDED)
```bash
docker build . -t siatune:master -f docker/Dockerfile
```

## Build From Source

```bash
# 1. install pytorch
pip install torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
# 2. install mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
# 3. clone siatune
git clone -b master https://github.com/SIAnalytics/siatune.git siatune
# 4. install siatune
cd siatune && pip install -e .
```

# Hyperparamer tuning with OpenMMLab's model frameworks.

### Install OpenMMLab's framework.
```bash
# MMDetection Example
pip install mmdet
```

### Start hyperparameter tuning with existed configuration file.
```bash
python tools/tune.py ${TUNE_CONFIG} [optional tune arguments] [optional task arguments]
```


```bash
# MMDetection Example
python tools/tune.py configs/siatune/mmdet_asynchb_nevergrad_pso.py --trainable_args configs/mmdet/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py
```
