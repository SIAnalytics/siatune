# Build MMTune

## Dockerfile (RECOMMENDED)
```bash
docker build . -t mmtune:master -f docker/Dockerfile
```

## Build From Source

```bash
# 1. install pytorch
pip install torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
# 2. install mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
# 3. clone mmtune
git clone -b master https://github.com/SIAnalytics/mmtune.git mmtune
# 4. install mmtune
cd mmtune && pip install -e .
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
python tools/tune.py configs/mmtune/mmdet_asynchb_nevergrad_pso.py --trainable_args configs/mmdet/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py
```
