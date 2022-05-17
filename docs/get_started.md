# Build MMTune

## Download MMTune

```bash
git clone -b master https://github.com/SIAnalytics/mmtune.git mmtune
```
 
## Build

### Dockerfile (RECOMMENDED)
```bash
docker build . -t mmtune:master -f docker/Dockerfile
```

### Build From Source

Install dependencies
```bash
```

# Hyperparamer tuning with OpenMMLab's frameworks.

### Install OpenMMLab's framework.
```bash
# MMDetection Example
pip install mmdet
```

### Start hyperparameter tuning with existed configuration file.
```bash
MMTUNE_TASK_NAME=${TASK_NAME} python tools/tune.py ${TUNE_CONFIG} --task-config ${TASK_CONFIG} [optional arguments]
```


```bash
# MMDetection Example
MMTUNE_TASK_NAME=MMDetection python tools/tune.py configs/mmtune/mmdet_asynchb_nevergrad_pso.py --task-config configs/mmdet/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py
```