# Build SIATune

## Dockerfile (RECOMMENDED)
```bash
docker build . -t siatune:main -f docker/Dockerfile
```

## Build From Source

```bash
# 1. Install PyTorch
pip install torch torchvision -f https://download.pytorch.org/whl/torch_stable.html

# 2. Install MIM and MMCV
pip install openmim
mim install mmcv-full

# 3. Install SIATune
git clone https://github.com/SIAnalytics/siatune.git
cd siatune
pip install -e '.[optional]'
```

# Hyperparameter tuning with OpenMMLab's model frameworks

### Start hyperparameter tuning with existed configuration file
```bash
python tools/tune.py ${TUNE_CONFIG} [optional tune arguments] \
    --trainable-args ${TASK_CONFIG} [optional task arguments]
```

## MMDetection

### Prepare datasets
Please refer to [this link](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md#prepare-datasets).

### Run scripts
```bash
# MMDetection Example
mim install mmdet
mim download mmdet --config faster_rcnn_r50_fpn_1x_coco --dest configs/mmdet/faster_rcnn
python tools/tune.py configs/mmdet/mmdet_asynchb_nevergrad_pso.py --trainable-args configs/mmdet/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py
```
