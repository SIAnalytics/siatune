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

# Hyperparamer tuning with OpenMMLab's model frameworks.

### Install OpenMMLab's framework.
```bash
# MMDetection Example
mim install mmdet
```

### Prepare dataset
```bash
ln -s /some/path/COCO2017Det data/coco
```

### Start hyperparameter tuning with existed configuration file.
```bash
python tools/tune.py ${TUNE_CONFIG} [optional tune arguments] --trainable-args [optional task arguments]
```

```bash
# MMDetection Example
mim download mmdet --config faster_rcnn_r50_fpn_1x_coco --dest configs/mmdet/faster_rcnn
python tools/tune.py configs/mmdet/mmdet_asynchb_nevergrad_pso.py --trainable-args configs/mmdet/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py
```
