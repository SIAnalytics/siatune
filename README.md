# MMTune - hyperparameter tuning toolbox for OpenMMLab frameworks

<div align="center">
  <img src="resources/mmtune-logo.png" width="450"/>
</div>

## Introduction
MMTune is an open-source deep learning model hyperparameter tuning toolset especially for OpenMMLab frameworks such as [mmdetection](https://github.com/open-mmlab/mmdetection) and [mmsegmentation](https://github.com/open-mmlab/mmsegmentation). Most of its tuning system is based on [Ray.tune](https://docs.ray.io/en/latest/tune/index.html).

Currently, it is maintained by SI-Analytics.

### Major features

- **Fully support OpenMMLab models**

  We provide a unified model hyperparameter tuning toolbox for the codebases in OpenMMLab. The supported codebases are listed as below, and more will be added in the future
  - [ ] [MMClassification](https://github.com/open-mmlab/mmclassification)
  - [x] [MMDetection](https://github.com/open-mmlab/mmdetection)
  - [x] [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
  - [ ] [MMEditing](https://github.com/open-mmlab/mmediting)

- **Support state-of-the-art hyperparameter tuning algorithms**

    We provide varoius tuning algorithms such as below;
  - [x] [Nevergrad](https://github.com/facebookresearch/nevergrad)
  - [x] [Adaptive Experimentation (AX)](https://ax.dev/)
  - [x] [HyperOpt](https://github.com/hyperopt/hyperopt)
  - [x] [FLAML](https://github.com/microsoft/FLAML)
  - [x] [Scikit-optimize](https://github.com/scikit-optimize/scikit-optimize)

- **Distributed tuning system based on Ray**

    Hyperparameter tuning with multi-GPU training or large-scale job scheduling are managed by Ray's distributed compute framework.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Installation

Please refer to [get_started.md](docs/get_started.md) for installation.
