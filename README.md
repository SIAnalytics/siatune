
<div align="center">
  <img src="resources/siatune-logo.png" width="450"/>
</div>

## Introduction
SIATune is an open-source deep learning model hyperparameter tuning toolbox especially for OpenMMLab's model frameworks such as [mmdetection](https://github.com/open-mmlab/mmdetection) and [mmsegmentation](https://github.com/open-mmlab/mmsegmentation). In order to support job scheduling and resource management, SIATune adopts [Ray](https://github.com/ray-project/ray) and [Ray.tune](https://docs.ray.io/en/latest/tune/index.html).

### Major features

- **Fully support OpenMMLab models**

  We provide a unified model hyperparameter tuning toolbox for the codebases in OpenMMLab. The supported codebases are listed as below, and more will be added in the future
  - [x] [MMClassification](https://github.com/open-mmlab/mmclassification)
  - [x] [MMDetection](https://github.com/open-mmlab/mmdetection)
  - [x] [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
  - [x] [MMEditing](https://github.com/open-mmlab/mmediting)

- **Support hyperparameter search algorithms**

  We provide hyperparameter search algorithms such as below;
  - [x] [FLAML](https://github.com/microsoft/FLAML)
  - [x] [HyperOpt](https://github.com/hyperopt/hyperopt)
  - [x] [Nevergrad](https://github.com/facebookresearch/nevergrad)
  - [x] [Optuna](https://github.com/optuna/optuna)
  - [ ] [Adaptive Experimentation (AX)](https://ax.dev/)
  - [ ] [Scikit-optimize](https://github.com/scikit-optimize/scikit-optimize)

- **Schedule multiple experiments**

  Various scheduling techniques are supported to efficiently manage many experiments.
  - [x] [Asynchronous HyperBand](https://arxiv.org/abs/1810.05934)
  - [x] [HyperBand](https://arxiv.org/abs/1603.06560)
  - [x] [Median Stopping Rule](https://research.google.com/pubs/pub46180.html)
  - [ ] [Population Based Training](https://www.deepmind.com/blog/population-based-training-of-neural-networks)
  - [ ] [Population Based Bandits](https://arxiv.org/abs/2002.02518)
  - [x] [Bayesian Optimization and HyperBand](https://arxiv.org/abs/1807.01774)


- **Distributed tuning system based on Ray**

    Hyperparameter tuning with multi-GPU training or large-scale job scheduling are managed by Ray's distributed compute framework.

## Installation and Getting Started

Please refer to [get_started.md](docs/get_started.md) for installation and getting started.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Citing SIATune

If you use SIATune in your research, please use the following BibTeX entry.

```BibTeX
@misc{na2022siatune,
  author =       {Younghwan Na and Hakjin Lee and Junhwa Song},
  title =        {SIATune},
  howpublished = {\url{https://github.com/SIAnalytics/siatune}},
  year =         {2022}
}
```
