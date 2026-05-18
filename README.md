# ProJo4D: Progressive Joint Optimization for Sparse-View Inverse Physics Estimation

**Transactions on Machine Learning Research (TMLR), 2026**

[Paper (OpenReview)](https://openreview.net/forum?id=pqvVrqlXCZ) | [arXiv](https://arxiv.org/abs/2506.05317) | [Project Page](https://daniel03c1.github.io/ProJo4D/)

> **Note:** Full code and configs are expected to be released by **late May 2026**.

![ProJo4D overview](figures/main.png)

## TL;DR

Estimating 4D geometry and physical parameters from sparse multi-view video is hard: sequential pipelines accumulate errors, while fully joint optimization is unstable on the non-convex landscape. ProJo4D's **progressive joint optimization** gradually expands the set of jointly optimized variables, achieving consistent improvements across synthetic and real-world benchmarks.

## Installation

> *TODO: full environment specification.*

- Python 3.9
- CUDA 12.1
- PyTorch 2.4.0
- Builds on the [GIC](https://github.com/Jukgei/gic) codebase.

## Data preparation

ProJo4D is evaluated on three datasets:

- **PAC-NeRF**: synthetic, dense-view. [project page](https://xuan-li.github.io/PAC-NeRF/)
- **Spring-Gaus**: synthetic, sparse-view evaluation. [project page](https://zlicheng.com/spring_gaus/)
- **Spring-Gaus Real-world**: real captures, sparse-view evaluation. Released with Spring-Gaus.

## Training

`train_projo4d.py` runs the full progressive joint optimization pipeline (4D Gaussian reconstruction, deformation, and physical parameter estimation) from a single entry point. Configs for each dataset and material live under `config/`.

The three required flags are:

- `-c`: path to the experiment config (`projo4d.json`)
- `-s`: source dataset directory
- `-m`: output directory for checkpoints, renders, and logs

Sparse-view experiments share the same config; pass `--cam_idxs` to select which cameras to train on.

```bash
# Dense-view setting (all cameras)
python train_projo4d.py -c config/pacnerf/elastic/projo4d.json \
                        -s data/pacnerf/elastic/0 \
                        -m output/pacnerf/elastic_0

# Sparse-view setting (3 cameras: 1, 5, 9)
python train_projo4d.py -c config/pacnerf/elastic/projo4d.json \
                        -s data/pacnerf/elastic/0 \
                        -m output/pacnerf/elastic_0_CAM1,5,9 \
                        --cam_idxs 1 5 9
```

## Roadmap

- [ ] Installation documentation
- [ ] Training code
- [ ] Per-dataset configs
- [ ] Evaluation scripts

## Citation

```bibtex
@article{rho2026projo4d,
  title   = {ProJo4D: Progressive Joint Optimization for Sparse-View Inverse Physics Estimation},
  author  = {Daniel Rho and Jun Myeong Choi and Biswadip Dey and Roni Sengupta},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  month   = {5},
  url     = {https://openreview.net/forum?id=pqvVrqlXCZ}
}
```

## Acknowledgements

This codebase is built on [Gaussian-Informed Continuum (GIC)](https://github.com/Jukgei/gic) (NeurIPS 2024 Oral).
We thank the authors of [PAC-NeRF](https://xuan-li.github.io/PAC-NeRF/) and [Spring-Gaus](https://zlicheng.com/spring_gaus/) for their datasets and code.

This work was supported by a National Institute of Health (NIH) project #1R21EB035832 *"Next-gen 3D Modeling of Endoscopy Videos"*.
