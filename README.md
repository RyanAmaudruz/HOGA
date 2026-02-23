HOGA
===============================

[![arXiv](https://img.shields.io/badge/arXiv-2403.01317-b31b1b.svg)](https://arxiv.org/abs/2403.01317)

HOGA is an attention model for scalable and generalizable learning on circuits. By leveraging a novel gated attention module on hop-wise features,  HOGA not only outperforms prior graph learning models on challenging circuit problems, but is also friendly to distributed training by mitigating communication overhead caused by graph dependencies. This renders HOGA applicable to industrial-scale circuit applications. More details are available in [our paper](https://arxiv.org/abs/2403.01317).

| ![HOGA.png](/figures/HOGA.png) | 
|:--:| 
| Figure1: An overview of HOGA and gated attention module. |

Requirements
------------
* python 3.9
* pytorch 1.12 (CUDA 11.3)
* torch_geometric 2.1

Datasets
------------
### Pre-processed CSA and Booth Multipliers (for Gamora experiments)
Check at: https://huggingface.co/datasets/yucx0626/Gamora-CSA-Multiplier/tree/main
### Pre-processed OpenABC-D benchmark (for OpenABC-D experiments)
Check at: https://zenodo.org/records/6399454#.YkTglzwpA5k

Note
------------
The implementation of hop-wise feature generation is available in `src/utils/preprocess.py`. The model (i.e., hop-wise gated attention) implementation is available in `src/models/hoga.py`. You can adjust them for your own tasks.

Repository structure
------------
```
├── src/
│   ├── models/       # HOGA model
│   ├── data/         # Datasets, dataloaders, evaluator
│   ├── training/     # Training and evaluation logic
│   ├── evaluation/   # Metrics
│   └── utils/        # Preprocessing, logging
├── scripts/
│   └── train.py      # Training entry point
├── configs/
├── notebooks/
├── tests/
└── results/
```

Minimal training command
------------
```bash
python scripts/train.py --root_dir /path/to/datasets
``` 

Citation
------------
If you use HOGA in your research, please cite our work
published in DAC'24.

```
@inproceedings{deng2024hoga,
  title={Less is More: Hop-Wise Graph Attention for Scalable and Generalizable Learning on Circuits},
  author={Chenhui Deng and Zichao Yue and Cunxi Yu and Gokce Sarar and Ryan Carey and Rajeev Jain and Zhiru Zhang},
  booktitle={DAC},
  year={2024},
}
```

