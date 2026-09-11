# RTG-IMVC

PyTorch implementation of RTG-IMVC for incomplete multi-view clustering.

## Environment

Python 3.12 and PyTorch 2.3.1.

## Requirements

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run

Run the included MSRC-v1 dataset:

```bash
python train/train.py --dataset MSRC_v1
```

Use `--paper-protocol` for missing rates 0.1/0.3/0.5 and five seeds.
Dataset and model settings are defined in `config.py`.

## Datasets

MSRC-v1 is included in `data/`. Download the remaining datasets separately
and place their MAT files in this directory: `NGs.mat`, `NoisyMNIST.mat`,
`3V_Fashion_MV.mat`, `Caltech-5V.mat`, `BBC.mat`, `handwritten.mat`, and
`3-sources.mat`. Select the dataset using `--dataset` and its filename
without `.mat`.

For a new dataset, add its loader to `utils/datasets.py` and its network
settings to `config.py`.
