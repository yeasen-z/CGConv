# CGConv

CGConv (Center Glance Convolution) is a method for coarse-grained activation sparsity. For each input channel and convolution position, it glances at the center activation to determine whether the complete convolution window is retained:

$$
g_p = \mathbb{1}[X_{p,\mathrm{center}} \neq 0], \qquad
\hat{X}_p = g_p X_p.
$$

The convolution operates on $\hat{X}_p$. A differentiable activation regularizer can be enabled during training to encourage more zero-valued gates:

$$
L = L_{CE} + \lambda\,\mathrm{mean}_{l,x}
\left(\frac{x^2}{x^2+\sigma}\right).
$$

This repository provides:

- a PyTorch implementation of `CGConv2d`;
- CIFAR-10 ResNet and CGConv model variants;
- training, evaluation, and checkpoint resume;
- hook-based FLOPs and activation-sparsity analysis;
- a C++ implementation of Center Glance compact convolution.

## Repository layout

```text
.
├── cgToolkit/
│   ├── CGConv.py             # CGConv2d implementation
│   ├── sponge_loss.py        # Activation regularizer
│   ├── flops_cal.py          # Active-window statistics
│   └── sparsity_analysis.py  # Activation sparsity statistics
├── models/resnet_cifar10.py  # CIFAR ResNet variants
├── cg_train.py               # Training and evaluation
├── cg_analysis.ipynb         # FLOPs analysis notebook
├── hls/
│   ├── modules/conv2d_spmv/  # Center Glance compact convolution
│   ├── modules/conv_mv/      # Dense convolution reference
│   └── MsLib/                # Neural-network operator library
├── demo.sh
└── requirements.txt
```

## Installation

Python 3.9 or later is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Training

Train CGConv ResNet-56 on CIFAR-10:

```bash
python cg_train.py \
  --dataset cifar10 \
  --data_path ./data \
  --arch cgc_resnet56 \
  --save_path ./save/cgc_resnet56 \
  --epochs 200 \
  --learning_rate 0.01 \
  --schedule 80 120 \
  --gammas 0.1 0.1 \
  --cgc
```

Train the standard ResNet-56 model:

```bash
python cg_train.py \
  --dataset cifar10 \
  --data_path ./data \
  --arch resnet56 \
  --save_path ./save/resnet56 \
  --epochs 200
```

The training script writes the latest checkpoint, best checkpoint, accuracy log, training curve, and run log to `save_path`.

Resume training or evaluate a checkpoint:

```bash
# Resume
python cg_train.py --arch cgc_resnet56 --cgc \
  --resume ./save/cgc_resnet56/checkpoint.pth.tar \
  --save_path ./save/cgc_resnet56

# Evaluate
python cg_train.py --arch cgc_resnet56 --evaluate \
  --resume ./save/cgc_resnet56/model_best.pth.tar
```

The demo script accepts environment-variable overrides:

```bash
DATA_PATH=./data GPU_ID=0 EPOCHS=200 bash demo.sh
```

## Analysis

`cgToolkit.flops_ratio` collects active-element and active-window ratios for selected layers through forward hooks:

- `f_type="Sponge"` applies the Center Glance rule;
- `f_type="Sponge0"` marks a window active when any element is nonzero.

`cgToolkit.get_sparsity` collects total, regular, and irregular activation-zero statistics for selected convolution layers.

The analysis workflow is demonstrated in `cg_analysis.ipynb`. The notebook records a dense ResNet-56 size of approximately 0.85M parameters and 127.62M FLOPs.

## Recorded experiment

A 500-epoch CIFAR-10 run of `cgc_resnet56` recorded a best test accuracy of **92.82%** at epoch 214. The run used Python 3.10.16, PyTorch 2.6.0+cu124, `sponge_lb=0.05`, and `sponge_sigma=1e-4`.

## C++ implementation

`hls/modules/conv2d_spmv` implements the following computation flow:

```text
im2col -> center-based window compaction -> compact GEMM
       -> position restoration -> input-channel accumulation -> bias
```

See [`hls/README.md`](hls/README.md) for the module layout.
