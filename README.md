# Deep Depth from Focus with Transformer

Official PyTorch implementation of a Transformer-based Depth from Focus (DFF) network.

Jing-Sheng Chen, Yong-Xiang Lin, Kai-Lung Hua — National Taiwan University of Science and Technology. *IEEE MultiMedia*, in final revision.

## Overview

Depth from Focus estimates depth from a *focal stack*: several photos of the same scene taken at different focus distances. For each pixel, the network predicts how likely each slice is to be the sharpest one, and the depth is the probability-weighted sum of the slice focus distances.

The blur pattern of an out-of-focus point changes with both its focus distance and its position in the image, so the cues a model needs are spread across slices and across the frame. 3D CNNs, which most deep DFF methods use, see only a local neighbourhood. This model uses a Transformer encoder to compare focus cues across slices and over longer ranges:

- **Hybrid CNN–Transformer encoder.** Residual CNN blocks extract local blur features. 3D Swin Transformer blocks, which attend within shifted windows across both image space and the slice dimension, then compare sharpness between slices at each location.
- **Up-Guided Channel Attention (UGCA).** At each decoder stage, UGCA re-weights the slice channels of the skip connection using global statistics from both the upsampled decoder feature and the skip feature. (In the code this block is the `CSA` class.)
- **Redesigned resampling.** Downsampling adds a strided-convolution path to a max-pooling path. Upsampling uses trilinear interpolation followed by convolution in place of transposed convolution, which reduces checkerboard artifacts and keeps thin structures.

The network has 1.19 M parameters (`W=16, D=4`, window size `4×4×4`).

```
focal stack (B, N, 3, H, W)
  -> CNN stem + ResBlock
  -> DownBlock + ResBlock
  -> [DownBlock + 3D Swin layer] x (D-2)
  -> [UpBlock (trilinear up + conv, UGCA on skip)] x (D-1)
  -> 1x1 conv + softmax over the N slices  -> focus probability (B, N, H, W)
depth = sum_n  probability_n * focus_distance_n
```

## Results

All numbers are from the paper. Lower is better for error metrics; higher is better for δ (the percentage of pixels whose predicted depth is within a factor of 1.25, 1.25², 1.25³ of the ground truth).

**DDFF 12-Scene (real light-field captures, validation set)**

| Method | MSE | RMSE | AbsRel | SqRel | Bump | δ1 | δ2 | δ3 |
|---|---|---|---|---|---|---|---|---|
| DFF-DFV | 5.70e-4 | 2.13e-2 | 0.17 | 6.26e-3 | 0.42 | 76.74 | 94.23 | 98.14 |
| FocDepthFormer | – | 1.96e-2 | 0.16 | 5.4e-3 | **0.23** | 79.06 | 96.08 | 98.57 |
| HybridDepth† | 3.26e-4 | 1.59e-2 | 0.15 | 3.26e-3 | 0.42 | 77.19 | 95.61 | 99.21 |
| AiFDepthNet | 4.28e-5 | 5.14e-3 | 0.28 | 5.00e-4 | 0.31 | 98.87 | 99.56 | 99.76 |
| DfFintheWild | 2.23e-4 | 1.35e-2 | 0.15 | 3.00e-3 | 0.35 | 79.92 | 96.58 | 99.28 |
| **Ours** | **2.22e-5** | **4.10e-3** | **0.03** | **2.44e-4** | 0.33 | **99.26** | **99.81** | **99.93** |

† HybridDepth fuses monocular depth priors with DFF; the other methods are pure DFF.

**FoD500 (synthetic, 5 slices per stack)**

| Method | MSE | RMSE | AbsRel | SqRel |
|---|---|---|---|---|
| AiFDepthNet | 1.27e-2 | 1.04e-1 | 0.11 | 1.87e-2 |
| DfFintheWild | 8.68e-3 | 8.59e-2 | **0.08** | **1.30e-2** |
| FocDepthFormer | – | 1.21e-1 | 0.13 | 2.36e-2 |
| **Ours** | **8.43e-3** | **8.48e-2** | 0.09 | 1.48e-2 |

**Generalization: train on FlyingThings3D, test on unseen Middlebury**

| Method | MAE | MSE | RMSE | AbsRel | SqRel |
|---|---|---|---|---|---|
| AiFDepthNet | 3.825 | 58.570 | 5.936 | 0.165 | 3.039 |
| DfFintheWild | 1.645 | 9.178 | 2.930 | 0.068 | 0.376 |
| **Ours** | **1.417** | **5.575** | **2.296** | **0.055** | **0.189** |

**Inference cost (4D Light Field)**

| Method | Time (ms) | FLOPs (G) | Params (M) |
|---|---|---|---|
| AiFDepthNet | 175.8 | 521 | 16.5 |
| DfFintheWild | 58.8 | 178 | 4.0 |
| DFF-DFV | 7.2 | 175 | 15.5 |
| Ours, FP32 | 188.1 | 252 | **1.2** |
| Ours, mixed precision (AMP) | 10.1 | 252 | **1.2** |

Automatic mixed precision cuts inference time by 94.6% with little change in output quality.

**Encoder ablation (4D Light Field)**

| Encoder | MSE | Bump | Params (M) |
|---|---|---|---|
| **3D Swin (ours)** | **0.0210** | **2.30** | 1.19 |
| 2D Swin | 0.0333 | 3.23 | 1.15 |
| ResNet | 0.0302 | 2.92 | 2.90 |
| DenseNet | 0.0328 | 3.05 | 1.79 |
| ResNeXt | 0.0376 | 3.41 | 0.72 |

The 2D Swin variant processes each slice separately; attention across slices lowers MSE by 36.9%. Removing the Swin encoder, UGCA or the bilinear resampling raises MSE by 43% to 129% depending on the combination; the full table is in the paper.

**Known failure cases.** Error concentrates at occlusion boundaries, where a thin foreground structure in front of a distant background mixes two depths inside one attention window. The model also assumes roughly even focus steps: on an exponentially spaced 5-slice subset of 4D Light Field, MSE rises by a factor of 24.

## Requirements

Tested with Python 3.8.18, PyTorch 1.13.1, torchvision 0.14.1 and a single NVIDIA RTX 2080 SUPER.

```
pip install -r requirements.txt
pip install einops timm
```

`test.py` and `train.py` expect a CUDA GPU.

## Datasets

Put all datasets under `Datasets/`.

* DDFF 12-Scene [1]: [trainval](https://vision.in.tum.de/webarchive/hazirbas/ddff12scene/ddff-dataset-trainval.h5), [test](https://vision.in.tum.de/webarchive/hazirbas/ddff12scene/ddff-dataset-test.h5) → `Datasets/DDFF/`
* FoD500 [2] and 4D Light Field [3]: follow the preparation steps of [AiFDepthNet](https://github.com/albert100121/AiFDepthNet) [6]. The 4D Light Field file goes to `Datasets/HCI/HCI_FS_trainval.h5`.
* FlyingThings3D focal stacks [4]: [FlyingThings3D_FS](https://drive.google.com/file/d/19n3QGhg-IViwt0aqQ4rR8J3sO60PoWgL/view?usp=sharing)
* Middlebury focal stacks [5]: [Middlebury_FS](https://drive.google.com/file/d/1FDXf47Qp1-dT_C7bo30ZySvvPAgJf5FU/view?usp=sharing)

| Dataset | Flag | Ground truth | Range | Slices | Resolution | δ metrics |
|---|---|---|---|---|---|---|
| DDFF 12-Scene | `ddff` | depth (m), normalized 0–1 | ~0.14–2.8 | 10 | 224×224 / 384×576 | yes |
| FoD500 | `def` | depth (m) | 0.1–1.5 | 5 | 256×256 | yes |
| 4D Light Field (HCI) | `hci` | disparity (px), can be negative | — | 10 | 256×256 / 512×512 | no |
| FlyingThings3D | `fly` | disparity (px), negatives masked to 0 | — | 15 | 256×256 / 540×960 | no |
| Middlebury | (test only) | depth, arbitrary unit | 10–60 | 15 | varies | yes |

FoD500 is the dataset released with DefocusNet; the code still calls it `def` / `FS6_dataset`. δ metrics need positive ground truth, so they are not reported on disparity datasets.

## Pretrained models

Checkpoints are included in `Results/`:

| Checkpoint | Trained on | Evaluated on |
|---|---|---|
| `Results/DDFF/ckpt.pth` | DDFF 12-Scene | DDFF 12-Scene |
| `Results/DefocusNet/ckpt.pth` | FoD500 | FoD500 |
| `Results/4D_Light_Field/ckpt.pth` | 4D Light Field | 4D Light Field |
| `Results/FlyingThings3D/Middlebury/ckpt.pth` | FlyingThings3D | Middlebury |
| `Results/FlyingThings3D/DefocusNet/ckpt.pth` | FlyingThings3D | FoD500 |

## Test

```
python test.py --dataset ddff   # ddff | def | hci | fly
```

`--dataset fly` evaluates the FlyingThings3D models on Middlebury and FoD500.

## Train

```
python train.py --dataset ddff --saveroot exp/ --lr 1e-4 --max_epoch 3000 --batch_size 4
```

| Argument | Default | Meaning |
|---|---|---|
| `--dataset` | `ddff` | `ddff`, `def`, `hci` or `fly` |
| `--saveroot` | `exp/` | output folder for checkpoints and logs |
| `--lr` | `1e-4` | Adam learning rate |
| `--max_epoch` | `3000` | number of epochs |
| `--load_epoch` | `0` | resume from this epoch |
| `--batch_size` | `4` | batch size |
| `--cpus` | `8` | data loader workers |

Stack sizes used in the paper are 10 (DDFF), 10 (4D Light Field), 15 (FlyingThings3D) and 5 (FoD500). Random flips, crops, rotations and brightness/contrast jitter are applied to all datasets except DDFF 12-Scene.

## Repository layout

```
model.py        network: DDFT (full model), Swin3DLayer, DownBlock, UpBlock, CSA (= UGCA in the paper)
VideoSwin.py    3D shifted-window Transformer block (from Video Swin Transformer [8])
Dataloader.py   loaders for the five datasets
metrics.py      MSE, RMSE, AbsRel, SqRel, Bump, δ
train.py        training loop
test.py         evaluation with the released checkpoints
Results/        pretrained checkpoints
```

## Citation

```bibtex
@article{chen_ddft,
  title   = {Deep Depth from Focus with Transformer},
  author  = {Chen, Jing-Sheng and Lin, Yong-Xiang and Hua, Kai-Lung},
  journal = {IEEE MultiMedia},
  note    = {In final revision}
}
```

## References

[1] C. Hazirbas et al. Deep Depth from Focus. ACCV 2018.
[2] M. Maximov, K. Galim, L. Leal-Taixé. Focus on Defocus: Bridging the Synthetic to Real Domain Gap for Depth Estimation. CVPR 2020.
[3] K. Honauer et al. A Dataset and Evaluation Methodology for Depth Estimation on 4D Light Fields. ACCV 2016.
[4] N. Mayer et al. A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation. CVPR 2016.
[5] D. Scharstein et al. High-Resolution Stereo Datasets with Subpixel-Accurate Ground Truth. GCPR 2014.
[6] N.-H. Wang et al. Bridging Unsupervised and Supervised Depth from Focus via All-in-Focus Supervision. ICCV 2021.
[7] C. Won, H.-G. Jeon. Learning Depth from Focus in the Wild. ECCV 2022.
[8] Z. Liu et al. Video Swin Transformer. CVPR 2022.
