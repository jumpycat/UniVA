# UniVA

Core training code for **A Unified Framework for Verification and Attribution of Generated Images and Generative Models**, by Jianwei Fei, Yunshu Dai, Daniele Baracchi, Dasara Shullani, Zhihua Xia, and Alessandro Piva.

UniVA learns content-agnostic model fingerprints from real images reconstructed by a dynamic fingerprint simulator. A frozen denoiser extracts noise residuals, their log-magnitude Fourier spectra are encoded by ResNet-50, and cross-content/fixed-content branches train the extractor with supervised contrastive and pairwise verification losses. The learned fingerprints support image source verification, model homology verification, and open-set attribution.

This release contains architecture JSONL generation and fingerprint extractor training, together with the required model definitions.

## Setup

Use Python 3.10+ and a compatible PyTorch/torchvision installation. A CUDA GPU is recommended.

```bash
python -m pip install -r requirements.txt
```

Prepare real training images in a flat or nested directory, plus these checkpoints:

- **Fingerprint simulator:** download `250708204953-4m.pth` from [DAIR](https://github.com/jumpycat/DAIR) ([Google Drive](https://drive.google.com/file/d/1v5V1u6873oNj1sSwURqnhIK4A3r3QrM1/view)). The checkpoint must contain `encoder` and `decoder` state dictionaries.
- **Residual denoiser:** provide the pretrained 17-layer Corvi DnCNN checkpoint (`model_best.th`) containing a `network` state dictionary. This is a separate model from the DAIR simulator.

Place weights in `checkpoints/`. ResNet-50 uses ImageNet initialization by default; `--no-pretrained` skips that download and initializes the backbone randomly.

## Generate simulator architectures

```bash
python generate_architectures.py \
  --output-dir data/architectures \
  --num-batches 20000 \
  --seed 42
```

Each numbered JSONL file contains eight `[name, architecture]` records by default. An architecture has 21 indexed layer slots describing the active upsampling stages and convolution/activation/normalization operations. Sampling progresses from independent random configurations to locally modified variants. Options include `--num-classes`, `--random-batches`, and `--mixed-batches`; the default stage boundaries are 4,000 and 8,000 architecture batches. Generated configurations within a batch are distinct.

## Train the fingerprint extractor

```bash
python train.py \
  --train-dir /path/to/real_images \
  --arch-dir data/architectures \
  --synthesis-checkpoint checkpoints/250708204953-4m.pth \
  --denoiser-checkpoint checkpoints/model_best.th \
  --output-dir runs
```

The default recipe uses 10,000 steps, eight configurations per branch, 16 images per configuration, 128-pixel training crops, 128-dimensional embeddings, and Adam with a linear learning-rate schedule from `1e-4` to `3e-6`. Temperature increases from `0.02` to `0.1`; the negative-similarity penalty starts at step 3,000. These defaults retain the supplied training script's recipe and are configurable with `python train.py --help`.

Each run saves configuration, logs, and periodic/final checkpoints. Resume with the same training arguments plus `--resume runs/<run>/step-0001000.pth`. This restores model/optimizer/scheduler state and architecture position; the image loader and random-number states restart. Reduce `--batch-size` or `--num-classes` if needed, keeping both at least two. Use `--workers 0` for single-process data loading.

## Code

| File | Purpose |
| --- | --- |
| `generate_architectures.py` | Architecture sampling and JSONL generation |
| `train.py` | Fingerprint extractor training and checkpointing |
| `encoder.py`, `decoder.py` | Frozen fingerprint simulator |
| `model.py` | Residual denoiser, spectral extractor, and verification heads |
| `losses.py` | Supervised contrastive and balanced pair losses |
| `data.py` | Real-image and architecture-batch loading |

The simulator follows [DAIR](https://github.com/jumpycat/DAIR) ([MIT notice](LICENSE-DAIR)), which also provides simulator pretraining code. The decoder here retains the local project's JSON-loading fix for inactive leading stages and returns only the reconstruction. Architecture generation is adapted from the local `gen_archv12.py` implementation; its stage lengths are expressed in architecture batches, with one batch consumed per enabled training branch. Files cycle if exhausted. Weights, datasets, evaluation scripts, and experimental results are not bundled.
