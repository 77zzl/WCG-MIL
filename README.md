# NEST-MIL

Code for **NEST-MIL: Noncompetitive normal referencing for weakly supervised metastasis grading**.

- `nest-mil`: context refinement, sparse presence estimation, conditional morphology, and normal-referenced evidence.
- `nest-mil-core`: the shared architecture without the prototype reference branch.

## Install

The experiments used Python 3.12.12 and PyTorch 2.3.0 on an RTX 3090 Ti. For GPU training, first install a CUDA-enabled PyTorch 2.3.0 build following the [official instructions](https://pytorch.org/get-started/previous-versions/#v230), then install the requirements:

```bash
pip install -r requirements.txt
```

For BRACS feature extraction, also install `requirements-encoder.txt`.

## Data

Store each feature bag as `<feature-root>/<sample-id>.pt`: a float32 tensor with shape `[number_of_patches, 1024]`.

| Dataset | Classes | Partitions |
|---|---|---|
| [Camelyon+](https://doi.org/10.57760/sciencedb.16442) | negative, ITC, micro, macro | Five grouped folds in `splits/camelyon_plus/` |
| [BRACS](https://www.bracs.icar.cnr.it/) | benign, atypical, malignant | Official ROI split in `splits/bracs/official.csv`; seeds 42, 43, 44 |

Camelyon+ uses the released UNI feature bags. For BRACS, obtain the original `train`, `val`, and `test` ROI folders, `BRACS.xlsx`, and your own authorized UNI checkpoint, then run:

```bash
python scripts/extract_bracs_features.py --data-root /path/to/BRACS --cache-dir /path/to/bracs_uni --uni-checkpoint /path/to/pytorch_model.bin
```

This creates `<cache-dir>/features/`. Each ROI is one bag; native 256-pixel tiles are white-padded at image borders, resized to 224 pixels, and encoded with frozen UNI.

## Run

Train both configurations on the five Camelyon+ folds:

```bash
python run.py --dataset camelyon_plus --feature-root /path/to/CamelyonPlus/UNI
```

Train both configurations on BRACS with all three seeds:

```bash
python run.py --dataset bracs --feature-root /path/to/bracs_uni/features
```

Train one configuration and fold:

```bash
python main.py --name nest-mil --dataset camelyon_plus --feature-root /path/to/CamelyonPlus/UNI --fold 1 --seed 42 --efficiency
```

Evaluate a saved checkpoint:

```bash
python main.py --name nest-mil --dataset camelyon_plus --feature-root /path/to/CamelyonPlus/UNI --fold 1 --checkpoint /path/to/best_model.pt
```

Use `--name nest-mil-core` for the Core configuration. Outputs are saved under `outputs/`. Repeating the queue command skips completed jobs and resumes unfinished runs; individual failures are recorded without stopping the entire queue. Remove a stale queue lock only after its previous process has exited. Full training recovery uses `last_training_state.pt` from the same run.

The default recipe is AdamW, learning rate and weight decay `1e-4`, dropout `0.3`, full bags, batch size one, 60 maximum epochs, five warm-up epochs, and validation Macro-F1 patience 12. Final probabilities combine presence with conditional grading as `[1-p, p*q]`.

## Files

| File | Purpose |
|---|---|
| `args.py`, `main.py`, `run.py` | Options, individual runs, and reproduction queue |
| `data.py`, `splits/` | Feature bags and fixed partitions |
| `model/` | NEST-MIL and NEST-MIL-Core |
| `engine.py`, `utils.py` | Loss, prototype updates, checkpoints, and metrics |
| `scripts/extract_bracs_features.py` | Frozen UNI encoding of BRACS ROIs |

## UNI encoder

UNI is governed by [CC BY-NC-ND 4.0 and MahmoodLab's additional access and usage terms](https://huggingface.co/MahmoodLab/UNI). Each user must obtain authorization and accept those terms. UNI weights are not included and must not be redistributed. Commercial use of UNI or derivatives, including models trained on its outputs, requires prior approval from MahmoodLab. Dataset use is governed by the respective providers' terms.

When using UNI, cite [Chen et al., Nature Medicine, 2024](https://doi.org/10.1038/s41591-024-02857-3), [ViT](https://openreview.net/forum?id=YicbFdNTTy), and [DINOv2](https://openreview.net/forum?id=a68SUt6zFt), as requested by its provider.
