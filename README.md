# WCG-MIL

This repository provides the training and evaluation code for the full WCG-MIL model using precomputed UNI patch features.

WCG-MIL combines three-stage context refinement with **SRG** instance scoring and **NRGC** normal-reference grading. The slide prediction combines lesion presence with conditional grade probabilities.

## Installation

Use Python 3.12 and a CUDA-enabled PyTorch 2.3 installation, then install the training dependencies:

```bash
pip install -r requirements.txt
```

## Input data

Provide one PyTorch tensor per sample at `<feature-root>/<sample-id>.pt`. Each tensor must have shape `[number_of_patches, 1024]`. Sample identifiers must match the supplied split CSV files. Obtain the datasets and UNI features or encoder access from their respective providers.

| Dataset | Label order | Split files |
|---|---|---|
| Camelyon+ | negative, ITC, micro, macro | `splits/camelyon_plus/fold_1.csv` through `fold_5.csv` |
| BRACS | benign, atypical, malignant | `splits/bracs/official.csv` |

The split CSV columns are `id`, `label`, `split`, and `group`. Preserve the supplied train/validation/test assignments. Camelyon+ group identifiers include dataset-defined case groups and should not all be interpreted as verified patient identifiers.

## Train WCG-MIL

Single Camelyon+ fold:

```bash
python main.py --name wcg-mil --dataset camelyon_plus --feature-root /path/to/UNI --fold 1 --seed 42
```

All five Camelyon+ folds:

```bash
python run.py --dataset camelyon_plus --feature-root /path/to/UNI
```

BRACS using the official split and seeds 42, 43 and 44:

```bash
python run.py --dataset bracs --feature-root /path/to/bracs_features
```

The queue runs the full WCG-MIL model. Defaults use full bags, batch size 1, AdamW, learning rate and weight decay 0.0001, dropout 0.3, up to 60 epochs, five warm-up epochs, and validation Macro-F1 early stopping with patience 12. SRG uses top-8 presence pooling with temperature 0.25. NRGC builds its reference bank from the training partition.

## Evaluate

```bash
python main.py --name wcg-mil --dataset camelyon_plus --feature-root /path/to/UNI --fold 1 --seed 42 --checkpoint /path/to/best_model.pt
```

Use the same dataset and fold as the checkpoint. Results are written under `outputs/`, including predictions and metrics. Training runs additionally save training history and checkpoints. The queue skips completed runs and attempts recovery from saved training states. Recovery requires matching source, split and training settings; a mismatch is recorded in `resume_rejected.json` and starts a fresh training run. Queue failures are recorded in `attempts.jsonl`, and the queue proceeds to the next job. For an individual recovery, provide the original `--run-id` together with `--resume-training /path/to/last_training_state.pt`.

## Code map

| File | Role |
|---|---|
| `model/wcgmil.py` | WCGMIL, SRG and NRGC forward computation |
| `engine.py` | Training, reference-bank updates and evaluation |
| `data.py`, `splits/` | Feature loading and fixed data partitions |
| `main.py`, `args.py` | Single-run interface |
| `run.py` | Full-model reproduction queue |
| `utils.py` | Seeding, checkpoint loading and output helpers |

Model parameter keys retain their existing descriptive names for state-dictionary compatibility. Single runs enable a small local forward-time probe with `--efficiency`; the reproduction queue enables this probe automatically; the paper's matched 30-bag benchmark uses a separate protocol.

Dataset access and UNI encoder use are subject to their providers' terms. Users must obtain the required data and encoder permissions separately.
