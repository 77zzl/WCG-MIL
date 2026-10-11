# NEST-MIL

Implementation accompanying **NEST-MIL: Noncompetitive normal referencing for weakly supervised metastasis grading**.

The repository contains two configurations of one shared framework:

- **NEST-MIL** (`nest-mil`): sparse presence estimation, whole-bag conditional morphology and normal-referenced prototype evidence.
- **NEST-MIL-Core** (`nest-mil-core`): the same presence and morphology architecture, trained without constructing or applying prototype references.

## Installation

The experiments used Python 3.12.12, PyTorch 2.3.0 and an NVIDIA RTX 3090 Ti. Install a PyTorch build appropriate for your CUDA runtime, then install the remaining pinned dependencies:

```bash
pip install -r requirements.txt
```

BRACS image encoding is optional and has additional dependencies:

```bash
pip install -r requirements-encoder.txt
```

No encoder weights, source images, extracted feature tensors or third-party model implementations are bundled. Obtain UNI separately from the authorized provider, accept its access and usage terms, and do not redistribute its weights.

## Data and fixed partitions

Training reads one float32 tensor of shape `[number_of_patches, 1024]` from `<feature-root>/<sample-id>.pt`. A WSI is one Camelyon+ sample; a complete released ROI is one BRACS sample. Subpatches are instances, not independently split samples.

| Dataset | Classes | Evaluation |
|---|---|---|
| Camelyon+ | negative, ITC, micro, macro | 1,350 slides; five fixed grouped folds; seed 42 |
| BRACS | benign, atypical, malignant | 3,657/312/570 official train/validation/test ROIs; seeds 42, 43, 44 |

The CSV files in `splits/` record the exact membership used in the article, without machine-specific paths. Camelyon+ groups use observable patient identifiers where supplied; anonymized slides are separate observable groups, so complete patient linkage is not claimed. BRACS groups are the released patient identifiers. The five Camelyon+ folds were used during method development; they are not an untouched prospective confirmation set. BRACS seeds share one official test cohort.

- Camelyon+: https://doi.org/10.57760/sciencedb.16442
- BRACS: https://www.bracs.icar.cnr.it/
- UNI: https://huggingface.co/MahmoodLab/UNI

For BRACS, use the original `train`, `val`, `test` ROI folders and `BRACS.xlsx`. To generate the same frozen UNI bags:

```bash
python scripts/extract_bracs_features.py --data-root /path/to/BRACS --cache-dir /path/to/bracs_uni --uni-checkpoint /path/to/pytorch_model.bin
```

This extracts native 256-pixel grid tiles, pads incomplete borders with white, resizes tiles to 224 pixels and applies ImageNet normalization. All tiles are retained. Features are saved in `<cache-dir>/features/`. Camelyon+ uses the released UNI feature bags, rather than this ROI extractor.

## Training and evaluation

One run:

```bash
python main.py --name nest-mil --dataset camelyon_plus --feature-root /path/to/CamelyonPlus/UNI --fold 1 --seed 42 --efficiency
```

All five folds for both configurations:

```bash
python run.py --dataset camelyon_plus --feature-root /path/to/CamelyonPlus/UNI
```

Both configurations and the three BRACS seeds:

```bash
python run.py --dataset bracs --feature-root /path/to/bracs_uni/features
```

`run.py` is a finite local queue. It skips completed runs, continues after an isolated task failure and records attempts. Repeat the same command to resume incomplete runs. A stale queue lock requires confirming that the previous process exited before manually removing it. Run from your own terminal; no assistant, scheduler or chat application is required.

Frozen evaluation of a final checkpoint:

```bash
python main.py --name nest-mil --dataset camelyon_plus --feature-root /path/to/CamelyonPlus/UNI --fold 1 --checkpoint /path/to/best_model.pt
```

Add `--disable-reference-residual` only for the frozen inference intervention. It does not reproduce training NEST-MIL-Core. Checkpoint files are not bundled; `utils.load_model_checkpoint` accepts original final model state dictionaries, including the earlier Core files without prototype buffers. Only load checkpoint files you trust. Full optimizer/RNG recovery uses `last_training_state.pt` from this release with the same run ID and protocol, not a historical experiment's full training state.

## Model and training recipe

UNI vectors are projected to 256 dimensions. Three context-refinement stages combine mean/maximum bag context with instance representations. Learned stage weights and a finite-gradient trajectory modulation produce instance risk. Top-8 normalized log-sum-exp estimates presence. Global and risk-weighted means of the **complete bag** feed a conditional morphology classifier.

NEST-MIL maintains eight prototypes per class using only the training partition. After five warm-up epochs, confident positive bags supply high-risk witnesses and negative bags supply low-risk references. Fixed reservoirs are re-embedded in complete-bag context and centers refreshed after training epochs. A positive-to-normal cosine margin is rectified **before** class-specific pooling; a signed bounded residual adjusts conditional log probabilities. The bank is frozen for validation and testing and is saved in the selected checkpoint. Final probabilities are `[1-p, p*q]`, not a multiplication of independent patch and slide classifications.

Both configurations use AdamW, learning rate/weight decay `1e-4`, dropout `0.3`, batch size one, full feature bags, at most 60 epochs, five warm-up epochs, cosine scheduling and validation Macro-F1 patience 12. The loss is joint negative log likelihood with effective-number class weights (`beta=0.99`) normalized by the training class prior. No ordinal, contrastive or perturbation auxiliary loss is used. Historical extent/arbitration parameters are retained solely for original checkpoint and reported parameter-count compatibility; they do not contribute to final classification or its loss.

Logs record per-epoch training/validation time, actual epochs, termination, numerical failures, selected model and full recovery state. Reference refresh time is included in training time; feature extraction is separate. The `--efficiency` flag measures a first-bag diagnostic, not the article's 30-bag matched benchmark.

## Recorded results

| Configuration | Camelyon+ Macro-F1, five folds | BRACS Macro-F1, three seeds |
|---|---|---|
| NEST-MIL | 74.96 ± 3.69% | 75.72 ± 2.21% |
| NEST-MIL-Core | 73.30 ± 2.39% | 77.14 ± 1.71% |

`results/predictions.csv` contains the saved predictions for both configurations and all ten external comparator transfers on both datasets. `results/runs.json` records run identity, prediction-file hashes and recipe. These records are not newly trained results. External-model code and its author-specific dependencies are not included in this compact release; the records preserve their comparison results and explicitly identify local transfers rather than complete upstream reproductions.

Recompute all comparison metrics without training:

```bash
python scripts/summarize_results.py
```

Means and sample standard deviations are computed across folds or seeds; pooled predictions do not substitute for fold means. Full comparisons and adaptations are described in the article and supplement.

## Release verification

The compact implementation was checked against 16 final original checkpoints: ten Camelyon+ checkpoints and six BRACS checkpoints. Strict loading and one complete held-out feature bag per checkpoint produced identical probabilities in both FP32 and CUDA mixed precision. This is an implementation-equivalence check, not a new full-cohort evaluation or a promise of identical results after retraining on another platform.

CUDA short-bag forward/backward checks cover three- and four-class outputs, both configurations and bags with fewer than eight instances. A three-epoch synthetic-data check exercised reference activation, checkpoint selection and optimizer/RNG recovery. No new paper experiment was run during release preparation. Run the small regression checks with `python -m unittest discover -s checks`.

## Files

```text
args.py / main.py / run.py   configuration, one run, finite reproduction queue
data.py                     full feature bags and fixed split manifests
engine.py                   joint loss, training-only references, checkpoints and evaluation
model/nestmil.py             shared architecture and normal-reference computation
model/registry.py            nest-mil and nest-mil-core
scripts/                    optional BRACS encoding and result recalculation
splits/                     portable original partitions
configs/paper_protocol.json original paper recipe and BRACS encoding protocol
results/                    derived predictions, provenance and recomputed metrics
checks/test_model.py         short-bag, finite-gradient and checkpoint regression checks
```

## Release status and rights

This code is released under the [MIT License](LICENSE), as confirmed by the authors. Dataset and encoder rights remain subject to their own terms; the MIT license does not relicense third-party data or UNI weights. Restricted encoder weights and original images must not be committed.

This is a local release preparation; no GitHub repository has been uploaded by this task. The public repository URL is intentionally left blank until the authors create it. Update this release-status paragraph and the manuscript's code-availability statement when the repository is published.

The manuscript is currently under preparation for submission; no accepted-paper DOI or citation metadata is invented.
