# Embedded ML Speech Command Recognition on ATmega32A
> [!IMPORTANT]
>## Version 2 of this project can be found at: https://www.kaggle.com/code/michaelmatta0/embedded-ml-keyword-spotting-2kb-ram-32kb-flash


This repository contains a complete machine learning pipeline for embedded deployment: train on PC, export to a C header, and run fixed-shape inference on ATmega32A firmware.



## 1) At a Glance

- Task: 8-word isolated command recognition (`on`, `off`, `go`, `stop`, `left`, `right`, `up`, `down`)
- Dataset: Google Speech Commands
- Baseline accuracy: 38.6%
- Final accuracy: **89.7%**
- Target: ATmega32A (2 KB RAM, 32 KB flash)
- Final model: `N_FEATURES=353`, `LDA_DIMS=7`, `K_TEMPLATES=16`
- Final flash usage: 28,704 B model+mel, with 564 B spare after 3,500 B reserve

## 2) Project Intent

The project goal is to push command recognition accuracy as high as possible while preserving strict embedded deployment constraints.

This is not only a training project; it is an end-to-end deployment project where exported parameters must run with a fixed AVR inference routine.

Methodology note: this pipeline is AI-assisted and research-driven, built by implementing and validating ideas inspired by a large body of keyword spotting and speech-processing literature to reach the best possible accuracy under fixed hardware limits without deep learning inference on-device.

## 3) Hard Constraints

- `classify_snippet.c` inference geometry is fixed:
  - z-score normalization
  - projection `fv @ W - xbar`
  - nearest-template (1-NN)
- `word_templates.h` schema is fixed:
  - `N_FEATURES`, `LDA_DIMS=7`, `K_TEMPLATES`, `N_WORDS=8`
  - arrays stored as `PROGMEM` floats
- Flash budget rule is fixed:
  - `model_flash + mel_flash + 3500 <= 32768`

These constraints dictated all model and feature decisions.

## 4) Final Deployment State

- Best accuracy: **89.7%** on deterministic fallback stratified split (20% test)
- `N_FEATURES=353`, `LDA_DIMS=7`, `K_TEMPLATES=16`
- Flash: model=16,320 B + mel=12,384 B = 28,704 B
- Reserve-aware total: 32,204 B / 32,768 B
- `word_templates.h` regenerated from final locked pipeline
- `realtime_test.py --test_mode` parity gate validated at 89.7% (within ±1%)

## 5) What Was Done and Why

The pipeline was improved in phases. A change was kept only if it improved constrained accuracy.

### 5.1 Baseline

- Initial pipeline: 14 hand-crafted features + LDA(7D) + 1-NN
- Accuracy: 38.6%
- Worst confusions: on<->down, left<->right, stop<->left/up
- Root cause: the initial feature stack was largely time-domain and too coarse for robust separation

### 5.2 Feature Replacement and Expansion (to ~66%)

Kept:

- MFCC (20 coeffs) + delta + delta-delta stats
- temporal binning for MFCC, delta, and log-mel trajectories
- log-mel mean/std, spectral contrast, spectral flux/rolloff/bandwidth
- robust extrema and energy dynamics

Why:

- richer spectral + temporal cues improved separability of acoustically similar words

Rejected:

- CMVN, pitch/F0 and autocorrelation pitch proxy (regressions or no gain)

### 5.3 Projection and Optimizer Improvements (to ~82%)

Kept:

- PCA before LDA with shrinkage LDA (`solver='eigen'`, `shrinkage='auto'`)
- affine collapse of PCA+LDA into one transform so AVR inference stayed unchanged
- LinearSVC-guided template selection
- joint projection/template optimization with Adam and cosine annealing
- 3-seed averaging for stability
- feature warper on training side to improve LDA conditioning

Why:

- improved class separation and prototype placement while keeping deployment-compatible inference

Rejected:

- NCA, triplet loss, focal loss, class-weighted loss, hard-negative mining variants, whitening variants, several epoch sweeps (all regressed)

### 5.4 Augmentation and Noise Robustness (to 89.7%)

Kept:

- speed perturbation, time shift, synthetic reverb
- white noise augmentation
- hard-class targeted augmentation for weak classes
- real background noise mixing strategy (50/50 clean-noisy ratio, SIR 18-30 dB)

Why:

- strongest gains came from realistic noise variability and targeted class strengthening

Rejected:

- MixUp, SpecAugment masking, curriculum-noise variants, 70/30 noise strategy, and other variants that regressed

### 5.5 Frontend and Cepstral Refinement

Kept:

- PCEN frontend (major single-step gain)
- LPCC experimentation then LFCC replacement in final set
- cepstral liftering

Why:

- improved robustness and better consonant-region discrimination under the same deployment budget

### 5.6 Finalization

Kept:

- train+val short final retrain pass
- locked final config and regenerated deployment header

## 6) Accuracy Timeline

| Stage | Key Change | Accuracy |
|---|---|---:|
| Baseline | 14 handcrafted features + LDA + 1-NN | 38.6% |
| Feature upgrade | MFCC + delta replacing coarse time-domain stack | ~58.7% |
| Feature expansion | Delta-delta + temporal bins + log-mel descriptors | ~66.0% |
| LDA improvement | PCA(60) + shrinkage LDA | ~72.7% |
| Template improvement | LinearSVC-guided template placement | ~77.9% |
| Augmentation stack | Speed + noise + shift + reverb | ~81.7% |
| Optimizer tuning | Soft-min temperature + cosine annealing | ~82.4% |
| PCA upgrade | PCA(80) | ~82.5% |
| Final retrain pass | Short train+val continuation | ~82.6% |
| FeatureWarper | Gaussian histogram equalization | ~83.3% |
| LPCC added | +24 LPCC dimensions | ~85.6% |
| LPCC -> LFCC | Linear-scale cepstrum swap | ~86.0% |
| Spectral contrast | Added sub-band contrast descriptors | ~86.2% |
| Log-mel mean/std | Added filterbank mean+std stats | ~86.2% |
| 3-seed averaging | Ensemble average of 3 optimizer runs | **86.7%** |
| PCEN frontend | Per-channel energy normalization | **88.6% to 89.0%** |
| Hard-class augmentation | Extra variants for weak classes | **89.0%** |
| Real noise strategy (kept) | 50/50 real-noise mixing at moderate SIR | **89.7% (final)** |

`~` indicates approximate values observed during iterative runs.

### 6.1 Final Feature Breakdown (353-D)

- MFCC + delta + delta-delta statistics: 120
- Spectral stats (flux, rolloff, bandwidth): 6
- Temporal MFCC bins: 50
- Temporal delta bins: 40
- Temporal log-mel bins: 40
- Energy stats: 3
- Robust extrema descriptors: 6
- Log-mel mean + std: 48
- Spectral contrast: 12
- LFCC mean + std: 24
- Formant stats: 4

### 6.2 Performance Engineering

- Vectorized delta computation to remove Python frame loops
- Parallel augmentation with `ProcessPoolExecutor`
- Subsampling in costly fitting stages (for example affine collapse and SVC-guided template selection)
- Optional GPU optimizer path with CPU fallback

These changes improved iteration speed while preserving the same deployed inference structure.


### 6.3 Budget-Constrained Ideas Not Adopted

- Higher LDA dimensions and larger template counts when they exceeded flash or regressed
- Additional feature expansions that violated byte budget at final model size
- Some budget-safe feature variants that still regressed (for example entropy/flatness variants)

## 7) Why Deep Learning Was Not Used

Deep models can often achieve higher keyword-spotting accuracy on larger systems, but were incompatible with this deployment contract:

- fixed AVR inference geometry
- strict flash/RAM limits
- static header export requirement consumed by firmware

This is a deployment-driven decision, not a claim that deep learning is weaker.

## 8) What Would Improve with More Hardware

With larger memory and compute budgets, likely improvements include:

- neural KWS backends (for example DS-CNN/BC-ResNet class models)
- larger embeddings or higher template counts
- richer feature stacks and broader search space
- additional classical frontends under same constraints (for example RASTA-PLP, PNCC)
- potentially higher than current 89.7% under comparable evaluation protocol

## 9) Repository Files and Intent

- `main-pipeline.py`: training, evaluation, and header export pipeline
- `realtime_test.py`: PC AVR emulator, parity self-test, and live mic testing
- `classify_snippet.c`: fixed firmware inference function
- `word_templates.h`: generated deployable parameters in `PROGMEM`
- `code_used_to_convert_audio.py`: optional audio format normalization utility
- `data.zip`: local dataset package for reproducible experiments
- `Dataset-README.md`: upstream dataset documentation/attribution
- `LICENSE`: MIT license

## 10) Setup

Install dependencies:

```bash
pip install numpy scipy scikit-learn soundfile librosa sounddevice
```

Extract dataset at repository root:

```powershell
Expand-Archive -Path .\data.zip -DestinationPath . -Force
```

## 11) Commands and Usage

Train + evaluate + export `word_templates.h`:

```bash
python main-pipeline.py --mode all --data_dir .
```

Evaluate current artifacts only:

```bash
python main-pipeline.py --mode eval --data_dir .
```

Run parity self-test:

```bash
python realtime_test.py --test_mode
```

Run live microphone mode:

```bash
python realtime_test.py --duration=1.0
```

## 12) Typical Workflow

1. Extract and verify dataset folders.
2. Run `--mode all` to train/evaluate/export.
3. Run `realtime_test.py --test_mode` for parity validation.
4. Integrate `classify_snippet.c` + `word_templates.h` into AVR firmware.
5. Test on device.

## 13) Evaluation Protocol Note

The 89.7% result is tied to the deterministic fallback stratified split used consistently in this project.
Different split protocols can produce different absolute scores.
In practice, the official speaker-based protocol is typically harder, so scores are only comparable when the split definition is matched.

## 14) License and Data Attribution

- Code license: MIT (see `LICENSE`)
- Dataset terms and attribution: see `Dataset-README.md`
