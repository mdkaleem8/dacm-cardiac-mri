# DACM: degradation-aware conditioning for cine cardiac MRI segmentation

Code for the paper *Does Telling the Network How an Image Is Degraded Help? A Controlled Study of
Degradation-Aware Conditioning for Cine Cardiac MRI Segmentation*.

Author: Md. Kaleem, Department of Artificial Intelligence and Data Science, Faculty of Science and
Technology (IcfaiTech), ICFAI Foundation for Higher Education, Hyderabad 501203, Telangana, India
(mdkaleem@ifheindia.org; ORCID [0009-0009-5643-299X](https://orcid.org/0009-0009-5643-299X)).

A small estimator predicts the type and severity of five degradations (respiratory motion, noise,
blur, bias field, undersampling) in each slice. A conditioning module feeds the estimate to an
unchanged segmentation backbone (U-Net or Swin-UNETR) through FiLM, attached by forward hooks.
The study compares it with the same backbone trained on identical degraded images, and with
zero-input and shuffled-input versions of the module.

## Setup

```
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

All experiments ran on one NVIDIA RTX 3060 (12 GB).

## Data

The datasets are not included; obtain them from their organisers.

* **ACDC**: training and test sets, placed in `../data/ACDC/ACDC/database`.
* **CMRxMotion** (training release, external test only): Synapse project syn28503327. Downloads need
  membership of the challenge participants team; `scripts/download_cmrxmotion.py` reads a Synapse
  token from the `SYNAPSE_AUTH_TOKEN` environment variable or from `--token_file`.
* **M&Ms** (open dataset, external test only): available from the challenge organisers.

## Running

The PowerShell scripts at the top level run the experiments in order and skip steps whose outputs
already exist under `work/`:

| Script | What it runs |
|---|---|
| `run_pipeline.ps1` | preparation, severity calibration, estimator, fixed test set, default / augmentation-only / DACM for both backbones, evaluation |
| `run_followups.ps1`, `run_seeds*.ps1`, `run_revision_*.ps1` | DACM (norm), seeds, zero-input and shuffled controls, probes, nnU-Net and Tent baselines |
| `run_mechanism.ps1` | stress tests (reduced training set, capped severity) |
| `run_external_swin.ps1` | CMRxMotion and M&Ms evaluation |
| `run_review.ps1`, `run_review2.ps1` | Swin-UNETR with affine norms, type-only conditioning, remaining seeds |
| `make_review_results.ps1` | every table, figure input and number of the paper, then compiles it |

Main entry points in `scripts/`:

* `prepare_acdc.py`, `calibrate.py`, `build_testset.py`: data cache, severity calibration (Eq. 2), degraded test set
* `train_estimator.py`, `train_seg.py`: Stage 1 and Stage 2 (`--mode default|aug|dacm`, `--placement stage|norm`,
  `--zero_cond`, `--shuffle_cond`, `--norm_affine`, `--cond_part type`)
* `evaluate.py`, `eval_slices.py`, `eval_cmrx.py`, `eval_mnms.py`, `probe_val.py`, `tent_eval.py`: evaluation
* `make_tables.py`, `stats_revision.py`, `stats_structures.py`, `stats_external.py`, `stats_review.py`: statistics and LaTeX output

The package `dacm/` holds the degradation simulator (`degradations.py`), the backbones and estimator
(`models.py`), the conditioning module (`conditioning.py`), data loading and metrics.

No result in the paper is typed by hand: the statistics scripts write every table and number macro.
