# Validating longitudinal structural connectome change

Analysis code and derived data accompanying the manuscript

> **How much of a longitudinal structural connectome change is real? A validation study in a
> trigeminal neuralgia cohort**

## Contents

| Path | What it contains |
|---|---|
| `scripts/` | End-to-end pipeline: manifest retrieval, containerised MRtrix3/FSL preprocessing, connectome construction, phenotype derivation, statistics, figures |
| `src/tnwt/` | Analysis library: connectome handling, phenotypes, statistics, rotation nulls, permutation, plotting |
| `tests/` | Unit tests and end-to-end consistency checks |
| `config/analysis.yaml` | **Pre-specified decision rules** (D5-D12) referred to in the Methods |
| `data/` | Derived phenotype matrices, effect maps, sensitivity analyses, run summary |
| `MANIFEST.md` | SHA256 of every file |

## What the pipeline does

Starting from the public dataset OpenNeuro **ds005713**, the pipeline

1. processes diffusion, T1-, T2-weighted and resting-state data in containerised MRtrix3 and FSL;
2. derives six parcel-level phenotypes on the Schaefer 200-parcel, 7-network atlas: count-weighted
   and anisotropy-weighted connectivity, grey-white contrast, two myelin ratios, and
   structure-function coupling;
3. runs cross-sectional group comparisons with false-discovery-rate control, together with
   whole-brain equivalence testing (TOST) and JZS Bayes factors;
4. runs paired pre-/post-operative comparisons before and after within-participant normalisation;
5. applies the four checks described in the paper: change-versus-baseline correlation,
   count-versus-anisotropy metric contrast, shuffled pre-/post-operative pairing (1,000 draws),
   and split-half replication (200 random splits);
6. simulates power in three designs and reproduces Figures 1-4.

## What is not here

Individual-level imaging data are not redistributed, and no demographic or clinical table is
included. The source data remain available from OpenNeuro ds005713 (CC0). The `data/` folder holds
parcel-level phenotype matrices and effect maps only, indexed by the public BIDS identifiers of the
source dataset; these are 200-parcel summaries with no age, sex, pain or outcome variable attached,
and no participant can be identified from them.

Source files carry comments in Chinese, inherited from the development of this pipeline. All
identifiers, function names and outputs are in English.

## Reproducing the analysis

```
python -m pip install -r requirements.txt   # or use your own environment
python scripts/fetch_manifest.py            # freeze the public file manifest
python scripts/run_a1_pipeline.py --stage all
python scripts/derive_phenotypes.py
python scripts/run_statistics.py
python scripts/make_figures.py
```

Container images and exact software versions are recorded in the Methods and in
`config/analysis.yaml`.

## Licence

Code: MIT (see `LICENSE`). Derived data: CC BY 4.0.
