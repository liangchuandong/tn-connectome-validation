# Validating longitudinal structural connectome change: analysis code and derived data

Code and derived data accompanying the manuscript

> **How much of a longitudinal structural connectome change is real? A validation study in a
> trigeminal neuralgia cohort**

## What is here

| 目录 | 内容 |
|---|---|
| `scripts/` | 端到端管线：OpenNeuro 清单获取、MRtrix3/FSL 容器化预处理、连接组构建、表型派生、统计、作图 |
| `src/tnwt/` | 分析库：连接组、表型、统计、旋转零模型、置换、作图 |
| `tests/` | 单元测试与端到端一致性检查 |
| `config/analysis.yaml` | **预先设定的决策规则**（D5–D12），论文方法中引用 |
| `data/` | 派生表型矩阵、效应图、敏感性分析结果、运行摘要 |
| `MANIFEST.md` | 每个文件的 SHA256 |

## What the pipeline does

Starting from the public dataset OpenNeuro **ds005713**, the pipeline

1. processes diffusion, T1-, T2-weighted and resting-state data in containerised MRtrix3 and FSL;
2. derives six parcel-level phenotypes on the Schaefer 200-parcel, 7-network atlas
   (count-weighted and anisotropy-weighted connectivity, grey-white contrast, two myelin ratios,
   structure-function coupling);
3. runs cross-sectional group comparisons with false-discovery-rate control, and whole-brain
   equivalence testing (TOST) with JZS Bayes factors;
4. runs paired pre-/post-operative comparisons before and after within-participant normalisation;
5. applies the four checks described in the paper: change-versus-baseline correlation,
   count-versus-anisotropy metric contrast, shuffled pre-/post-operative pairing (1,000 draws),
   and split-half replication (200 random splits);
6. simulates power in three designs and reproduces Figures 1-4.

## What is not here

Individual-level imaging data are not redistributed, and no demographic or clinical table is
included. The source data remain available from OpenNeuro ds005713 (CC0). The `data/` folder holds
parcel-level phenotype matrices and effect maps only, indexed by the public BIDS identifiers of
the source dataset; these are 200-parcel summaries with no age, sex, pain or outcome variable
attached, and no participant can be identified from them.

## Reproducing the analysis

```
python -m pip install -r requirements.txt   # or use your own environment
python scripts/fetch_manifest.py            # freeze the public file manifest
python scripts/run_a1_pipeline.py --stage all
python scripts/derive_phenotypes.py
python scripts/run_statistics.py
python scripts/make_figures.py
```

Container images and exact versions are recorded in the Methods and in `config/analysis.yaml`.

## Licence

Code: MIT (see `LICENSE`). Derived data: CC BY 4.0.
