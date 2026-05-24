# GENE electron cluster analysis

Standalone repository for **unsupervised clustering experiments** on GENE electron velocity_space (vsp) simulations. This repo holds analysis scripts, logged runs, and curated outputs so the main [TPED](https://github.com/drdrhatch/TPED) codebase stays focused on validation tooling.

**Dataset:** 1,116 simulations — RGB difference images (`_diff.png`) plus matching NetCDF physics files.

**Goal:** Compare non–k-means clustering methods on a shared **20D feature space**, validate partitions with embeddings and physics scores, and document what did and did not work for the team.

---

## Project overview

```text
  RGB diff images (DCEC/all_rgb)
           │
           ▼
  Convolutional autoencoder (13D latent)  ──┐
           │                                 ├──► StandardScaled 20D features
  Physics scalars from .nc (7D)  ───────────┘
           │
           ▼
  Clustering: GMM · Agglomerative · HDBSCAN · DeepDPM
           │
           ├── Assignment CSVs + metric curves (BIC / silhouette)
           ├── UMAP + PaCMAP robustness sweeps (combined panels, kNN)
           ├── Maxwellianity coloring (score v7, sensitivity 500)
           ├── TDA: persistence, Kepler Mapper, filtration demo
           └── Cluster browser: per-cluster RGB + f(v∥, μ) galleries
```

**Where code runs:** Clustering and embedding scripts live under `DCEC/torch_DCEC_RGB/` on your machine. This repository **orchestrates** those scripts (`src/`, `*.cmd`) and stores **results, logs, and manifests** under `analysis/outputs/`. Edit paths in `config.yaml` if your layout differs.

**Related repos / paths (local):**

| Piece | Location |
|-------|----------|
| AE training / clustering scripts | `D:/GENE_simulation_AI/DCEC/torch_DCEC_RGB/` |
| DeepDPM pipeline | `D:/GENE_simulation_AI/DCEC/run_deepdpm_combined20d_pipeline.cmd` |
| Maxwellian score (v7) | `TPED/.../maxwellian_score_calc/score_v7_Cursor.py` |
| Raw `.nc` (normal / anomaly) | `simulation_normals/electron`, `simulation_anomalies/electron/...` |

---

## Feature space (all methods)

| Component | Source |
|-----------|--------|
| 13D latent | `CAE_bn3_AE_lat13.pt` on RGB difference images |
| 7D physics | `temp_e, dens_e, omn_e, omt_e, beta, rhostar, x0` from matching `.nc` |
| Scaling | `StandardScaler` on physics, then on full 20D vector |

---

## Methods compared

| Method | Role | Result on this dataset |
|--------|------|-------------------------|
| **GMM + BIC** | Probabilistic; choose k by BIC | **k = 12** |
| **Agglomerative (Ward)** | Hierarchical; silhouette vs k | **k = 8** |
| **HDBSCAN** | Density clustering; noise label `-1` | Many small clusters + ~47% noise (default `min_cluster_size`) |
| **DeepDPM** | Deep nonparametric clustering (20D) | **13** clusters (`init_k=12`) |
| **Robustness sweep** | ~24 combined UMAP+PaCMAP configs + kNN gaps | Per-method `robustness_*/sweep_*/index.html` |
| **Maxwellian score** | Physics-based scalar for coloring / 3D views | v7, sensitivity **500** |

We keep **combined** UMAP|PaCMAP panels only (not separate single-method embedding folders).

---

## Repository layout

```text
cluster_analysis/
  README.md                 # this file
  config.yaml               # paths + hyperparameters
  run_sync_results.cmd      # copy DCEC reports → analysis/outputs
  run_full_pipeline.cmd     # full validation pipeline
  run_browse_cluster.cmd    # interactive per-cluster gallery
  src/                      # orchestration Python
  analysis/
    outputs/                # figures, CSVs, HTML galleries (by method)
    manifest.json           # index of key artifacts (from sync)
  logs/                     # YYYYMMDD-<method>/run.log
  data/processed/           # shared CSVs (e.g. Maxwellian scores)
```

See `analysis/README.md` for output folder details.

---

## Quick start (Windows)

Clone this repo, then point `config.yaml` at your local `DCEC` and simulation directories.

```cmd
cd D:\GENE_simulation_AI\cluster_analysis
```

**1. Sync existing DCEC outputs (fast)**

```cmd
run_sync_results.cmd
```

**2. Full pipeline** (Maxwellian embed, TDA, per-method robustness + 3D + cluster-colored maps)

```cmd
run_full_pipeline.cmd
```

Skip re-clustering if assignments are already current:

```cmd
run_full_pipeline_skip_clustering.cmd
```

**3. Browse one cluster** (RGB diff + f_sim contours)

```cmd
run_browse_cluster.cmd
```

**4. DeepDPM only** (long; run from DCEC)

```cmd
D:\GENE_simulation_AI\DCEC\run_deepdpm_combined20d_pipeline.cmd
```

Python venv: `D:\GENE_simulation_AI\DCEC\torch_DCEC_RGB\venv`  
(`pip install pyyaml` if orchestration scripts fail on import)

---

## Where to open results

| What | Path |
|------|------|
| Index | `analysis/manifest.json` |
| GMM / agglomerative / HDBSCAN / DeepDPM assignments | `analysis/outputs/<method>/` |
| UMAP+PaCMAP galleries | `analysis/outputs/robustness_*/sweep_*/index.html` |
| Cluster-colored embeddings | `analysis/outputs/embed_clusters_*/` |
| Maxwellian-colored embeddings | `analysis/outputs/embed_maxwellian_combined/` |
| Persistence / Mapper | `analysis/outputs/tda_exploratory/` (or `z_tda_exploratory/`) |
| Per-cluster sim browser | `analysis/outputs/cluster_browser/<method>/cluster_*/gallery.html` |

**Large files not in Git** (GitHub 100 MB limit): filtration slider HTML, some Plotly 3D HTML, robustness panel PNGs, cluster_browser `rgb/` / `f_sim/` copies. Regenerate locally or read from `DCEC/reports/umap_pacmap_validation/`. See `.gitignore`.

---

## Findings (summary)

- **GMM (k=12)** and **agglomerative (k=8)** disagree on partition count — compare assignments before treating either as ground truth.
- **HDBSCAN** at default settings is **over-fragmented** here; use as exploratory unless `min_cluster_size` is retuned.
- **Robustness:** look for groups that persist across UMAP/PaCMAP seeds and configs; do not tune k only until plots “look right.”
- **kNN gap ratios** in each sweep folder help judge within-cluster cohesion vs separation.
- **Next science step:** test whether clusters correspond to real physical modes (parameters + Maxwellian scores per cluster), not embedding aesthetics alone.

---

## Logs

Each orchestrated run writes stdout under `logs/YYYYMMDD-<method>/`. Pipeline summary: `logs/pipeline_summary.txt`.

---

## Configuration

All paths and hyperparameters: `config.yaml` (read by `src/paths_config.py`).

---

## Citation / context

Part of the GENE simulation AI / VSP validator workflow. Main application code remains in TPED; this repository is the **analysis log and artifact store** for clustering comparisons.
