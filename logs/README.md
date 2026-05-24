# Logs

Each run of `src/run_clustering_logged.py` writes:

```
logs/YYYYMMDD-<method>/run.log
```

Examples:

- `20260515-gmm_physics/`
- `20260515-agglomerative_ward/`
- `20260515-hdbscan_physics/`
- `20260515-umap_pacmap_robustness/`
- `20260515-cluster_score_3d/`

DeepDPM uses multiple files: `01_prepare.log`, `02_deepdpm_train.log`, `03_export.log`.

A summary line per invocation is appended to `logs/run_summary.txt`.
