# Cluster browser

Interactive tool to inspect **every simulation** in a chosen cluster.

## Run

```cmd
cd cluster_analysis
run_browse_cluster.cmd
```

Or non-interactive:

```cmd
venv\python.exe src\browse_cluster.py --method gmm --cluster 9
```

## Output layout (per selection)

```
cluster_browser/<method>/cluster_<id>/
  gallery.html      ← open this in a browser
  rgb/              ← AE difference PNGs (copied for sharing)
  f_sim/            ← contour plots from .nc (v_parallel vs mu)
```

Each card shows **RGB diff** and **f_sim** side by side, with normal/anomaly tag and Maxwellian score.

Algorithms: `gmm`, `agglomerative`, `hdbscan`, `deepdpm`

Large clusters (e.g. HDBSCAN noise) prompt for confirmation before generating hundreds of plots.
