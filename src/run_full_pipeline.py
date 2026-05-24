"""
Full cluster analysis pipeline:
  1) Re-run GMM, agglomerative, HDBSCAN (optional: skip with --skip-clustering)
  2) Per method (gmm, agglomerative, hdbscan, deepdpm): robustness sweep, 3D score plot, cluster-colored UMAP|PaCMAP
  3) Once: Maxwellian-colored UMAP|PaCMAP, TDA (persistence / Mapper / filtration slider)
  4) sync_existing_results.py

Usage:
  python src/run_full_pipeline.py
  python src/run_full_pipeline.py --skip-clustering --methods deepdpm
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from method_registry import EMBED_CLUSTER_GLOB, METHODS
from paths_config import analysis_out, dcec_rgb, load_config, log_dir, repo_root, script, venv_python

# Import runners from logged script
from run_clustering_logged import (
    _run,
    run_agglomerative,
    run_gmm,
    run_hdbscan,
    run_robustness,
    run_score3d,
)


def run_embed_clusters(cfg, assignments_rel: str, out_method: str) -> int:
    """UMAP + PaCMAP side-by-side colored by cluster (no TDA)."""
    ld = log_dir(f"embed_clusters_{out_method}")
    rgb = dcec_rgb()
    out_sub = rgb / "reports" / "cluster_visualizations" / out_method
    out_sub.mkdir(parents=True, exist_ok=True)
    py = str(venv_python())
    assign = rgb / assignments_rel
    cmd = [
        py,
        "-u",
        str(script("visualize_latent_umap_pacmap.py")),
        "--dataset_path",
        cfg["dataset_rgb"],
        "--model",
        cfg["model_ae"],
        "--latent_dim",
        str(cfg["latent_dim"]),
        "--representation",
        "combined",
        "--physics_vars",
        cfg["physics_vars"],
        "--normal_dir",
        cfg["normal_dir"],
        "--anomaly_dir",
        cfg["anomaly_dir"],
        "--assignments",
        str(assign),
        "--out_dir",
        str(out_sub),
        "--random_state",
        "42",
        "--no_tda",
    ]
    rc = _run(cmd, ld / "run.log", cwd=rgb)
    dest = analysis_out(out_method)
    for name in EMBED_CLUSTER_GLOB:
        src = out_sub / name
        if src.is_file():
            shutil.copy2(src, dest / name)
    return rc


def run_maxwellian_embed(cfg) -> int:
    ld = log_dir("embed_maxwellian_combined")
    out = analysis_out("embed_maxwellian_combined")
    py = str(venv_python())
    cmd = [
        py,
        "-u",
        str(script("visualize_latent_umap_pacmap_maxwellian_v7.py")),
        "--dataset_path",
        cfg["dataset_rgb"],
        "--model",
        cfg["model_ae"],
        "--latent_dim",
        str(cfg["latent_dim"]),
        "--representation",
        "combined",
        "--physics_vars",
        cfg["physics_vars"],
        "--normal_dir",
        cfg["normal_dir"],
        "--anomaly_dir",
        cfg["anomaly_dir"],
        "--out_dir",
        str(out),
        "--random_state",
        "42",
        "--maxwellian_score_py",
        cfg["score_v7_py"],
        "--maxwellian_sensitivity",
        str(cfg["maxwellian_sensitivity"]),
    ]
    rc = _run(cmd, ld / "run.log", cwd=dcec_rgb())
    data_proc = repo_root() / "data" / "processed"
    data_proc.mkdir(parents=True, exist_ok=True)
    src_csv = out / "maxwellian_scores_combined.csv"
    if src_csv.is_file():
        shutil.copy2(src_csv, data_proc / src_csv.name)
        shutil.copy2(src_csv, dcec_rgb() / "reports/cluster_visualizations/maxwellian_scores_combined.csv")
    return rc


def run_tda_exploratory(cfg, assignments_rel: str) -> int:
    """Persistence (birth/death), Kepler Mapper, filtration PaCMAP slider."""
    ld = log_dir("tda_exploratory")
    out = analysis_out("tda_exploratory")
    out.mkdir(parents=True, exist_ok=True)
    rgb = dcec_rgb()
    py = str(venv_python())
    assign = rgb / assignments_rel
    cmd = [
        py,
        "-u",
        str(script("visualize_latent_umap_pacmap.py")),
        "--dataset_path",
        cfg["dataset_rgb"],
        "--model",
        cfg["model_ae"],
        "--latent_dim",
        str(cfg["latent_dim"]),
        "--representation",
        "combined",
        "--physics_vars",
        cfg["physics_vars"],
        "--normal_dir",
        cfg["normal_dir"],
        "--anomaly_dir",
        cfg["anomaly_dir"],
        "--assignments",
        str(assign),
        "--out_dir",
        str(out),
        "--random_state",
        "42",
        "--filtration_demo",
        "pacmap",
        "--filtration_demo_slider",
        "--filtration_demo_frames",
        "36",
        "--tda_evolution_frames",
        "12",
        "--tda_rips_max_points",
        "600",
    ]
    return _run(cmd, ld / "run.log", cwd=rgb)


def run_sync() -> int:
    py = str(venv_python())
    sync_py = repo_root() / "src" / "sync_existing_results.py"
    return subprocess.call([py, "-u", str(sync_py)], cwd=str(repo_root()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        default="gmm,agglomerative,hdbscan,deepdpm",
        help="Comma-separated method keys from method_registry",
    )
    parser.add_argument("--skip-clustering", action="store_true")
    parser.add_argument("--skip-maxwellian", action="store_true", help="Reuse existing Maxwellian CSV/plots")
    parser.add_argument("--skip-tda", action="store_true", help="Reuse existing persistence/filtration artifacts")
    parser.add_argument("--skip-sync", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    summary_path = repo_root() / "logs" / "pipeline_summary.txt"
    lines: list[str] = []

    if not args.skip_clustering:
        cluster_runners = {
            "gmm": lambda: run_gmm(cfg, "gmm_bic"),
            "agglomerative": lambda: run_agglomerative(cfg, "agglomerative_ward"),
            "hdbscan": lambda: run_hdbscan(cfg, "hdbscan"),
        }
        for key in methods:
            m = METHODS.get(key)
            if not m or not m.get("run_cluster"):
                continue
            rc_name = m["run_cluster"]
            print(f"\n=== Clustering: {key} ===")
            rc = cluster_runners[rc_name]()
            lines.append(f"cluster_{key}: {rc}")
            print(f"  -> exit {rc}")

    if not args.skip_maxwellian:
        print("\n=== Maxwellian UMAP|PaCMAP (shared) ===")
        rc = run_maxwellian_embed(cfg)
        lines.append(f"maxwellian_embed: {rc}")
        print(f"  -> exit {rc}")

    if not args.skip_tda:
        print("\n=== TDA: persistence, Mapper, filtration slider (shared) ===")
        rc = run_tda_exploratory(cfg, METHODS["gmm"]["assignments_rel"])
        lines.append(f"tda_exploratory: {rc}")
        print(f"  -> exit {rc}")

    for key in methods:
        m = METHODS.get(key)
        if not m:
            print(f"Unknown method: {key}", file=sys.stderr)
            continue
        assign_rel = m["assignments_rel"]
        assign_path = dcec_rgb() / assign_rel
        if not assign_path.is_file():
            print(f"\n=== SKIP {key}: missing {assign_path} ===")
            lines.append(f"{key}: skipped (no assignments)")
            continue

        print(f"\n=== Validation: {key} — robustness ===")
        rc1 = run_robustness(cfg, assign_rel, m["robustness_dir"])
        lines.append(f"robustness_{key}: {rc1}")
        print(f"  -> exit {rc1}")

        print(f"\n=== Validation: {key} — score3d ===")
        rc2 = run_score3d(cfg, assign_rel, m["score3d_dir"])
        lines.append(f"score3d_{key}: {rc2}")
        print(f"  -> exit {rc2}")

        print(f"\n=== Validation: {key} — cluster UMAP|PaCMAP ===")
        rc3 = run_embed_clusters(cfg, assign_rel, m["embed_dir"])
        lines.append(f"embed_{key}: {rc3}")
        print(f"  -> exit {rc3}")

    if not args.skip_sync:
        print("\n=== Sync to analysis/outputs ===")
        rc = run_sync()
        lines.append(f"sync: {rc}")
        print(f"  -> exit {rc}")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nPipeline summary: {summary_path}")


if __name__ == "__main__":
    main()
