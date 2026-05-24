"""
Run clustering + validation pipelines with logs under logs/YYYYMMDD-<method>/.

Usage (from cluster_analysis repo root):
  ..\\..\\..\\..\\..\\..\\DCEC\\torch_DCEC_RGB\\venv\\Scripts\\python.exe src/run_clustering_logged.py --methods gmm,agglomerative,hdbscan
  ..\\..\\..\\..\\..\\..\\DCEC\\torch_DCEC_RGB\\venv\\Scripts\\python.exe src/run_clustering_logged.py --methods robustness --assignments gmm

Methods:
  gmm           - GMM on 20D AE+physics, BIC/silhouette curves, cluster_assignments.csv
  agglomerative - Ward agglomerative, silhouette curve, assignments CSV
  hdbscan       - density clustering (noise label -1)
  deepdpm       - prepare 20D + DeepDPM + export (long)
  robustness    - UMAP+PaCMAP combined sweep + kNN (needs --assignments)
  score3d       - interactive 3D Maxwellian vs cluster (needs --assignments)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from paths_config import analysis_out, dcec_rgb, load_config, log_dir, repo_root, script, venv_python


def _run(cmd: list, log_path: Path, cwd: Path | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    return proc.returncode


def run_gmm(cfg, out_method: str) -> int:
    ld = log_dir("gmm_physics")
    out_dcec = dcec_rgb() / "reports" / "cluster_visualizations"
    out_dcec.mkdir(parents=True, exist_ok=True)
    py = str(venv_python())
    cmd = [
        py,
        "-u",
        str(script("cluster_ae_physics_gmm.py")),
        "--method",
        "gmm",
        "--k_min",
        str(cfg["k_min"]),
        "--k_max",
        str(cfg["k_max"]),
        "--out_dir",
        str(out_dcec),
    ]
    rc = _run(cmd, ld / "run.log", cwd=dcec_rgb())
    for name in ("gmm_physics_bic_vs_k.png", "gmm_physics_silhouette_vs_k.png", "cluster_assignments.csv"):
        src = out_dcec / name
        if src.is_file():
            shutil.copy2(src, analysis_out(out_method) / name)
    return rc


def run_agglomerative(cfg, out_method: str) -> int:
    ld = log_dir("agglomerative_ward")
    out_dcec = dcec_rgb() / "reports" / "cluster_visualizations"
    fk = cfg.get("agglomerative_final_k")
    py = str(venv_python())
    cmd = [
        py,
        "-u",
        str(script("cluster_ae_physics_gmm.py")),
        "--method",
        "agglomerative",
        "--k_min",
        str(cfg["k_min"]),
        "--k_max",
        str(cfg["k_max"]),
        "--out_dir",
        str(out_dcec),
    ]
    if fk:
        cmd.extend(["--final_k", str(fk)])
    rc = _run(cmd, ld / "run.log", cwd=dcec_rgb())
    for p in out_dcec.glob("agglomerative_*_physics_silhouette_vs_k.png"):
        shutil.copy2(p, analysis_out(out_method) / p.name)
    for p in out_dcec.glob("cluster_assignments_agglomerative_*.csv"):
        shutil.copy2(p, analysis_out(out_method) / p.name)
    return rc


def run_hdbscan(cfg, out_method: str) -> int:
    ld = log_dir("hdbscan_physics")
    out_dcec = dcec_rgb() / "reports" / "cluster_visualizations_hdbscan"
    out_dcec.mkdir(parents=True, exist_ok=True)
    py = str(venv_python())
    cmd = [py, "-u", str(script("cluster_ae_physics_hdbscan.py")), "--out_dir", str(out_dcec)]
    rc = _run(cmd, ld / "run.log", cwd=dcec_rgb())
    for name in (
        "cluster_assignments.csv",
        "ae_physics_hdbscan_cluster_sizes.png",
        "ae_physics_hdbscan_umap.png",
    ):
        src = out_dcec / name
        if src.is_file():
            shutil.copy2(src, analysis_out(out_method) / name)
    return rc


def run_deepdpm(cfg, out_method: str) -> int:
    ld = log_dir("deepdpm_combined20d")
    dcec = Path(cfg["dce_root"])
    py = str(venv_python())
    rc = _run(
        [py, "-u", str(dcec / "prepare_deepdpm_data.py"), "--representation", "combined"],
        ld / "01_prepare.log",
        cwd=dcec,
    )
    if rc != 0:
        return rc
    deepdpm = Path(cfg["deepdpm_dir"])
    rc = _run(
        [
            py,
            "-u",
            "DeepDPM.py",
            "--dataset",
            "custom",
            "--dir",
            cfg["deepdpm_data"],
            "--offline",
            "--init_k",
            str(cfg["deepdpm_init_k"]),
            "--NIW_prior_nu",
            str(cfg["deepdpm_niw_nu"]),
            "--max_epochs",
            str(cfg["deepdpm_max_epochs"]),
            "--transform_input_data",
            "None",
            "--batch-size",
            "64",
            "--gpus",
            "0",
        ],
        ld / "02_deepdpm_train.log",
        cwd=deepdpm,
    )
    if rc != 0:
        return rc
    out_csv = dcec_rgb() / "reports/cluster_visualizations/cluster_assignments_deepdpm_combined20d.csv"
    rc = _run(
        [py, "-u", str(dcec / "export_deepdpm_assignments.py"), "--data_dir", cfg["deepdpm_data"], "--out_csv", str(out_csv)],
        ld / "03_export.log",
        cwd=dcec,
    )
    if out_csv.is_file():
        shutil.copy2(out_csv, analysis_out(out_method) / out_csv.name)
    return rc


def run_robustness(cfg, assignments_rel: str, out_method: str) -> int:
    ld = log_dir("umap_pacmap_robustness")
    out_root = dcec_rgb() / "reports" / "umap_pacmap_robustness"
    assign = dcec_rgb() / assignments_rel
    py = str(venv_python())
    cmd = [
        py,
        "-u",
        str(script("run_umap_pacmap_robustness_sweep.py")),
        "--out_dir",
        str(out_root),
        "--assignments",
        str(assign),
        "--representation",
        "combined",
    ]
    rc = _run(cmd, ld / "run.log", cwd=dcec_rgb())
    sweeps = sorted(out_root.glob("sweep_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if sweeps:
        sweep = sweeps[0]
        dest = analysis_out(out_method) / sweep.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(sweep, dest)
        (analysis_out(out_method) / "LATEST_SWEEP.txt").write_text(sweep.name, encoding="utf-8")
    return rc


def run_score3d(cfg, assignments_rel: str, out_method: str) -> int:
    ld = log_dir("cluster_score_3d")
    out_dcec = dcec_rgb() / "reports" / "cluster_visualizations"
    assign = dcec_rgb() / assignments_rel
    maxwellian = out_dcec / "maxwellian_scores_combined.csv"
    tag = Path(assignments_rel).stem.replace("cluster_assignments", "").strip("_") or "clusters"
    out_name = f"cluster_score_anomaly_3d{tag}.png"
    py = str(venv_python())
    cmd = [
        py,
        "-u",
        str(script("visualize_cluster_score_anomaly_3d.py")),
        "--cluster_assignments",
        str(assign),
        "--maxwellian_scores",
        str(maxwellian),
        "--out_dir",
        str(out_dcec),
        "--out_name",
        out_name,
    ]
    rc = _run(cmd, ld / "run.log", cwd=dcec_rgb())
    for fname in (out_name, out_name.replace(".png", "_interactive.html")):
        src = out_dcec / fname
        if src.is_file():
            shutil.copy2(src, analysis_out(out_method) / fname)
    return rc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        default="gmm,agglomerative,hdbscan",
        help="Comma-separated: gmm,agglomerative,hdbscan,deepdpm,robustness,score3d",
    )
    parser.add_argument(
        "--assignments",
        default="reports/cluster_visualizations/cluster_assignments.csv",
        help="Relative to torch_DCEC_RGB; used for robustness + score3d",
    )
    args = parser.parse_args()
    cfg = load_config()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    summary = repo_root() / "logs" / "run_summary.txt"
    lines = []

    runners = {
        "gmm": lambda: run_gmm(cfg, "gmm_bic"),
        "agglomerative": lambda: run_agglomerative(cfg, "agglomerative_ward"),
        "hdbscan": lambda: run_hdbscan(cfg, "hdbscan"),
        "deepdpm": lambda: run_deepdpm(cfg, "deepdpm_combined20d"),
        "robustness": lambda: run_robustness(cfg, args.assignments, "robustness_umap_pacmap"),
        "score3d": lambda: run_score3d(cfg, args.assignments, "score3d_maxwellian"),
    }

    for m in methods:
        if m not in runners:
            print(f"Unknown method: {m}", file=sys.stderr)
            continue
        print(f"\n=== Running {m} ===")
        rc = runners[m]()
        lines.append(f"{m}: exit_code={rc}")
        print(f"  -> exit {rc}")

    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSummary: {summary}")


if __name__ == "__main__":
    main()
