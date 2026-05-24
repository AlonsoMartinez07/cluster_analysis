"""
Copy artifacts from DCEC into analysis/outputs/ and write manifest.json.
Picks newest robustness sweep per assignment CSV when multiple exist.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from method_registry import EMBED_CLUSTER_GLOB, METHODS, SHARED_ARTIFACTS
from paths_config import analysis_out, dcec_rgb, repo_root

RGB = dcec_rgb()
ROBUST = RGB / "reports" / "umap_pacmap_robustness"
CV = RGB / "reports" / "cluster_visualizations"
CV_H = RGB / "reports" / "cluster_visualizations_hdbscan"
DCEC_VALIDATION = Path("D:/GENE_simulation_AI/DCEC/reports/umap_pacmap_validation")


def _copy(src: Path, dest_dir: Path, name: str | None = None) -> bool:
    if not src.is_file():
        return False
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / (name or src.name))
    return True


def _copy_glob(src_dir: Path, dest_dir: Path, names: list[str]) -> list[str]:
    copied = []
    for name in names:
        if _copy(src_dir / name, dest_dir, name):
            copied.append(name)
    return copied


# Known sweep folders per method (updated when pipeline creates new sweeps)
PREFERRED_SWEEP: dict[str, str] = {
    "robustness_gmm": "sweep_20260512_160036",
    "robustness_agglomerative_k8": "sweep_20260512_160228",
    "robustness_hdbscan": "sweep_20260512_225849",
    "robustness_deepdpm": "sweep_20260521_123100",
}


def _resolve_sweep(label: str) -> Path | None:
    latest_txt = analysis_out(label) / "LATEST_SWEEP.txt"
    if latest_txt.is_file():
        name = latest_txt.read_text(encoding="utf-8").strip()
        p = ROBUST / name
        if p.is_dir():
            return p
    preferred = PREFERRED_SWEEP.get(label)
    if preferred:
        p = ROBUST / preferred
        if p.is_dir():
            return p
    if not ROBUST.is_dir():
        return None
    candidates = sorted(ROBUST.glob("sweep_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _sync_robustness(label: str, assign_rel: str, manifest: dict) -> None:
    assign = RGB / assign_rel
    sweep = _resolve_sweep(label)
    if sweep is None:
        return
    dest = analysis_out(label) / sweep.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(sweep, dest)
    (analysis_out(label) / "assignments_source.txt").write_text(str(assign), encoding="utf-8")
    manifest[label] = {
        "sweep_folder": sweep.name,
        "index_html": f"{sweep.name}/index.html",
        "assignments": assign_rel,
    }


def main():
    manifest: dict = {}

    # GMM
    gmm = analysis_out("gmm_bic")
    for f in ("gmm_physics_bic_vs_k.png", "gmm_physics_silhouette_vs_k.png", "cluster_assignments.csv"):
        _copy(CV / f, gmm)
    manifest["gmm_bic"] = {"final_k_bic": 12, "assignments": "cluster_assignments.csv"}

    # Agglomerative
    agg = analysis_out("agglomerative_ward")
    for p in CV.glob("agglomerative_*"):
        _copy(p, agg)
    _copy(CV / "cluster_assignments_agglomerative_ward_k8.csv", agg)
    manifest["agglomerative_ward"] = {
        "silhouette_best_k": 8,
        "assignments": "cluster_assignments_agglomerative_ward_k8.csv",
    }

    # HDBSCAN
    hdb = analysis_out("hdbscan")
    for f in ("cluster_assignments.csv", "ae_physics_hdbscan_umap.png", "ae_physics_hdbscan_cluster_sizes.png"):
        _copy(CV_H / f, hdb)
    manifest["hdbscan"] = {"note": "density clustering; noise label -1"}

    # DeepDPM
    dpm = analysis_out("deepdpm_combined20d")
    _copy(CV / "cluster_assignments_deepdpm_combined20d.csv", dpm)
    n_clusters = None
    csv_path = CV / "cluster_assignments_deepdpm_combined20d.csv"
    if csv_path.is_file():
        import csv

        with open(csv_path, newline="", encoding="utf-8") as f:
            n_clusters = len({int(r["cluster"]) for r in csv.DictReader(f)})
    manifest["deepdpm_combined20d"] = {
        "assignments": "cluster_assignments_deepdpm_combined20d.csv",
        "n_clusters": n_clusters,
    }

    # Maxwellian scores
    data_dir = repo_root() / "data" / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    for src in (CV / "maxwellian_scores_combined.csv", DCEC_VALIDATION / "maxwellian_scores_combined.csv"):
        if _copy(src, data_dir):
            break

    # 3D score views
    score3d_map = [
        ("score3d_gmm", CV / "cluster_score_anomaly_3d_combined.png", CV / "cluster_score_anomaly_3d_combined_interactive.html"),
        ("score3d_agglomerative_k8", CV / "cluster_score_anomaly_3d_agglomerative_k8.png", CV / "cluster_score_anomaly_3d_agglomerative_k8_interactive.html"),
        ("score3d_hdbscan", CV_H / "cluster_score_anomaly_3d_hdbscan.png", CV_H / "cluster_score_anomaly_3d_hdbscan_interactive.html"),
        ("score3d_deepdpm", CV / "cluster_score_anomaly_3d_deepdpm_combined20d.png", CV / "cluster_score_anomaly_3d_deepdpm_combined20d_interactive.html"),
    ]
    for folder, png, html in score3d_map:
        out = analysis_out(folder)
        _copy(png, out)
        _copy(html, out)

    # Robustness sweeps (newest per method)
    for key, meta in METHODS.items():
        _sync_robustness(meta["robustness_dir"], meta["assignments_rel"], manifest)

    # Per-method cluster-colored UMAP|PaCMAP (if generated under CV)
    for key, meta in METHODS.items():
        sub = CV / meta["embed_dir"]
        if sub.is_dir():
            copied = _copy_glob(sub, analysis_out(meta["embed_dir"]), EMBED_CLUSTER_GLOB)
            if copied:
                manifest[meta["embed_dir"]] = {"files": copied}

    # Shared Maxwellian embeddings
    max_out = analysis_out("embed_maxwellian_combined")
    src_dirs = [DCEC_VALIDATION, CV]
    copied_mv = []
    for d in src_dirs:
        if not d.is_dir():
            continue
        for name in SHARED_ARTIFACTS["embed_maxwellian"]:
            if _copy(d / name, max_out, name) and name not in copied_mv:
                copied_mv.append(name)
    if copied_mv:
        manifest["embed_maxwellian_combined"] = {"files": copied_mv}

    # Shared TDA / persistence / filtration
    tda_out = analysis_out("tda_exploratory")
    for d in (DCEC_VALIDATION, CV):
        if not d.is_dir():
            continue
        for name in SHARED_ARTIFACTS["tda_exploratory"]:
            _copy(d / name, tda_out, name)
    # Optional evolution GIF/frames directory
    evo = DCEC_VALIDATION / "embed_persistence_combined_evolution"
    if evo.is_dir():
        dest_evo = tda_out / "embed_persistence_combined_evolution"
        if dest_evo.exists():
            shutil.rmtree(dest_evo)
        shutil.copytree(evo, dest_evo)
    if any(tda_out.iterdir()):
        manifest["tda_exploratory"] = {
            "persistence_png": "embed_persistence_combined.png",
            "filtration_slider": "filtration_blob_demo_combined_pacmap_slider.html",
            "mapper_html": "embed_mapper_combined.html",
        }

    out_json = repo_root() / "analysis" / "manifest.json"
    out_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")
    print("Synced DCEC outputs into analysis/outputs/")


if __name__ == "__main__":
    main()
