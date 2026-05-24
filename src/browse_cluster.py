"""
Interactive cluster browser: pick algorithm + cluster, view all member graphs.

Generates an HTML gallery under analysis/outputs/cluster_browser/<method>/cluster_<id>/
  - rgb/     copies of AE difference PNGs (from assignments file_path)
  - f_sim/   contour plots from matching .nc files (v_parallel vs mu)
  - gallery.html

Usage (interactive):
  python src/browse_cluster.py

Non-interactive:
  python src/browse_cluster.py --method gmm --cluster 9
"""
from __future__ import annotations

import argparse
import csv
import html
import os
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from method_registry import METHODS
from paths_config import analysis_out, load_config, repo_root

# Reuse .nc loading from DCEC
_DCEC_RGB = Path(load_config()["torch_dcec_rgb"])
sys.path.insert(0, str(_DCEC_RGB))
from visualize_clusters_with_data import build_sim_maps, load_simulation_data  # noqa: E402


ASSIGNMENT_NAMES = {
    "gmm": "cluster_assignments.csv",
    "agglomerative": "cluster_assignments_agglomerative_ward_k8.csv",
    "hdbscan": "cluster_assignments.csv",
    "deepdpm": "cluster_assignments_deepdpm_combined20d.csv",
}


def _assignments_csv(method: str) -> Path:
    meta = METHODS[method]
    local = analysis_out(meta["output_dir"]) / ASSIGNMENT_NAMES[method]
    if local.is_file():
        return local
    return _DCEC_RGB / meta["assignments_rel"]


def load_assignments(method: str) -> list[dict]:
    path = _assignments_csv(method)
    if not path.is_file():
        raise FileNotFoundError(f"Assignments not found: {path}")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "sim_id": Path(r["file_path"]).stem.replace("_diff", ""),
                    "file_path": r["file_path"],
                    "cluster": int(r["cluster"]),
                }
            )
    return rows


def load_maxwellian_scores() -> dict[str, float]:
    scores: dict[str, float] = {}
    for p in (
        repo_root() / "data/processed/maxwellian_scores_combined.csv",
        _DCEC_RGB / "reports/cluster_visualizations/maxwellian_scores_combined.csv",
    ):
        if not p.is_file():
            continue
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    scores[r["sim_id"]] = float(r["maxwellianity_score"])
                except (ValueError, KeyError):
                    pass
        break
    return scores


def cluster_counts(rows: list[dict]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for r in rows:
        c = r["cluster"]
        counts[c] = counts.get(c, 0) + 1
    return counts


def prompt_method() -> str:
    keys = list(METHODS.keys())
    print("\n=== Cluster browser — choose algorithm ===")
    for i, k in enumerate(keys, 1):
        print(f"  {i}. {k}")
    while True:
        raw = input(f"Enter number [1-{len(keys)}] or name: ").strip().lower()
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(keys):
                return keys[idx]
        elif raw in METHODS:
            return raw
        print("  Invalid choice, try again.")


def prompt_cluster(counts: dict[int, int]) -> int:
    print("\n=== Available clusters (id: count) ===")
    for c in sorted(counts.keys()):
        label = "noise" if c == -1 else f"cluster {c}"
        print(f"  {c:>4}  ({label}): {counts[c]} samples")
    while True:
        raw = input("Enter cluster id: ").strip()
        try:
            cid = int(raw)
        except ValueError:
            print("  Enter an integer (e.g. 9, or -1 for HDBSCAN noise).")
            continue
        if cid in counts:
            return cid
        print(f"  Cluster {cid} not in this assignment file.")


def save_fsim_plot(nc_path: str, out_png: Path, sim_id: str, sim_type: str) -> bool:
    try:
        f_sim, vpar, mu = load_simulation_data(nc_path)
        fig, ax = plt.subplots(figsize=(3.2, 2.8))
        ax.contourf(vpar, mu, f_sim, levels=40, cmap="viridis")
        ax.set_xlabel(r"$v_\parallel$", fontsize=8)
        ax.set_ylabel(r"$\mu$", fontsize=8)
        ax.set_title(f"{sim_id} [{sim_type}]", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"  Warning: f_sim plot failed for {sim_id}: {e}")
        return False


def build_gallery(
    method: str,
    cluster_id: int,
    members: list[dict],
    sim_to_type: dict,
    sim_to_path: dict,
    scores: dict[str, float],
    out_root: Path,
) -> Path:
    out_dir = out_root / method / f"cluster_{cluster_id:02d}" if cluster_id >= 0 else out_root / method / "cluster_noise"
    rgb_dir = out_dir / "rgb"
    fsim_dir = out_dir / "f_sim"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    fsim_dir.mkdir(parents=True, exist_ok=True)

    # Sort: highest Maxwellian score first (easier to spot outliers)
    def sort_key(r: dict):
        s = scores.get(r["sim_id"])
        return (-s if s is not None else 999.0, r["sim_id"])

    members = sorted(members, key=sort_key)

    stats = {"normal": 0, "anomaly": 0, "unknown": 0}
    cards = []

    print(f"\nBuilding gallery for {method} cluster {cluster_id} ({len(members)} samples)...")
    for i, r in enumerate(members):
        sim_id = r["sim_id"]
        st = sim_to_type.get(sim_id, "unknown")
        stats[st] = stats.get(st, 0) + 1
        score = scores.get(sim_id)
        score_txt = f"{score:.4f}" if score is not None else "n/a"

        # RGB diff (copy into repo for portable HTML)
        src_rgb = Path(r["file_path"])
        rgb_name = f"{sim_id}_diff.png"
        dst_rgb = rgb_dir / rgb_name
        if src_rgb.is_file():
            if not dst_rgb.is_file() or dst_rgb.stat().st_size != src_rgb.stat().st_size:
                shutil.copy2(src_rgb, dst_rgb)
        else:
            print(f"  Missing RGB: {src_rgb}")

        # f_sim from .nc
        nc_path = sim_to_path.get(sim_id)
        fsim_name = f"{sim_id}_fsim.png"
        dst_fsim = fsim_dir / fsim_name
        fsim_ok = False
        if nc_path and os.path.isfile(nc_path):
            fsim_ok = save_fsim_plot(nc_path, dst_fsim, sim_id, st)
        elif not nc_path:
            print(f"  No .nc for {sim_id}")

        cards.append(
            {
                "sim_id": sim_id,
                "type": st,
                "score": score_txt,
                "rgb": f"rgb/{rgb_name}" if dst_rgb.is_file() else None,
                "fsim": f"f_sim/{fsim_name}" if fsim_ok and dst_fsim.is_file() else None,
            }
        )
        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(members)}")

    html_path = out_dir / "gallery.html"
    _write_html(html_path, method, cluster_id, len(members), stats, cards)
    print(f"\nSaved gallery: {html_path}")
    return html_path


def _write_html(
    path: Path,
    method: str,
    cluster_id: int,
    n_total: int,
    stats: dict,
    cards: list[dict],
) -> None:
    title = f"{method} — cluster {cluster_id}"
    rows_html = []
    for c in cards:
        rgb_img = (
            f'<img src="{html.escape(c["rgb"])}" alt="rgb" loading="lazy"/>'
            if c["rgb"]
            else '<div class="missing">RGB missing</div>'
        )
        fsim_img = (
            f'<img src="{html.escape(c["fsim"])}" alt="fsim" loading="lazy"/>'
            if c["fsim"]
            else '<div class="missing">f_sim missing</div>'
        )
        rows_html.append(
            f"""
        <div class="card">
          <div class="meta"><b>{html.escape(c["sim_id"])}</b>
            <span class="tag {c["type"]}">{c["type"]}</span>
            <span class="score">Maxwellian: {c["score"]}</span></div>
          <div class="pair">
            <div><div class="lbl">RGB diff (AE input)</div>{rgb_img}</div>
            <div><div class="lbl">f_sim (v∥ vs μ)</div>{fsim_img}</div>
          </div>
        </div>"""
        )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 1rem 2rem; background: #111; color: #eee; }}
    h1 {{ font-size: 1.4rem; }}
    .stats {{ color: #aaa; margin-bottom: 1.5rem; }}
    .grid {{ display: flex; flex-direction: column; gap: 1.2rem; }}
    .card {{ background: #1e1e1e; border-radius: 8px; padding: 0.8rem; }}
    .meta {{ margin-bottom: 0.5rem; font-size: 0.9rem; }}
    .tag {{ padding: 0.1rem 0.4rem; border-radius: 4px; margin-left: 0.5rem; font-size: 0.75rem; }}
    .tag.normal {{ background: #2d5a27; }}
    .tag.anomaly {{ background: #6b2d2d; }}
    .tag.unknown {{ background: #444; }}
    .score {{ margin-left: 0.8rem; color: #8cf; }}
    .pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; }}
    .lbl {{ font-size: 0.75rem; color: #888; margin-bottom: 0.25rem; }}
    img {{ max-width: 100%; height: auto; border-radius: 4px; background: #000; }}
    .missing {{ color: #666; font-size: 0.85rem; padding: 2rem; text-align: center;
                border: 1px dashed #444; border-radius: 4px; }}
    @media (max-width: 900px) {{ .pair {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class="stats">{n_total} simulations —
    normal: {stats.get("normal", 0)},
    anomaly: {stats.get("anomaly", 0)},
    unknown: {stats.get("unknown", 0)}.
    Sorted by Maxwellian score (high → low).</p>
  <div class="grid">
    {"".join(rows_html)}
  </div>
</body>
</html>"""
    path.write_text(doc, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Browse all graphs in one cluster")
    parser.add_argument("--method", choices=list(METHODS.keys()), help="Skip interactive method prompt")
    parser.add_argument("--cluster", type=int, help="Cluster id (-1 = HDBSCAN noise)")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Default: analysis/outputs/cluster_browser",
    )
    args = parser.parse_args()

    cfg = load_config()
    out_root = Path(args.out_dir) if args.out_dir else analysis_out("cluster_browser")

    method = args.method or prompt_method()
    rows = load_assignments(method)
    counts = cluster_counts(rows)

    cluster_id = args.cluster if args.cluster is not None else prompt_cluster(counts)
    members = [r for r in rows if r["cluster"] == cluster_id]
    if not members:
        print("No members in that cluster.")
        return 1

    n = len(members)
    if n > 80 and args.cluster is None:
        ans = input(f"Cluster has {n} samples — generate all plots? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted. Use --cluster with smaller cluster or confirm with 'y'.")
            return 0

    print("Mapping .nc files (normal + anomaly dirs)...")
    sim_to_type, sim_to_path = build_sim_maps(cfg["normal_dir"], cfg["anomaly_dir"])
    scores = load_maxwellian_scores()

    html_path = build_gallery(
        method, cluster_id, members, sim_to_type, sim_to_path, scores, out_root
    )
    print(f"Open in browser: file:///{html_path.resolve().as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
