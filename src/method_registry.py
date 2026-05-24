"""Cluster methods: DCEC assignment paths and output folder names."""
from __future__ import annotations

METHODS: dict[str, dict] = {
    "gmm": {
        "assignments_rel": "reports/cluster_visualizations/cluster_assignments.csv",
        "output_dir": "gmm_bic",
        "robustness_dir": "robustness_gmm",
        "score3d_dir": "score3d_gmm",
        "embed_dir": "embed_clusters_gmm",
        "run_cluster": "gmm",
    },
    "agglomerative": {
        "assignments_rel": "reports/cluster_visualizations/cluster_assignments_agglomerative_ward_k8.csv",
        "output_dir": "agglomerative_ward",
        "robustness_dir": "robustness_agglomerative_k8",
        "score3d_dir": "score3d_agglomerative_k8",
        "embed_dir": "embed_clusters_agglomerative_k8",
        "run_cluster": "agglomerative",
    },
    "hdbscan": {
        "assignments_rel": "reports/cluster_visualizations_hdbscan/cluster_assignments.csv",
        "output_dir": "hdbscan",
        "robustness_dir": "robustness_hdbscan",
        "score3d_dir": "score3d_hdbscan",
        "embed_dir": "embed_clusters_hdbscan",
        "run_cluster": "hdbscan",
    },
    "deepdpm": {
        "assignments_rel": "reports/cluster_visualizations/cluster_assignments_deepdpm_combined20d.csv",
        "output_dir": "deepdpm_combined20d",
        "robustness_dir": "robustness_deepdpm",
        "score3d_dir": "score3d_deepdpm",
        "embed_dir": "embed_clusters_deepdpm",
        "run_cluster": None,
    },
}

SHARED_ARTIFACTS = {
    "embed_maxwellian": [
        "embed_umap_combined_maxwellian.png",
        "embed_pacmap_combined_maxwellian.png",
        "embed_umap_pacmap_combined_maxwellian.png",
        "maxwellian_scores_combined.csv",
    ],
    "tda_exploratory": [
        "embed_persistence_combined.png",
        "embed_mapper_combined.html",
        "tda_summary_combined.json",
        "tda_summary_combined.txt",
        "filtration_blob_demo_combined_pacmap_slider.html",
        "embed_persistence_combined_evolution_stages.csv",
    ],
}

EMBED_CLUSTER_GLOB = [
    "embed_umap_combined.png",
    "embed_pacmap_combined.png",
    "embed_umap_pacmap_combined.png",
]
