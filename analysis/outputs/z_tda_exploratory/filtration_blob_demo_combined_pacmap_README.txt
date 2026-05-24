Disk-overlap animation (intuition for PM/team)
----------------------------------------------
Each point grows a disk of radius r. When two disks touch, we draw an edge
between them (equivalent to dist(i,j) <= 2r). As r increases, more edges
appear and separate components merge — same *idea* as growing neighborhoods
in a Vietoris–Rips filtration.

Important: this animation is built on a 2D projection (PaCMAP/UMAP/PCA).
Your Ripser persistence diagram uses distances in the full feature space
(often on a subsample). The two are related conceptually, not identical.
