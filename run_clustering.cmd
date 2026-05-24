@echo off
REM Re-run clustering with logs (slow). Edit METHODS below.
cd /d "%~dp0"
set PY=D:\GENE_simulation_AI\DCEC\torch_DCEC_RGB\venv\Scripts\python.exe
set METHODS=gmm,agglomerative,hdbscan,score3d
set ASSIGN=reports/cluster_visualizations/cluster_assignments.csv
"%PY%" -u src\run_clustering_logged.py --methods %METHODS% --assignments %ASSIGN%
echo.
echo For robustness sweep per assignment:
echo   "%PY%" -u src\run_clustering_logged.py --methods robustness --assignments reports/cluster_visualizations/cluster_assignments_agglomerative_ward_k8.csv
pause
