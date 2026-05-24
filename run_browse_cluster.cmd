@echo off
cd /d "%~dp0"
set PY=D:\GENE_simulation_AI\DCEC\torch_DCEC_RGB\venv\Scripts\python.exe
echo Interactive cluster browser (pick algorithm + cluster, builds HTML gallery)
"%PY%" -u src\browse_cluster.py %*
pause
