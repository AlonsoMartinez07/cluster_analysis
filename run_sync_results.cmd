@echo off
cd /d "%~dp0"
set PY=D:\GENE_simulation_AI\DCEC\torch_DCEC_RGB\venv\Scripts\python.exe
echo Syncing existing DCEC reports into analysis/outputs ...
"%PY%" -u src\sync_existing_results.py
echo Done. See analysis\manifest.json
pause
