@echo off
REM Full pipeline: clustering + per-method validation + Maxwellian + TDA + sync
REM DeepDPM training is NOT re-run (use existing assignments CSV).
cd /d "%~dp0"
set PY=D:\GENE_simulation_AI\DCEC\torch_DCEC_RGB\venv\Scripts\python.exe
echo Starting full pipeline (hours). Logs under logs\YYYYMMDD-*
"%PY%" -u src\run_full_pipeline.py %*
pause
