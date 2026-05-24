@echo off
REM Re-run validation + sync only (keeps existing cluster assignments / DeepDPM).
cd /d "%~dp0"
set PY=D:\GENE_simulation_AI\DCEC\torch_DCEC_RGB\venv\Scripts\python.exe
"%PY%" -u src\run_full_pipeline.py --skip-clustering
pause
