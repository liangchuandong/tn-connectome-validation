@echo off
rem ============================================================================
rem  A1 batch launcher  (ASCII only on purpose -- cmd.exe reads .cmd as GBK,
rem  and non-ASCII bytes here have broken this project before)
rem
rem  WHAT IT DOES
rem  ------------
rem  Runs scripts\run_a1_pipeline.py over every subject listed in
rem  build\a1_subjects.txt, producing for each subject:
rem
rem      a2\sc\<sub>_streamlines.csv   200x200   streamline-count SC matrix
rem      a2\sc\<sub>_fa.csv            200x200   FA-weighted SC matrix
rem      a1_streamlines\<sub>.csv      200       node strength (off-diag row sum)
rem      a1_fa\<sub>.csv               200       FA-weighted node strength
rem
rem  A1 and A2 share one tractography run, so this script also satisfies the
rem  A2 contract's structural-connectivity requirement.
rem
rem  PREREQUISITES (checked below)
rem  -----------------------------
rem  * A3 must have produced, per subject:
rem        <a3-work>\fastsurfer\<sub>\mri\aseg.auto.mgz
rem        <a3-work>\fastsurfer\<sub>\mri\orig.mgz
rem        <a3-work>\registration\Schaefer200_in_<sub>.nii.gz
rem    Regenerate the subject list with:
rem        python scripts\make_a1_a2_lists.py --raw F:\ds005713\raw
rem               --a3-work <a3-work> --out-dir build
rem  * Docker image mrtrix3/mrtrix3 must be present (it bundles FSL 6.0.7.7).
rem
rem  RESUMABILITY
rem  ------------
rem  Each stage (prep / preproc / fod / 5tt / tract / connectome) is skipped
rem  when its output already exists and is non-trivial in size. Worst case on
rem  an abrupt shutdown is losing the subject currently being processed.
rem
rem  SINGLE WORKER ONLY
rem  ------------------
rem  Measured on this host: 8 logical cores, ~16 GB RAM, Docker limited to
rem  8 GB. A single FastSurfer/tractography run already leaves under ~2 GB of
rem  free physical memory. Running two workers has previously pushed commit
rem  past its limit and grown C:\pagefile.sys by ~7.5 GB. Do NOT parallelise.
rem
rem  USAGE
rem  -----
rem      scripts\run_a1_batch.cmd
rem ============================================================================
setlocal EnableExtensions EnableDelayedExpansion

chcp 65001 >nul 2>&1

set "SCRIPT_DIR=%~dp0"
set "PROJ_DIR=%SCRIPT_DIR%.."
set "RAW=F:\ds005713\raw"
set "WORK=%PROJ_DIR%\build\a1_work"
set "OUT=%PROJ_DIR%\build\a1_out"
set "A3WORK=%PROJ_DIR%\build\work"
set "DEST=F:\ds005713\derivatives"
set "LIST=%PROJ_DIR%\build\a1_subjects.txt"
set "LOGDIR=%PROJ_DIR%\build\download_logs"
set "PY=F:\venvs\tnwt\Scripts\python.exe"

if not exist "%PY%" (
  echo [ERR] analysis venv not found: %PY%
  exit /b 2
)
if not exist "%LIST%" (
  echo [ERR] subject list not found: %LIST%
  echo       run: python scripts\make_a1_a2_lists.py --raw %RAW% --a3-work %A3WORK% --out-dir %PROJ_DIR%\build
  exit /b 2
)
if not exist "%A3WORK%" (
  echo [ERR] A3 work dir not found: %A3WORK%   (A1 reuses A3 segmentation + atlas)
  exit /b 2
)

if not exist "%WORK%"   mkdir "%WORK%"   2>nul
if not exist "%OUT%"    mkdir "%OUT%"    2>nul
if not exist "%LOGDIR%" mkdir "%LOGDIR%" 2>nul

set "TMPDIR=%PROJ_DIR%\build\tmp"
if not exist "%TMPDIR%" mkdir "%TMPDIR%" 2>nul
set "TEMP=%TMPDIR%"
set "TMP=%TMPDIR%"
set "PYTHONIOENCODING=utf-8"
set "OMP_NUM_THREADS=4"

rem ---- disk safety gate: refuse to start below 25 GB free on C: ----
rem NOTE: do NOT parse `dir /-c | findstr "bytes free"`: findstr treats the
rem space as an OR, so both the "File(s)" and "Dir(s)" lines match, and any
rem fixed token index picks up "Dir(s)" instead of the byte count (verified).
rem PowerShell reads the value directly and unambiguously.
set "FREEGB="
for /f "usebackq delims=" %%F in (`powershell -NoProfile -Command "[int]((Get-PSDrive C).Free/1GB)"`) do set "FREEGB=%%F"
if not defined FREEGB (
  echo [ABORT] Could not determine free space on C:.
  exit /b 4
)
if %FREEGB% LSS 25 (
  echo.
  echo [ABORT] C: free space is %FREEGB% GB, below the 25 GB safety threshold.
  echo         Refusing to start a multi-day job. Free space first.
  exit /b 3
)

echo.
echo ===================== A1 batch (resumable) =====================
echo   python   : %PY%
echo   raw      : %RAW%
echo   work     : %WORK%
echo   out      : %OUT%
echo   a3-work  : %A3WORK%
echo   dest     : %DEST%
echo   subjects : %LIST%
echo   threads  : 4    (single worker: host RAM is tight, do NOT parallelise)
echo   free C:  : %FREEGB% GB
echo ===============================================================
echo.
echo   Expected cost: ~45 min per subject if eddy dominates.
echo   Safe to stop at any time. Re-run this same script to resume.
echo.

"%PY%" -u "%SCRIPT_DIR%run_a1_pipeline.py" ^
  --raw "%RAW%" ^
  --work "%WORK%" ^
  --out "%OUT%" ^
  --a3-work "%A3WORK%" ^
  --dest "%DEST%" ^
  --subjects-file "%LIST%" ^
  --threads 4 --streamlines 10000000 --timeit
set "CODE=%ERRORLEVEL%"

echo.
if "%CODE%"=="0" (
  echo [OK] A1 batch finished.  logs: %LOGDIR%
) else (
  echo [WARN] A1 batch exited with code %CODE%.  logs: %LOGDIR%
  echo        Re-run this script to resume; completed stages are skipped.
)
exit /b %CODE%
