@echo off
rem ============================================================================
rem  A2 batch launcher  (ASCII only on purpose)
rem
rem  WHAT IT DOES
rem  ------------
rem  Runs scripts\run_a2_pipeline.py over every subject in build\a2_subjects.txt,
rem  producing for each subject:
rem
rem      a2\fc\<sub>.csv          200x200  Pearson r functional connectivity
rem      a2\timeseries\<sub>.npy  T x 200  denoised parcel time series
rem      a2\qc\<sub>.json         motion + denoising diagnostics (FD, confounds)
rem
rem  PREREQUISITES (checked below)
rem  -----------------------------
rem  * A3 must have produced, per subject:
rem        <a3-work>\fastsurfer\<sub>\mri\orig.mgz
rem        <a3-work>\fastsurfer\<sub>\mri\aseg.auto.mgz
rem        <a3-work>\registration\Schaefer200_in_<sub>.nii.gz
rem  * Docker image mrtrix3/mrtrix3 (bundles FSL 6.0.7.7: mcflirt, flirt).
rem
rem  ORDERING NOTE
rem  -------------
rem  A2's own contract row only needs FC + time series, but the downstream
rem  structure-function coupling analysis needs the SC matrices that A1
rem  produces. Suggest running A1 first (it is the long pole), then A2 -- but
rem  A2 is fine to run standalone for FC-only work, and it is much faster
rem  (~10 min/subject vs ~45).
rem
rem  RESUMABILITY
rem  ------------
rem  The FSL stage (mcflirt + flirt + resample) is skipped when its outputs
rem  exist; the cleaning stage always recomputes from them (it is cheap).
rem
rem  USAGE
rem  -----
rem      scripts\run_a2_batch.cmd
rem ============================================================================
setlocal EnableExtensions EnableDelayedExpansion

chcp 65001 >nul 2>&1

set "SCRIPT_DIR=%~dp0"
set "PROJ_DIR=%SCRIPT_DIR%.."
set "RAW=F:\ds005713\raw"
set "WORK=%PROJ_DIR%\build\a2_work"
set "OUT=%PROJ_DIR%\build\a2_out"
set "A3WORK=%PROJ_DIR%\build\work"
set "DEST=F:\ds005713\derivatives"
set "LIST=%PROJ_DIR%\build\a2_subjects.txt"
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
  echo [ERR] A3 work dir not found: %A3WORK%   (A2 reuses A3 segmentation + atlas)
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
set "MPLCONFIGDIR=%TMPDIR%\mplconfig"
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%" 2>nul

rem ---- disk safety gate (same threshold and same parsing caveat as A1) ----
set "FREEGB="
for /f "usebackq delims=" %%F in (`powershell -NoProfile -Command "[int]((Get-PSDrive C).Free/1GB)"`) do set "FREEGB=%%F"
if not defined FREEGB (
  echo [ABORT] Could not determine free space on C:.
  exit /b 4
)
if %FREEGB% LSS 25 (
  echo.
  echo [ABORT] C: free space is %FREEGB% GB, below the 25 GB safety threshold.
  echo         Refusing to start. Free space first.
  exit /b 3
)

echo.
echo ===================== A2 batch (resumable) =====================
echo   python   : %PY%
echo   raw      : %RAW%
echo   work     : %WORK%
echo   out      : %OUT%
echo   a3-work  : %A3WORK%
echo   dest     : %DEST%
echo   subjects : %LIST%
echo   free C:  : %FREEGB% GB
echo ===============================================================
echo.
echo   Expected cost: ~10 min per subject (mcflirt dominates).
echo   Safe to stop at any time. Re-run this same script to resume.
echo.

"%PY%" -u "%SCRIPT_DIR%run_a2_pipeline.py" ^
  --raw "%RAW%" ^
  --work "%WORK%" ^
  --out "%OUT%" ^
  --a3-work "%A3WORK%" ^
  --dest "%DEST%" ^
  --subjects-file "%LIST%" ^
  --timeit
set "CODE=%ERRORLEVEL%"

echo.
if "%CODE%"=="0" (
  echo [OK] A2 batch finished.  logs: %LOGDIR%
) else (
  echo [WARN] A2 batch exited with code %CODE%.  logs: %LOGDIR%
  echo        Re-run this script to resume; FSL stage outputs are reused.
)
exit /b %CODE%
