@echo off
rem ============================================================================
rem  ds005713 staged download launcher  (ASCII only on purpose)
rem
rem  Chinese docs: see the module docstring in scripts/fetch_slice.py,
rem  or docs/download_and_initial_analysis_plan.md
rem
rem  WHY ASCII-ONLY: cmd.exe reads .cmd files using the OEM codepage (GBK on a
rem  Chinese Windows), not UTF-8. Non-ASCII bytes in this file would be
rem  mis-decoded and turn comments into bogus commands. All Chinese text lives
rem  in the Python script instead, which handles UTF-8 correctly.
rem
rem  WHY .cmd AND NOT .ps1: PowerShell execution policy blocks .ps1 by default
rem  on this machine, and a UTF-8-without-BOM .ps1 is also mis-read as GBK.
rem
rem  USAGE (run from the project root or anywhere):
rem      scripts\fetch_all.cmd                 -> meta, anat, dwi, func
rem      scripts\fetch_all.cmd meta anat       -> only these stages
rem      scripts\fetch_all.cmd dry             -> list only, no download
rem      scripts\fetch_all.cmd dry meta anat   -> dry-run for those stages
rem
rem  Resumable: re-running skips files whose size already matches the manifest.
rem ============================================================================
setlocal EnableExtensions EnableDelayedExpansion

rem ---- UTF-8 console so Python's UTF-8 output (Chinese stage logs) is legible ----
chcp 65001 >nul 2>&1

set "SCRIPT_DIR=%~dp0"
set "PROJ_DIR=%SCRIPT_DIR%.."
set "DEST=F:\ds005713\raw"
set "WORKTMP=F:\ds005713\tmp"

rem ---- pick Python: prefer the analysis venv, fall back to DSH runtime ----
set "PY=F:\venvs\tnwt\Scripts\python.exe"
if not exist "%PY%" set "PY=C:\Users\HUAWEI\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
if not exist "%PY%" (
  echo [ERR] Python interpreter not found
  exit /b 2
)

set "FETCH=%SCRIPT_DIR%fetch_slice.py"
if not exist "%FETCH%" ( echo [ERR] missing %FETCH% & exit /b 2 )
set "MANIFEST=%PROJ_DIR%\data\manifests\ds005713_files.csv"
if not exist "%MANIFEST%" (
  echo [ERR] missing manifest: %MANIFEST%
  echo       run scripts\fetch_manifest.py first.
  exit /b 2
)

rem ---- child-process temp dir must be writable (DSH sandbox temp is not) ----
if not exist "%WORKTMP%" mkdir "%WORKTMP%" 2>nul
set "TEMP=%WORKTMP%"
set "TMP=%WORKTMP%"
set "PYTHONIOENCODING=utf-8"

rem ---- parse args: "dry" is a flag, everything else is a stage name ----
set "STAGE_ARGS="
set "DRY="
for %%A in (%*) do (
  if /i "%%A"=="dry" ( set "DRY=--dry-run" ) else ( set "STAGE_ARGS=!STAGE_ARGS! --slice %%A" )
)
if "%STAGE_ARGS%"=="" set "STAGE_ARGS=--slice meta --slice anat --slice dwi --slice func"

rem ---- log file ----
set "LOGDIR=%PROJ_DIR%\build\download_logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%" 2>nul

echo.
echo ================ ds005713 staged download ================
echo   Python   : %PY%
echo   manifest : %MANIFEST%
echo   dest     : %DEST%
echo   stages   :%STAGE_ARGS%
if not "%DRY%"=="" echo   mode     : DRY-RUN
echo =========================================================
echo.

rem ---- run; pipe through PowerShell Tee-Object for console+file logging ----
"%PY%" -u "%FETCH%" --manifest "%MANIFEST%" --dest "%DEST%" --workers 8 %STAGE_ARGS% %DRY% 2>&1 | powershell -NoProfile -Command "$input | Tee-Object -FilePath (Join-Path '%LOGDIR%' ('download_' + (Get-Date -Format yyyyMMdd_HHmmss) + '.log'))"
set "CODE=%ERRORLEVEL%"

echo.
if "%CODE%"=="0" (
  echo [OK] download finished.  logs: %LOGDIR%
  echo.
  echo      next: deep gzip-level verification
  echo        "%PY%" "%SCRIPT_DIR%verify_download.py" --dest "%DEST%" --deep
) else (
  echo [WARN] download incomplete ^(exit=%CODE%^)  logs: %LOGDIR%
  echo        re-run the same command to resume; finished files are skipped.
)
exit /b %CODE%
