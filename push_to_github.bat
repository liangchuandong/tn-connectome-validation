@echo off
REM 在 deposit 目录下运行；需要先安装 Git for Windows
setlocal

git init
git add -A
git commit -m "v1.0.0: analysis code and derived data"
git branch -M main
git remote remove origin 2>nul
git remote add origin https://github.com/liangchuandong/tn-connectome-validation.git
git push -u origin main

echo.
echo 推送完成。下一步：
echo   1. GitHub 页面 -^> Releases -^> Draft a new release -^> Tag: v1.0.0
echo   2. Zenodo -^> Settings -^> GitHub -^> 打开该仓库开关
echo   3. 发布 release 后，Zenodo 自动分配 DOI
pause
