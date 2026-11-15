# 监测 C: 空间与 Docker vhdx 增长，用于评估 A3 全量批处理（193 例 / 约 6 天）的可行性。
#
# 为什么需要它
# ------------
# docker_data.vhdx 是**稀疏分配且不自动收缩**的：容器每次写入都可能让它变大，
# 而删除容器/镜像不会让它变小。193 例的长时间批处理必须知道：
#   1) 每例平均让 vhdx 增长多少 -> 由此推算全量需求
#   2) C: 可用空间的下降速率 -> 何时触及安全线
#
# 用法：作为后台作业运行；输出追加到 build/space_monitor.csv
$ErrorActionPreference = 'SilentlyContinue'
$wd = "C:\Users\HUAWEI\Documents\deepseek-harness\default-workspace\tn-whiteterm"
$csv = "$wd\build\space_monitor.csv"
$vhd = "$env:LOCALAPPDATA\Docker\wsl\disk\docker_data.vhdx"

if (-not (Test-Path $csv)) {
    "timestamp,c_free_gb,vhdx_gb,work_mb,fastsurfer_n,done_subjects" | Set-Content $csv -Encoding utf8
}

$deadline = (Get-Date).AddHours(12)
while ((Get-Date) -lt $deadline) {
    $free = [math]::Round([System.IO.DriveInfo]::new('C:\').AvailableFreeSpace / 1GB, 2)
    $v = 0
    if (Test-Path $vhd) { $v = [math]::Round((Get-Item $vhd).Length / 1GB, 3) }
    $workMb = [math]::Round(((Get-ChildItem "$wd\build\work" -Recurse -File | Measure-Object Length -Sum).Sum) / 1MB, 1)
    $n = @(Get-ChildItem "$wd\build\work\fastsurfer" -Directory).Count
    $log = Get-Content "$wd\build\a3_batch_A2.log" -Encoding utf8
    $done = @($log | Select-String '单例总耗时').Count
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    "$ts,$free,$v,$workMb,$n,$done" | Add-Content $csv -Encoding utf8
    # 若 C: 低于安全线则显著告警（不自行终止 A3，交由上层决策）
    if ($free -lt 25) { Write-Output "[ALERT] C: 可用仅 $free GB" }
    Start-Sleep -Seconds 300
}
