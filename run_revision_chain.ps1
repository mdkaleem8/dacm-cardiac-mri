# Run the revision queues one after another (sharing the GPU slowed each job about 2.7x).
# Waits for the U-Net queue process to exit, then runs the Swin-UNETR queue. The extra queue
# (run_revision_extra.ps1) waits for "queue finished" in revision_swin.log and runs last.
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$work = Join-Path $root "work"
while (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_revision_unet\.ps1' }) {
  Start-Sleep -Seconds 60
}
Write-Host "U-Net queue ended $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'); starting Swin-UNETR queue"
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "run_revision_swin.ps1") *> (Join-Path $work "revision_swin.log")
Write-Host "Swin-UNETR queue ended $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
