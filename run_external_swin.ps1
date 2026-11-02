# External tests (CMRxMotion, M&Ms) for the Swin-UNETR seeds 3 and 4.
# Waits until the Swin-UNETR training queue has finished; runs next to the extra queue (two GPU jobs at most).
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
while (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_revision_(unet|swin|chain)\.ps1' }) { Start-Sleep -Seconds 120 }
$runs = foreach ($v in 'aug', 'dacmnorm', 'dacmzero', 'dacmshuf') { foreach ($s in '_s3', '_s4') { "work\runs\swinunetr_$v$s" } }
"===== cmrx $(Get-Date -Format HH:mm:ss) ====="
& $py -u scripts\eval_cmrx.py --data_root ..\data\CMRxMotion\CMRxMotion_Training --cache work\cmrx_cache --estimator work\estimator --out work\cmrx_eval --runs $runs
"===== mnms $(Get-Date -Format HH:mm:ss) ====="
& $py -u scripts\eval_mnms.py --data_root ..\data\MnMs\OpenDataset --cache work\mnms_cache --estimator work\estimator --out work\mnms_eval --runs $runs
"===== done $(Get-Date -Format HH:mm:ss) ====="
