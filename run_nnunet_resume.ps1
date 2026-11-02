# Resume the degradation-trained nnU-Net from its last checkpoint (the first run stopped at
# epoch 235 when a data-loading worker died), then evaluate it. Runs next to the Tent step,
# so at most two jobs share the GPU. Retries up to three times; each retry resumes again.
$root  = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$py    = Join-Path $root ".venv\Scripts\python.exe"
$work  = Join-Path $root "work"
$cache = Join-Path $work "acdc_cache"; $tset = Join-Path $work "testset"; $runs = Join-Path $work "runs"
$nn    = Join-Path $work "nnunet"
$env:nnUNet_raw = "$nn\raw"; $env:nnUNet_preprocessed = "$nn\preprocessed"; $env:nnUNet_results = "$nn\results"
$env:PYTHONPATH = $root; $env:DACM_CALIB = Join-Path $work "calib.json"; $env:nnUNet_n_proc_DA = "6"
$tr = "nnUNetTrainerDeg_250epochs"
$fold = "$nn\results\Dataset501_ACDCdacm\$($tr)__nnUNetPlans__2d\fold_0"
for ($i = 1; $i -le 3 -and -not (Test-Path "$fold\checkpoint_final.pth"); $i++) {
  Write-Host "===== resume try $i  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="
  & (Join-Path $root ".venv\Scripts\nnUNetv2_train.exe") 501 2d 0 -tr $tr --c
  Write-Host "exit $LASTEXITCODE"
}
if (Test-Path "$fold\checkpoint_final.pth") {
  Write-Host "===== eval nnunet_deg  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="
  & $py scripts\nnunet\nnunet_eval.py --model_dir (Split-Path $fold) --cache $cache `
      --testset $tset --out (Join-Path $runs "nnunet_deg") --name nnunet_deg
  Write-Host "exit $LASTEXITCODE"
}
Write-Host "===== done  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="
