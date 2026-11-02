# Revision, third queue. Starts when the Swin-UNETR revision queue has finished, so that
# at most two jobs share the GPU. Each step is independent: a failure is logged and the
# queue moves on.
#  1. probes fitted on validation subjects, including conditioned models
#  2. official nnU-Net 2D (default augmentation; plus our degradation simulator), 250 epochs
#  3. nnU-Net evaluation on the fixed degraded test set
#  4. Tent test-time adaptation of the augmentation-only models (U-Net, then Swin-UNETR)
$root  = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$py    = Join-Path $root ".venv\Scripts\python.exe"
$work  = Join-Path $root "work"
$cache = Join-Path $work "acdc_cache"; $calib = Join-Path $work "calib.json"
$estd  = Join-Path $work "estimator"; $tset = Join-Path $work "testset"; $vset = Join-Path $work "valset"
$runs  = Join-Path $work "runs"; $paper = Join-Path $root "..\literature_review\results"
$nn    = Join-Path $work "nnunet"
$env:nnUNet_raw = "$nn\raw"; $env:nnUNet_preprocessed = "$nn\preprocessed"; $env:nnUNet_results = "$nn\results"
$env:PYTHONPATH = $root; $env:DACM_CALIB = $calib; $env:nnUNet_n_proc_DA = "10"

function Step($label, [scriptblock]$body) {
  Write-Host "===== $label  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="
  try { & $body; if ($LASTEXITCODE -ne 0) { Write-Host "!!!!! $label exited with code $LASTEXITCODE" } }
  catch { Write-Host "!!!!! $label failed: $_" }
}

# Wait until no other revision queue is running (U-Net, Swin-UNETR, or the chain that runs them).
while (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_revision_(unet|swin|chain)\.ps1' }) {
  Start-Sleep -Seconds 120
}

Step "probes" {
  if (-not (Test-Path (Join-Path $work "probe_val.done"))) {
    & $py scripts\probe_val.py --valset $vset --testset $tset --estimator $estd --runs_root $runs --paper_results $paper
    if ($LASTEXITCODE -eq 0) { Set-Content (Join-Path $work "probe_val.done") "ok" }
  }
}

$nnDir = "$nn\results\Dataset501_ACDCdacm"
foreach ($tr in "nnUNetTrainer_250epochs", "nnUNetTrainerDeg_250epochs") {
  Step "nnU-Net train $tr" {
    if (-not (Test-Path "$nnDir\$($tr)__nnUNetPlans__2d\fold_0\checkpoint_final.pth")) {
      $c = "$nnDir\$($tr)__nnUNetPlans__2d\fold_0\checkpoint_latest.pth"
      $extra = @(); if (Test-Path $c) { $extra = @("--c") }
      & (Join-Path $root ".venv\Scripts\nnUNetv2_train.exe") 501 2d 0 -tr $tr @extra
    }
  }
}
foreach ($pair in @(@("nnUNetTrainer_250epochs", "nnunet_default"), @("nnUNetTrainerDeg_250epochs", "nnunet_deg"))) {
  Step "nnU-Net eval $($pair[1])" {
    if (-not (Test-Path (Join-Path $runs "$($pair[1])\eval_seg.csv"))) {
      & $py scripts\nnunet\nnunet_eval.py --model_dir "$nnDir\$($pair[0])__nnUNetPlans__2d" --cache $cache `
          --testset $tset --out (Join-Path $runs $pair[1]) --name $pair[1]
    }
  }
}

Step "Tent U-Net" {
  $r = @("unet_aug", "unet_aug_s1", "unet_aug_s2", "unet_aug_s3", "unet_aug_s4") |
       Where-Object { Test-Path (Join-Path $runs "$_\eval_seg.csv") } | ForEach-Object { Join-Path $runs $_ }
  & $py scripts\tent_eval.py --cache $cache --testset $tset --valset $vset --runs @r
}
Step "Tent Swin-UNETR" {
  $r = @("swinunetr_aug", "swinunetr_aug_s1", "swinunetr_aug_s2", "swinunetr_aug_s3", "swinunetr_aug_s4") |
       Where-Object { Test-Path (Join-Path $runs "$_\eval_seg.csv") } | ForEach-Object { Join-Path $runs $_ }
  & $py scripts\tent_eval.py --cache $cache --testset $tset --valset $vset --runs @r
}
Write-Host "Extra revision queue finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
