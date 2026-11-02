# Rebuild every table, figure input and number macro of the paper from the finished runs,
# then compile paper_access.tex. Statistics go to work\stats_review_final first and are
# copied to literature_review\results only if every script succeeded.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$py   = Join-Path $root ".venv\Scripts\python.exe"
$tmp  = Join-Path $root "work\stats_review_final"
$res  = Resolve-Path (Join-Path $root "..\literature_review\results")
$env:PYTHONPATH = "$root;$root\scripts"
if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force -Confirm:$false }
New-Item -ItemType Directory -Force $tmp | Out-Null

function Py($label) {
  Write-Host "===== $label $(Get-Date -Format HH:mm:ss) ====="
  & $py @args
  if ($LASTEXITCODE -ne 0) { throw "$label failed" }
}
Py "make_tables (all seeds)" scripts\make_tables.py --runs_root work\runs --estimator work\estimator --paper_results $tmp --all_seeds --tag review
Py "stats_revision"   scripts\stats_revision.py --runs_root work\runs --cache work\acdc_cache --paper_results $tmp
Py "stats_structures" scripts\stats_structures.py --runs_root work\runs --cache work\acdc_cache --paper_results $tmp
Py "stats_external"   scripts\stats_external.py --work work --paper_results $tmp

Write-Host "===== copy to $res ====="
Copy-Item "$tmp\*" $res -Recurse -Force
Py "stats_review (probes, slices, HD95)" scripts\stats_review.py --work work --paper_results $res

Write-Host "===== compile ====="
Set-Location (Split-Path $res)
foreach ($c in "pdflatex", "bibtex", "pdflatex", "pdflatex") {
  if ($c -eq "bibtex") { & bibtex paper_access | Out-Null } else { & pdflatex -interaction=nonstopmode paper_access.tex | Out-Null }
}
Select-String -Path paper_access.log -Pattern "undefined|Output written" | ForEach-Object Line
Select-String -Path paper_access.log -Pattern "XX\.X|pending" | Measure-Object | ForEach-Object { "placeholder hits in log: $($_.Count)" }
Write-Host "done $(Get-Date -Format HH:mm:ss)"
