# FaceMatch launcher. Defaults to http://127.0.0.1:7861 -- see app/app.py's
# --host/--port/--share for overrides.
# Isolated venv - deliberately NOT the ComfyUI or ai-toolkit environments.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "venv missing at $py -- see README.md Quick start" }

# insightface pulls in a torch-free stack, but clear this in case the parent
# shell set it (Electron apps do)
$env:ELECTRON_RUN_AS_NODE = $null

Write-Host "FaceMatch -> http://127.0.0.1:7861" -ForegroundColor Green
& $py (Join-Path $root "app\app.py") @args
