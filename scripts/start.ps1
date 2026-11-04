param([int]$Port = 8765, [switch]$NoBrowser, [switch]$DiagnoseStartup)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExecutable = Join-Path $repoRoot 'backend\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable)) { throw '先に .\scripts\setup.ps1 を実行してください。' }
if (-not (Test-Path -LiteralPath (Join-Path $repoRoot 'frontend\dist\index.html'))) { throw 'UIが未ビルドです。 .\scripts\setup.ps1 を実行してください。' }
Push-Location $repoRoot
try {
    $launchArgs = @('-m', 'transformer_observatory', '--port', "$Port")
    if ($NoBrowser) { $launchArgs += '--no-browser' }
    if ($DiagnoseStartup) {
        $diagnosticPath = Join-Path $repoRoot ('.tools\startup-diagnostics\startup-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + "-$PID.log")
        $launchArgs += @('--startup-log', $diagnosticPath)
        Write-Host "起動診断ログ: $diagnosticPath"
    }
    & $pythonExecutable @launchArgs
    if ($LASTEXITCODE -ne 0) { throw "バックエンドが終了コード $LASTEXITCODE で停止しました" }
} finally { Pop-Location }
