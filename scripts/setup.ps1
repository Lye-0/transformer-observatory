param([switch]$SkipFrontend)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$previousUvEnvironment = @{}
foreach ($variableName in @('UV_PYTHON_INSTALL_DIR', 'UV_PYTHON_BIN_DIR', 'UV_CACHE_DIR')) {
    $previousUvEnvironment[$variableName] = [Environment]::GetEnvironmentVariable($variableName, 'Process')
}
Push-Location $repoRoot
try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'uvが必要です。https://docs.astral.sh/uv/getting-started/installation/ を参照してください。' }
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $repoRoot '.tools\python'
    $env:UV_PYTHON_BIN_DIR = Join-Path $repoRoot '.tools\bin'
    $env:UV_CACHE_DIR = Join-Path $repoRoot '.tools\uv-cache'
    uv python install 3.13.15
    if ($LASTEXITCODE -ne 0) { throw 'Pythonの準備に失敗しました' }
    uv sync --project backend --python 3.13.15 --locked
    if ($LASTEXITCODE -ne 0) { throw 'バックエンド依存関係の準備に失敗しました' }
    if (-not $SkipFrontend) {
        if (-not (Get-Command mise.exe -ErrorAction SilentlyContinue)) { throw 'miseが必要です。' }
        mise.exe trust
        if ($LASTEXITCODE -ne 0) { throw 'mise設定の確認に失敗しました' }
        mise.exe install
        if ($LASTEXITCODE -ne 0) { throw 'Node・pnpmの準備に失敗しました' }
        mise.exe exec -- pnpm --dir frontend install --frozen-lockfile
        if ($LASTEXITCODE -ne 0) { throw 'フロントエンド依存関係の準備に失敗しました' }
        mise.exe exec -- pnpm --dir frontend run build
        if ($LASTEXITCODE -ne 0) { throw 'フロントエンドのビルドに失敗しました' }
    }
    Write-Host '準備ができました。 .\scripts\start.ps1 で起動できます。'
} finally {
    foreach ($variableName in $previousUvEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($variableName, $previousUvEnvironment[$variableName], 'Process')
    }
    Pop-Location
}
