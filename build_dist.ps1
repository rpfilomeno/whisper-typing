# Build script for Whisper Typing distribution
# This script uses PyInstaller via uv to create a standalone executable

$projectName = "whisper-typing"
$distPath = "dist/$projectName"

# 1. Cleanup old builds
if (Test-Path "build") {
    Write-Host "Cleaning up build directory..." -ForegroundColor Cyan
    Remove-Item -Path "build" -Recurse -Force
}
if (Test-Path $distPath) {
    Write-Host "Cleaning up old distribution..." -ForegroundColor Cyan
    Remove-Item -Path $distPath -Recurse -Force
}

# 2. Run PyInstaller
Write-Host "Starting build process for $projectName..." -ForegroundColor Green
Write-Host "This may take a few minutes due to heavy dependencies (Torch, Faster-Whisper)..." -ForegroundColor Yellow

uv run pyinstaller `
    --noconfirm `
    --onedir `
    --console `
    --name "$projectName" `
    --collect-all "whisper_typing" `
    --collect-all "textual" `
    --collect-all "faster_whisper" `
    --hidden-import "pynput.keyboard._win32" `
    --hidden-import "pynput.mouse._win32" `
    --paths "src" `
    "src/whisper_typing/__main__.py"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

# 3. Post-build: Copy configuration templates
Write-Host "Build successful! Preparing distribution folder..." -ForegroundColor Green

if (Test-Path "config.json") {
    Write-Host "Copying config.json..." -ForegroundColor Gray
    Copy-Item "config.json" "$distPath/config.json"
}

# Create a template .env file if it doesn't exist in dist
$envTemplate = "# Add your Gemini API Key here`nGEMINI_API_KEY="
if (-not (Test-Path "$distPath/.env")) {
    Write-Host "Creating .env template..." -ForegroundColor Gray
    $envTemplate | Out-File -FilePath "$distPath/.env" -Encoding utf8
}

Write-Host "`nBuild Complete!" -ForegroundColor Green
Write-Host "The executable and its dependencies are located in: $(Get-Item $distPath).FullName" -ForegroundColor Cyan
Write-Host "Run it by executing: $distPath/$projectName.exe" -ForegroundColor Yellow
