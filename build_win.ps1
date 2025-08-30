# PowerShell Build Script for OmniMind Studio

# --- Configuration ---
$AppName = "OmniMindStudio"
$AppVersion = "3.0.0.0" # MSIX requires a 4-part version
$ScriptFile = "omnimind_studio.py"
$DistPath = "dist"
$BuildPath = "build"
$InstallerPath = "installer"
$VenvPath = ".venv"
$PythonExe = Join-Path -Path $VenvPath -ChildPath "Scripts/python.exe"

# --- MSIX Configuration ---
$PublisherName = "CN=YourPublisherName" # IMPORTANT: Replace with your publisher name from your certificate
$PublisherDisplayName = "Your Company"
$AppId = "YourCompany.OmniMindStudio" # IMPORTANT: Replace with your unique App ID

# --- Pre-build Checks ---
Write-Host "Starting OmniMind Studio build process..."
if (-not (Test-Path $ScriptFile)) {
    Write-Host "Error: Main script '$ScriptFile' not found." -ForegroundColor Red; exit 1
}
if (-not (Test-Path $PythonExe)) {
    Write-Host "Python venv not found. Running script to generate it..."
    & python $ScriptFile
    if ($LASTEXITCODE -ne 0) { Write-Host "Failed to create venv." -ForegroundColor Red; exit 1 }
    Write-Host "Venv created. Continuing build."
}

# --- Install PyInstaller ---
Write-Host "Ensuring PyInstaller is installed..."
& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install pyinstaller

# --- Build with PyInstaller ---
Write-Host "Running PyInstaller to create the executable..."
$pyinstaller_options = @(
    "--noconfirm", "--onefile", "--windowed", "--name $AppName",
    "--distpath `"$DistPath`"", "--workpath `"$BuildPath`"",
    "--add-data `"claudex_ultra.db;.`""
)
if (Test-Path "assets/icon.ico") { $pyinstaller_options += "--icon `"assets/icon.ico`"" }
$command = "& `"$PythonExe`" -m PyInstaller " + ($pyinstaller_options -join " ") + " `"$ScriptFile`""
Write-Host "Executing: $command"
Invoke-Expression $command
if ($LASTEXITCODE -ne 0) { Write-Host "PyInstaller build failed." -ForegroundColor Red; exit 1 }
Write-Host "Build successful! Executable is at: $DistPath\$AppName.exe" -ForegroundColor Green

# --- MSIX Installer Creation ---
Write-Host "Starting MSIX packaging process..."

# Function to generate the AppxManifest.xml
function Generate-AppxManifest {
    param($ManifestPath)
    $xml = @"
<?xml version="1.0" encoding="utf-8"?>
<Package
  xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"
  xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10"
  xmlns:rescap="http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities"
  IgnorableNamespaces="uap rescap">

  <Identity
    Name="$AppId"
    Version="$AppVersion"
    Publisher="$PublisherName"
    ProcessorArchitecture="x64" />

  <Properties>
    <DisplayName>$AppName</DisplayName>
    <PublisherDisplayName>$PublisherDisplayName</PublisherDisplayName>
    <Logo>assets\Logo50x50.png</Logo>
  </Properties>

  <Dependencies>
    <TargetDeviceFamily Name="Windows.Desktop" MinVersion="10.0.17763.0" MaxVersionTested="10.0.22621.0" />
  </Dependencies>

  <Resources>
    <Resource Language="en-us" />
  </Resources>

  <Applications>
    <Application Id="App" Executable="$AppName.exe" EntryPoint="Windows.FullTrustApplication">
      <uap:VisualElements
        DisplayName="$AppName"
        Description="The Indie-Pro AI Desktop That Beats Claude for Windows."
        Square150x150Logo="assets\Logo150x150.png"
        Square44x44Logo="assets\Logo44x44.png"
        BackgroundColor="transparent">
        <uap:DefaultTile Wide310x150Logo="assets\Logo310x150.png" />
      </uap:VisualElements>
    </Application>
  </Applications>

  <Capabilities>
    <rescap:Capability Name="runFullTrust" />
  </Capabilities>
</Package>
"@
    $xml | Out-File -FilePath $ManifestPath -Encoding utf8
}

# Create a directory for the MSIX package content
$PackageLayout = Join-Path -Path $BuildPath -ChildPath "msix_layout"
if (Test-Path $PackageLayout) { Remove-Item -Recurse -Force $PackageLayout }
New-Item -Path $PackageLayout -ItemType Directory | Out-Null

# Copy EXE and assets into the layout directory
Copy-Item -Path (Join-Path -Path $DistPath -ChildPath "$AppName.exe") -Destination $PackageLayout
New-Item -Path (Join-Path -Path $PackageLayout -ChildPath "assets") -ItemType Directory | Out-Null
# Copy placeholder assets if they exist, user should replace these
Get-ChildItem -Path "assets" -Filter "*.png" | ForEach-Object { Copy-Item -Path $_.FullName -Destination (Join-Path -Path $PackageLayout -ChildPath "assets") }

# Generate the manifest
$ManifestFile = Join-Path -Path $PackageLayout -ChildPath "AppxManifest.xml"
Generate-AppxManifest -ManifestPath $ManifestFile
Write-Host "Generated AppxManifest.xml at $ManifestFile"

# Create installer directory
if (-not (Test-Path $InstallerPath)) { New-Item -Path $InstallerPath -ItemType Directory | Out-Null }

# Package with makeappx.exe
Write-Host "Attempting to package with makeappx.exe..."
Write-Host "NOTE: This requires the Windows 10/11 SDK to be installed and in your PATH."
$MSIXFile = Join-Path -Path $InstallerPath -ChildPath "$AppName.msix"
try {
    makeappx pack /d "$PackageLayout" /p "$MSIXFile" /o
    Write-Host "MSIX package created successfully at $MSIXFile" -ForegroundColor Green
} catch {
    Write-Host "makeappx.exe failed. Please ensure the Windows SDK is installed and configured in your PATH." -ForegroundColor Yellow
    Write-Host "You can create the package manually by running:" -ForegroundColor Yellow
    Write-Host "makeappx pack /d `"$PackageLayout`" /p `"$MSIXFile`" /o" -ForegroundColor Yellow
}

# (Optional) Signing the package
Write-Host "Signing step is a placeholder. To sign the package, you need a trusted code signing certificate."
Write-Host "You would use a command like:"
Write-Host "signtool.exe sign /fd SHA256 /a /f YourCertificate.pfx /p YourPassword `"$MSIXFile`"" -ForegroundColor Cyan

Write-Host "Build script finished."
