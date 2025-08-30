# PowerShell Build Script for OmniMind Studio

# --- Configuration ---
feature/omnimind-studio-final
$AppName = "OmniMind Studio"
$AppVersion = "3.0.0"
$AppFile = "omnimind_studio.py"
$IconFile = "assets/icon.ico" # Make sure you have an icon file here
$DistPath = "dist"
$BuildPath = "build"

# --- Clean previous builds ---
Write-Host "Cleaning up previous build artifacts..."
if (Test-Path $DistPath) {
    Remove-Item -Recurse -Force $DistPath
}
if (Test-Path $BuildPath) {
    Remove-Item -Recurse -Force $BuildPath
}
if (Test-Path "$AppName.spec") {
    Remove-Item -Force "$AppName.spec"
}

# --- Check for dependencies ---
Write-Host "Checking for PyInstaller..."
if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
    Write-Host "PyInstaller not found. Please install it by running: pip install pyinstaller"
    exit 1
}

# --- Run PyInstaller to create the EXE ---
Write-Host "Running PyInstaller to bundle the application..."
pyinstaller --name "$AppName" `
    --onefile `
    --windowed `
    --icon "$IconFile" `
    --add-data "assets;assets" `
    "$AppFile"

if ($LASTEXITCODE -ne 0) {
    Write-Host "PyInstaller failed. See the output above for details."
    exit 1
}

Write-Host "PyInstaller build successful. EXE created in '$DistPath' directory."

# --- (Optional) Create MSIX Installer for Microsoft Store ---
# This is an advanced step for creating a modern Windows application package.
# It provides a better installation experience but requires additional setup.

# Prerequisites for MSIX Packaging:
# 1. Windows SDK: You must install the Windows SDK, which includes the 'makeappx.exe' and 'signtool.exe' tools.
#    - Download the SDK from the Windows Dev Center: https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/
# 2. Code Signing Certificate: To distribute the MSIX package, it must be signed with a trusted code signing certificate.
#    - For development, you can create a self-signed certificate using PowerShell:
#      New-SelfSignedCertificate -DnsName "YourPublisherName" -CertStoreLocation "cert:\CurrentUser\My" -Type CodeSigningCert
#    - For production, you must purchase a certificate from a Certificate Authority (e.g., DigiCert, Sectigo).
# 3. Publisher Name: The 'PublisherName' variable below must exactly match the subject name of your certificate.
#    (e.g., "CN=YourPublisherName")

$CreateMSIX = $false # Set to $true to attempt MSIX creation

if ($CreateMSIX) {
    Write-Host "Attempting to create MSIX package..."

    # IMPORTANT: Change this to match your code signing certificate's subject name.
    $PublisherName = "CN=YourPublisherName"
    $PackageName = "YourCompany.$AppName"
    $PackageDisplayName = "$AppName"
    $PackageVersion = "$AppVersion.0"
    $InstallerName = "$AppName-$AppVersion-Installer.msix"
    $ManifestPath = Join-Path $DistPath "AppxManifest.xml"

    # Create a basic AppX Manifest
    # Note: You'll need to create the referenced logo assets (StoreLogo.png, etc.)
    $ManifestContent = @"

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
main
<?xml version="1.0" encoding="utf-8"?>
<Package
  xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"
  xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10"
feature/omnimind-studio-final
  IgnorableNamespaces="uap">

  <Identity
    Name="$PackageName"
    Publisher="$PublisherName"
    Version="$PackageVersion" />

  <Properties>
    <DisplayName>$PackageDisplayName</DisplayName>
    <PublisherDisplayName>Your Company</PublisherDisplayName>
    <Logo>assets\StoreLogo.png</Logo>
  </Properties>

  <Dependencies>
    <TargetDeviceFamily Name="Windows.Desktop" MinVersion="10.0.17763.0" MaxVersionTested="10.0.22000.0" />
  </Dependencies>

  <Resources>
    <Resource Language="x-generate"/>
  </Resources>

  <Applications>
    <Application Id="App"
      Executable="$AppName\$AppName.exe"
      EntryPoint="Windows.FullTrustApplication">
      <uap:VisualElements
        DisplayName="$PackageDisplayName"
        Description="$AppName"
        BackgroundColor="transparent"
        Square150x150Logo="assets\Square150x150Logo.png"
        Square44x44Logo="assets\Square44x44Logo.png">
        <uap:DefaultTile Wide310x150Logo="assets\Wide310x150Logo.png" />
      </uap:VisualElements>
    </Application>
  </Applications>
</Package>
"@
    $ManifestContent | Out-File -Encoding utf8 -FilePath $ManifestPath

    # Pre-flight checks for MSIX packaging tools
    $MakeAppxPath = (Get-Command makeappx -ErrorAction SilentlyContinue).Source
    $SignToolPath = (Get-Command signtool -ErrorAction SilentlyContinue).Source

    if (-not $MakeAppxPath) {
        Write-Warning "makeappx.exe not found. This tool is required for MSIX packaging and is part of the Windows SDK."
        Write-Warning "Skipping MSIX creation. Please install the Windows SDK and add it to your PATH."
    }
    elseif (-not $SignToolPath) {
        Write-Warning "signtool.exe not found. This tool is required for signing the MSIX package and is part of the Windows SDK."
        Write-Warning "Skipping MSIX signing. The package will be created but will not be installable until signed."
    }
    else {
        Write-Host "Creating MSIX package..."
        makeappx pack /d "$DistPath" /p "$DistPath\$InstallerName" /o

        Write-Host "Signing MSIX package..."
        Write-Host "NOTE: This next step will likely fail unless you have a valid .pfx certificate and have configured the command below."
        # This is a placeholder. You must replace 'YourCert.pfx' and provide the password.
        # signtool sign /a /fd SHA256 /f YourCert.pfx /p YourPassword "$DistPath\$InstallerName"

        Write-Host "MSIX package created (unsigned) at '$DistPath\$InstallerName'. You must sign it to install."
    }
}

Write-Host "Build process finished."

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
main
