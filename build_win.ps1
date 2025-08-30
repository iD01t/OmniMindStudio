# PowerShell Build Script for OmniMind Studio

# --- Configuration ---
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
<?xml version="1.0" encoding="utf-8"?>
<Package
  xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"
  xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10"
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
