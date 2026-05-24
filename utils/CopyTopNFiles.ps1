<#
.SYNOPSIS
    Copie les N premiers fichiers d'un répertoire source vers un répertoire de destination.
.DESCRIPTION
    Ce script copie les N premiers fichiers (triés par nom) depuis un dossier source vers un dossier de destination.
    Les sous-dossiers ne sont pas inclus (seulement les fichiers à la racine du répertoire source).
.PARAMETER SourcePath
    Chemin du répertoire source.
.PARAMETER DestinationPath
    Chemin du répertoire de destination.
.PARAMETER NumberOfFiles
    Nombre de fichiers à copier (par défaut : 10).
.PARAMETER Overwrite
    Si présent, écrase les fichiers existants dans la destination.
.EXAMPLE
    .\CopyTopNFiles.ps1 -SourcePath "C:\Temp\Source" -DestinationPath "C:\Temp\Dest" -NumberOfFiles 5
#>

param (
    [Parameter(Mandatory=$true)]
    [string]$SourcePath,

    [Parameter(Mandatory=$true)]
    [string]$DestinationPath,

    [int]$NumberOfFiles = 10,

    [switch]$Overwrite
)

# Vérifier que le répertoire source existe
if (-not (Test-Path -Path $SourcePath -PathType Container)) {
    Write-Error "Le répertoire source '$SourcePath' n'existe pas."
    exit 1
}

# Créer le répertoire de destination s'il n'existe pas
if (-not (Test-Path -Path $DestinationPath -PathType Container)) {
    New-Item -ItemType Directory -Path $DestinationPath -Force | Out-Null
    Write-Host "Répertoire de destination créé : $DestinationPath"
}

# Récupérer les N premiers fichiers (triés par nom)
$filesToCopy = Get-ChildItem -Path $SourcePath -File |
               Sort-Object -Property Name |
               Select-Object -First $NumberOfFiles

if ($filesToCopy.Count -eq 0) {
    Write-Host "Aucun fichier trouvé dans le répertoire source."
    exit
}

# Copier les fichiers
foreach ($file in $filesToCopy) {
    $destinationFile = Join-Path -Path $DestinationPath -ChildPath $file.Name
    try {
        Copy-Item -Path $file.FullName -Destination $destinationFile -Force:$Overwrite -ErrorAction Stop
        Write-Host "Copié : $($file.Name)"
    }
    catch {
        Write-Error "Erreur lors de la copie de $($file.Name) : $_"
    }
}

Write-Host "Copie terminée. $($filesToCopy.Count) fichiers copiés vers $DestinationPath."
