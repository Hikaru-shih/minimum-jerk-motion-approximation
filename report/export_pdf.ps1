$ErrorActionPreference = 'Stop'
$source = Join-Path $PSScriptRoot 'final_motion_decomposition_report.docx'
$target = Join-Path $PSScriptRoot 'final_motion_decomposition_report.pdf'
$temporaryDirectory = Join-Path (Split-Path $PSScriptRoot -Parent) 'tmp'
$temporarySource = Join-Path $temporaryDirectory ('word-layout-' + [guid]::NewGuid().ToString() + '.docx')
$temporaryPdf = [IO.Path]::ChangeExtension($temporarySource, '.pdf')
New-Item -ItemType Directory -Path $temporaryDirectory -Force | Out-Null
Copy-Item -LiteralPath $source -Destination $temporarySource
Write-Output 'Starting Word with a temporary read-only copy.'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$document = $null
try {
    Write-Output 'Opening report.'
    $document = $word.Documents.Open($temporarySource, $false, $true)
    Write-Output 'Exporting PDF.'
    $document.ExportAsFixedFormat($temporaryPdf, 17)
    Copy-Item -LiteralPath $temporaryPdf -Destination $target -Force
    Write-Output "Equations: $($document.OMaths.Count)"
    Write-Output $target
} finally {
    if ($null -ne $document) { $document.Close(0) }
    $word.Quit()
    Remove-Item -LiteralPath $temporarySource
    if (Test-Path -LiteralPath $temporaryPdf) { Remove-Item -LiteralPath $temporaryPdf }
}
