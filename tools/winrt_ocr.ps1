param(
    [Parameter(Mandatory = $true)]
    [string]$InputJson,

    [Parameter(Mandatory = $true)]
    [string]$OutputJson
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# This helper only calls the Windows WinRT OCR engine.
# The Python converter writes image paths to JSON; this script reads them,
# runs OCR image by image, and writes a JSON result file.

Add-Type -AssemblyName System.Runtime.WindowsRuntime

[Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime] | Out-Null
[Windows.Globalization.Language, Windows.Globalization, ContentType = WindowsRuntime] | Out-Null
[Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime] | Out-Null
[Windows.Storage.FileAccessMode, Windows.Storage, ContentType = WindowsRuntime] | Out-Null
[Windows.Storage.Streams.IRandomAccessStream, Windows.Foundation, ContentType = WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType = WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType = WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.BitmapPixelFormat, Windows.Graphics.Imaging, ContentType = WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.BitmapAlphaMode, Windows.Graphics.Imaging, ContentType = WindowsRuntime] | Out-Null

function Await-WinRt {
    param(
        [Parameter(Mandatory = $true)]
        $AsyncOperation,

        [Parameter(Mandatory = $true)]
        [Type]$ResultType
    )

    # WinRT async objects are not plain .NET tasks. Convert them through
    # WindowsRuntimeSystemExtensions.AsTask<T>, then wait synchronously.
    $method = [System.WindowsRuntimeSystemExtensions].GetMethods() |
        Where-Object {
            $_.Name -eq "AsTask" -and
            $_.IsGenericMethod -and
            $_.GetParameters().Count -eq 1 -and
            $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
        } |
        Select-Object -First 1

    if ($null -eq $method) {
        throw "Cannot find WindowsRuntimeSystemExtensions.AsTask<T>."
    }

    $task = $method.MakeGenericMethod($ResultType).Invoke($null, @($AsyncOperation))
    return $task.GetAwaiter().GetResult()
}

function New-OcrEngine {
    # Prefer the user's profile languages. On this machine this usually
    # covers both Chinese and English OCR.
    $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
    if ($null -ne $engine) {
        return $engine
    }

    # Fall back to Simplified Chinese, then English.
    foreach ($tag in @("zh-Hans", "en-US")) {
        try {
            $language = [Windows.Globalization.Language]::new($tag)
            $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($language)
            if ($null -ne $engine) {
                return $engine
            }
        }
        catch {
            continue
        }
    }

    throw "Windows OCR engine is unavailable for the current user profile."
}

function Invoke-ImageOcr {
    param(
        [Parameter(Mandatory = $true)]
        [Windows.Media.Ocr.OcrEngine]$Engine,

        [Parameter(Mandatory = $true)]
        [string]$ImagePath
    )

    $file = Await-WinRt ([Windows.Storage.StorageFile]::GetFileFromPathAsync($ImagePath)) ([Windows.Storage.StorageFile])
    $stream = Await-WinRt ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    try {
        $decoder = Await-WinRt ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
        $bitmap = Await-WinRt ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
        $converted = [Windows.Graphics.Imaging.SoftwareBitmap]::Convert(
            $bitmap,
            [Windows.Graphics.Imaging.BitmapPixelFormat]::Bgra8,
            [Windows.Graphics.Imaging.BitmapAlphaMode]::Premultiplied
        )
        $result = Await-WinRt ($Engine.RecognizeAsync($converted)) ([Windows.Media.Ocr.OcrResult])
        return $result.Text
    }
    finally {
        if ($null -ne $stream) {
            $stream.Dispose()
        }
    }
}

$payload = Get-Content -Raw -Encoding UTF8 -LiteralPath $InputJson | ConvertFrom-Json
$engine = New-OcrEngine
$results = @()

foreach ($item in $payload.items) {
    $id = [string]$item.id
    $path = [string]$item.path
    $text = ""
    $errorMessage = ""

    try {
        $text = Invoke-ImageOcr -Engine $engine -ImagePath $path
    }
    catch {
        $errorMessage = $_.Exception.Message
    }

    $results += [PSCustomObject]@{
        id = $id
        path = $path
        text = $text
        error = $errorMessage
    }
}

$output = [PSCustomObject]@{
    ok = $true
    count = $results.Count
    results = $results
}

$output |
    ConvertTo-Json -Depth 6 |
    Set-Content -LiteralPath $OutputJson -Encoding UTF8
