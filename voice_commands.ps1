$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
Add-Type -AssemblyName System.Speech
$recognizerInfo = [System.Speech.Recognition.SpeechRecognitionEngine]::InstalledRecognizers() |
    Where-Object { $_.Culture.Name -eq "zh-TW" } |
    Select-Object -First 1
if ($null -eq $recognizerInfo) { throw "Traditional Chinese speech recognizer is unavailable" }
$recognizer = [System.Speech.Recognition.SpeechRecognitionEngine]::new($recognizerInfo)
$phrases = @("6ZaL5aeL6YyE6KO9", "6ZaL5aeL6Z2c5oWL6YyE6KO9", "57WQ5p2f6YyE6KO9") |
    ForEach-Object { [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($_)) }
foreach ($phrase in $phrases) {
    $builder = [System.Speech.Recognition.GrammarBuilder]::new($phrase)
    $builder.Culture = $recognizerInfo.Culture
    $recognizer.LoadGrammar([System.Speech.Recognition.Grammar]::new($builder))
}
$recognizer.SetInputToDefaultAudioDevice()
Register-ObjectEvent -InputObject $recognizer -EventName SpeechRecognized -Action {
    if ($Event.SourceEventArgs.Result.Confidence -ge 0.72) {
        [Console]::Out.WriteLine($Event.SourceEventArgs.Result.Text)
        [Console]::Out.Flush()
    }
} | Out-Null
$recognizer.RecognizeAsync([System.Speech.Recognition.RecognizeMode]::Multiple)
[Console]::Out.WriteLine("__READY__")
[Console]::Out.Flush()
try {
    while ($true) { Start-Sleep -Milliseconds 250 }
} finally {
    $recognizer.RecognizeAsyncCancel()
    $recognizer.Dispose()
}
