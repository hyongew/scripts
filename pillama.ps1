<#
Starts the llama.cpp server for a chosen model, waits until it is ready,
then launches pi to connect to it.

Model files are found recursively in the directory set by the environment variable
PILLAMA_MODELS_DIR.

Flag names are not case-sensitive.
  pillama                       show a model menu
  pillama -silent               llama is served silently and stopped when pi exits
  pillama -ramthreshold 4       warn if the host-RAM leaves under that many GiB free

Ctx sizes: [65536, 98304, 131072, 163840, 196608, 228376, 262144]
#>
[CmdletBinding()]
param(
    [switch]$Silent,
    [double]$RamThreshold = 16
)

$modelsDir = $env:PILLAMA_MODELS_DIR
if (-not $modelsDir) {
    throw 'PILLAMA_MODELS_DIR not set'
}
if (-not (Test-Path -LiteralPath $modelsDir -PathType Container)) {
    throw "Models directory not found: $modelsDir"
}
$port = 8080

$forceReplacePortOwner = $false
$startupTimeoutSec = 600

$vramReserve = @(0.125, 0.5)   # GiB of vram to reserve on GPU 0, and every other GPU

# Controls how PowerShell reacts to errors that aren’t given their own -ErrorAction setting
$ErrorActionPreference = 'Stop'

$common = @('--jinja', '--flash-attn', 'on',
            '--log-verbosity', '4',
            '--cache-ram', '2048',
            '--port', "$port")

$cfg = [ordered]@{
    'qwen3.8-27b' = @{   # in chat - <|think_off|>, <|think_low|>, <|think_medium|>, <|think_xhigh|>
        Desc = '27b dense'
        File = 'Qwen3.8-27B-UD-Q4_K_XL.gguf'
        # Draft = 'mtp-Qwen3.8-27B-Q4_0.gguf'
        Args = @('--ctx-size', '163840', '--cache-type-k', 'f16', '--cache-type-v', 'f16',
                 '--ubatch-size', '256',
                 '--chat-template-file', "$modelsDir\chat_templates\chat_template_qwen.jinja",
                 '--reasoning-format', 'deepseek',   # needed for froggeric chat template
                 '--spec-type', 'draft-mtp,ngram-mod', '--model-draft', '{draft}',
                 '--spec-draft-type-k', 'q4_0', '--spec-draft-type-v', 'q4_0',
                 '--spec-draft-p-min', '0.8',
                 '--spec-ngram-mod-n-match', '24',
                 '--spec-ngram-mod-n-min', '48',
                 '--spec-ngram-mod-n-max', '64',
                # --- thinking mode ---
                 '--temp', '1.0', '--top-p', '0.95', '--top-k', '20', '--min-p', '0.0',
                 '--presence-penalty', '0.0', '--repeat-penalty', '1.0',
                 '--reasoning-effort', 'xhigh',
                # --- non-thinking mode ---
                #  '--temp', '0.7', '--top-p', '0.80', '--top-k', '20', '--min-p', '0.0',
                #  '--presence-penalty', '1.5', '--repeat-penalty', '1.0',
                #  '--reasoning-effort', 'off',
                 '--reasoning-preserve')
    }
    'gemma4-26b-a4b' = @{
        Desc = '26b MoE a4b'
        File = 'gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf'
        # Draft = 'mtp-gemma-4-26B-A4B-it-Q4_0.gguf'
        Mmproj = 'mmproj-BF16.gguf'
        Args = @('--ctx-size', '262144', '--cache-type-k', 'f16', '--cache-type-v', 'f16',
                 '--batch-size', '1024', '--ubatch-size', '512', # change to 256 if OOM
                 '--mmproj', '{mmproj}', '--no-mmproj-offload',
                 '--chat-template-file', "$modelsDir\chat_templates\chat_template_gemma4.jinja",
                #  '--spec-type', 'draft-mtp', '--model-draft', '{draft}',
                #  '--spec-draft-type-k', 'q4_0', '--spec-draft-type-v', 'q4_0',
                 '--temp', '1.0', '--top-p', '0.95', '--top-k', '64')
    }
    'gemma-31b' = @{
        Desc = '31b dense'
        File = 'Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf'
        # Draft = 'mtp-gemma-4-31B-it.gguf'
        # Mmproj = 'mmproj-Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-BF16.gguf'
        Args = @('--ctx-size', '65536', '--cache-type-k', 'f16', '--cache-type-v', 'f16',
                 '--mmproj', '{mmproj}', '--no-mmproj-offload',
                 '--chat-template-file', "$modelsDir\chat_templates\chat_template_gemma4.jinja",
                #  '--spec-type', 'draft-mtp', '--model-draft', '{draft}',
                #  '--spec-draft-type-k', 'q4_0', '--spec-draft-type-v', 'q4_0',
                #  '--spec-draft-p-min', '0.8',
                 '--temp', '0.6', '--top-p', '0.9', '--top-k', '64', '--min-p', '0.05',
                 '--repeat-penalty', '1.1')
    }
    'qwen3.8-27b-q6-kv8' = @{   # in chat - <|think_off|>, <|think_low|>, <|think_medium|>, <|think_xhigh|>
        Desc = '27b dense, all GPU, Q8 KV'
        File = 'Qwen3.8-27B-UD-Q6_K.gguf'
        # EmbeddedMtp = $true
        Args = @('--ctx-size', '131072', '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0',
                 '--ubatch-size', '256',
                 '--fit', 'off', '--gpu-layers', '999',
                 '--chat-template-file', "$modelsDir\chat_templates\chat_template_qwen.jinja",
                 '--reasoning-format', 'deepseek',   # needed for froggeric chat template
                 '--spec-type', 'draft-mtp,ngram-mod',
                 '--spec-draft-n-max', '3',
                 '--spec-draft-p-min', '0.8',
                 '--spec-ngram-mod-n-match', '24',
                 '--spec-ngram-mod-n-min', '48',
                 '--spec-ngram-mod-n-max', '64',
                # --- thinking mode ---
                 '--temp', '1.0', '--top-p', '0.95', '--top-k', '20', '--min-p', '0.0',
                 '--presence-penalty', '0.0', '--repeat-penalty', '1.0',
                 '--reasoning-effort', 'xhigh',
                # --- non-thinking mode ---
                #  '--temp', '0.7', '--top-p', '0.80', '--top-k', '20', '--min-p', '0.0',
                #  '--presence-penalty', '1.5', '--repeat-penalty', '1.0',
                #  '--reasoning-effort', 'off',
                 '--reasoning-preserve')
    }
    'qwen3.8-flash-next' = @{   # in chat - <|think_off|>, <|think_low|>, <|think_medium|>, <|think_xhigh|>
        Desc = '125b a6b n51b'
        File = 'Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf'
        Args = @('--ctx-size', '131072', '--cache-type-k', 'f16', '--cache-type-v', 'f16',
                #  '--ubatch-size', '256',
                 '--lazy-mode', 'on', '--load-mode', 'mmap',
                 '--fit', 'off', '--gpu-layers', '49', '--tensor-split', '9,40',
                 '--override-tensor', 'blk\.8\.ffn_down.*=CUDA1,blk\.13\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.14\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.15\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.16\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.17\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.18\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.19\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.20\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.21\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.22\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.23\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.24\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.25\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.26\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.27\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.28\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.29\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.30\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.31\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.32\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.33\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.34\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.35\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.36\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.37\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.38\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.39\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.40\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.41\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.42\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.43\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.44\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.45\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.46\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.47\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.48\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU',
                #  '--chat-template-file', "$modelsDir\chat_templates\chat_template_qwen.jinja",
                 '--reasoning-format', 'deepseek',   # needed for froggeric chat template
                 '--spec-type', 'ngram-mod', # '--model-draft', '{draft}',
                 '--spec-ngram-mod-n-match', '24',
                 '--spec-ngram-mod-n-min', '48',
                 '--spec-ngram-mod-n-max', '64',
                # --- thinking mode ---
                 '--temp', '1.0', '--top-p', '0.95', '--top-k', '20', '--min-p', '0.0',
                 '--presence-penalty', '0.0', '--repeat-penalty', '1.0',
                 '--reasoning-effort', 'medium',
                # --- non-thinking mode ---
                #  '--temp', '0.7', '--top-p', '0.80', '--top-k', '20', '--min-p', '0.0',
                #  '--presence-penalty', '1.5', '--repeat-penalty', '1.0',
                #  '--reasoning-effort', 'off',
                 '--reasoning-preserve')
    }
    'muse-glimmer-30b' = @{   # system prompt - "Reasoning strength: low / medium / (high) / xhigh"
        Desc = '30b dense'
        File = 'Muse-Glimmer-30B-UD-Q5_K_M.gguf'
        Args = @('--ctx-size', '131072', '--cache-type-k', 'f16', '--cache-type-v', 'f16',
                 '--chat-template-file', "$modelsDir\chat_templates\chat_template_glimmer.jinja",
                 '--temp', '1.0', '--top-p', '0.95', '--top-k', '64')
    }
}

# ---------------------------------------------------------------- model files

# Find a file recursively anywhere under $modelsDir.
$script:fileIndex = $null
function Resolve-ModelFile {
    param([Parameter(Mandatory)][string]$Name, [string]$What = 'model file')

    if ($null -eq $script:fileIndex) {
        $script:fileIndex = @(Get-ChildItem -LiteralPath $modelsDir -Recurse -File -ErrorAction SilentlyContinue)
    }

    $hits = @($script:fileIndex | Where-Object { $_.Name -ieq $Name })
    if ($hits.Count -eq 0) {
        # Fall back to a prefix match if not found with suffix
        $stem = [IO.Path]::GetFileNameWithoutExtension($Name)
        $near = @($script:fileIndex | Where-Object { $_.Name -like "$stem*" } | Select-Object -First 5)
        $hint = if ($near) { "`n  near matches:`n    " + (($near | ForEach-Object { $_.FullName }) -join "`n    ") } else { '' }
        throw "$What '$Name' not found under $modelsDir$hint"
    }
    if ($hits.Count -gt 1) {
        Write-Host "  note: $($hits.Count) copies of $Name found, using $($hits[0].FullName)" -ForegroundColor DarkYellow
    }
    return $hits[0].FullName
}

# Replace placeholders in an arg list with actual values, remove unused flags.
function Build-ArgPlaceholders {
    param(
        [string[]]$Arguments,
        [string]$DraftPath,
        [string]$MmprojPath,
        [switch]$EmbeddedMtp
    )

    $draftFlags = @('-md', '--model-draft', '--spec-draft-model',
                    '--spec-draft-n-max',
                    '-ctkd', '--spec-draft-type-k', '--cache-type-k-draft',
                    '-ctvd', '--spec-draft-type-v', '--cache-type-v-draft',
                    '--spec-draft-p-min')
    $out = [Collections.Generic.List[string]]::new()
    $dropValue = $false
    for ($i = 0; $i -lt $Arguments.Count; $i++) {
        $a = $Arguments[$i]

        # Remove external-drafter flag-value pairs if no draft path is found;
        # embedded MTP profiles retain their native draft flags.
        if ($dropValue) { $dropValue = $false; continue }

        # --spec-type may contain both draft-mtp and draftless methods. Remove
        # only draft-dependent methods so ngram-mod remains active on fallback.
        if (-not $DraftPath -and -not $EmbeddedMtp -and $a -eq '--spec-type') {
            if ($i + 1 -lt $Arguments.Count) {
                $types = @($Arguments[$i + 1] -split ',' |
                           Where-Object { $_ -and $_ -notmatch 'draft' })
                if ($types.Count) {
                    $out.Add($a)
                    $out.Add(($types -join ','))
                }
                $i++
            }
            continue
        }
        if (-not $DraftPath -and -not $EmbeddedMtp -and $a -in $draftFlags) { $dropValue = $true; continue }

        $v = $a
        if ($v -match '\{draft\}') { $v = $v -replace '\{draft\}', $DraftPath }
        if ($v -match '\{mmproj\}') {
            if (-not $MmprojPath) {
                # Remove flag-value pair if no path (--mmproj {mmproj})
                if ($v -eq '{mmproj}' -and $out.Count -and $out[-1] -eq '--mmproj') {
                    $out.RemoveAt($out.Count - 1)
                }
                continue
            }
            $v = $v -replace '\{mmproj\}', $MmprojPath
        }
        # Skip flag if no mmproj path provided
        if (-not $MmprojPath -and $v -eq '--no-mmproj-offload') { continue } 
        $out.Add($v)
    }
    return $out.ToArray()
}

# Escape characters in arguments
function Format-Arguments {
    param([Parameter(Mandatory)][AllowEmptyString()][string]$Value)

    if ($Value -ne '' -and $Value -notmatch '[\s"]') { return $Value }

    $sb = [Text.StringBuilder]::new('"')
    for ($i = 0; $i -lt $Value.Length; $i++) {
        $slashes = 0
        while ($i -lt $Value.Length -and $Value[$i] -eq '\') { $slashes++; $i++ }
        if ($i -eq $Value.Length) {
            # Trailing backslashes would otherwise escape the closing quote
            [void]$sb.Append('\' * ($slashes * 2))
        } elseif ($Value[$i] -eq '"') {
            [void]$sb.Append('\' * ($slashes * 2 + 1)).Append('"')
        } else {
            [void]$sb.Append('\' * $slashes).Append($Value[$i])
        }
    }
    return $sb.Append('"').ToString()
}

# Get full command for Command Line
function ConvertTo-CommandLine {
    param([string[]]$Arguments)
    return (($Arguments | ForEach-Object { Format-Arguments -Value $_ }) -join ' ')
}

# ------------------------------------------------------------------ gpu helpers

# Check if flag is present in current version of llama
$script:llamaHelp = $null
function Test-LlamaFlag {
    param([Parameter(Mandatory)][string]$Flag)

    if ($null -eq $script:llamaHelp) {
        try { $script:llamaHelp = (& llama serve --help 2>&1 | Out-String) }
        catch { $script:llamaHelp = '' }
    }
    if (-not $script:llamaHelp) { return $false }
    # Make sure that words are not matched as substrings.
    return $script:llamaHelp -match ('(?<![\w-])' + [regex]::Escape($Flag) + '(?![\w-])')
}

# Get number of GPUs on machine
function Get-GpuCount {
    $smi = Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue
    if (-not $smi) { return 0 }
    $lines = @(& $smi.Source '--list-gpus' 2>$null)
    if ($LASTEXITCODE -ne 0) { return 0 }
    return @($lines | Where-Object { $_ -match '^GPU \d+:' }).Count
}

# Get values for --fit-target flag in MiB
function Get-FitTarget {
    param([double[]]$ReserveGib)

    if ($null -eq $ReserveGib -or $ReserveGib.Count -eq 0) { return $null }
    $count = Get-GpuCount
    if ($count -lt 1) {
        # Unknown GPU count: send what was asked for and let llama default the rest.
        $count = $ReserveGib.Count
    }
    $mib = for ($i = 0; $i -lt $count; $i++) {
        $gib = $ReserveGib[[Math]::Min($i, $ReserveGib.Count - 1)]
        [int][Math]::Round([Math]::Max(0.0, $gib) * 1024)
    }
    return ($mib -join ',')
}

# Format bytes to GiB
function Format-Gib {
    param([double]$Bytes)
    return ('{0:0.0} GiB' -f ($Bytes / 1GB))
}

# Total free VRAM in bytes
function Get-FreeVramBytes {
    $smi = Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue
    if (-not $smi) { return [int64]0 }
    $out = @(& $smi.Source '--query-gpu=memory.free' '--format=csv,noheader,nounits' 2>$null)
    if ($LASTEXITCODE -ne 0) { return [int64]0 }
    [int64]$sum = 0
    foreach ($line in $out) {
        [int]$v = 0
        if ([int]::TryParse("$line".Trim(), [ref]$v)) { $sum += [int64]$v * 1MB }
    }
    return $sum
}

# Total free RAM in bytes
function Get-FreeRamBytes {
    try { return [int64](Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1KB }
    catch { return [int64]0 }
}

# RAM headroom estimation
function Test-HostMemoryHeadroom {
    param([string[]]$Paths, [string]$ReserveMib, [double]$KeepFreeGib)

    if ($KeepFreeGib -le 0) { return }

    [int64]$weights = 0
    foreach ($p in $Paths) {
        if ($p -and (Test-Path -LiteralPath $p)) { $weights += (Get-Item -LiteralPath $p).Length }
    }
    if ($weights -le 0) { return }

    $vram = Get-FreeVramBytes
    if ($vram -le 0) { return }
    [int64]$reserved = 0
    foreach ($m in ("$ReserveMib" -split ',')) {
        [int]$v = 0
        if ([int]::TryParse($m.Trim(), [ref]$v)) { $reserved += [int64]$v * 1MB }
    }

    $onHost = $weights - [Math]::Max([int64]0, $vram - $reserved)
    if ($onHost -le 0) {
        Write-Host ("  {0} of weights vs {1} of usable VRAM, no host spill expected" -f
                    (Format-Gib $weights), (Format-Gib ($vram - $reserved))) -ForegroundColor DarkGray
        return
    }

    $freeRam = Get-FreeRamBytes
    if ($freeRam -le 0) { return }
    $keep = [int64]($KeepFreeGib * 1GB)
    if ($freeRam - $onHost -lt $keep) {
        Write-Warning ("about {0} of weights will not fit in VRAM and spill into system RAM, which has {1} is free.
                        Remaining RAM would be less than {2}." -f
                       (Format-Gib $onHost), (Format-Gib $freeRam), (Format-Gib $keep))
    } else {
        Write-Host ("  about {0} of weights spill into system RAM, {1} free" -f
                    (Format-Gib $onHost), (Format-Gib $freeRam)) -ForegroundColor DarkGray
    }
}

# Shows the number of kv slots and their context, and total context
function Show-ServerContext {
    param([string[]]$Arguments)

    # The value after a flag, if there is one.
    function Get-ArgValue {
        param([string[]]$From, [string[]]$Flags)
        for ($i = 0; $i -lt $From.Count - 1; $i++) {
            if ($From[$i] -in $Flags) { return $From[$i + 1] }
        }
        return $null
    }

    $asked = Get-ArgValue -From $Arguments -Flags @('--ctx-size', '-c')
    $parallel = Get-ArgValue -From $Arguments -Flags @('--parallel', '-np')
    $auto = -not ($parallel -match '^[1-9]\d*$')
    $slots = if ($auto) { 4 } else { [int]$parallel }
    $unified = $auto -or ($Arguments -contains '--kv-unified') -or ($Arguments -contains '-kvu')
    if ($Arguments -contains '--no-kv-unified' -or $Arguments -contains '-no-kvu') { $unified = $false }

    try { $props = Invoke-RestMethod "http://127.0.0.1:$port/props" -TimeoutSec 5 } catch { return }
    $perSlot = $props.default_generation_settings.n_ctx
    if ($perSlot -isnot [int] -and $perSlot -isnot [long]) { return }

    $total = if ($unified) { [int]$perSlot } else { [int]$perSlot * $slots }
    $detail = if ($slots -le 1) { '' }
              elseif ($unified) { " (shared by $slots slots)" }
              else { " ($perSlot per slot x $slots)" }
    if ($asked -match '^[1-9]\d*$' -and [int]$asked -ne $total) {
        Write-Host ("  context: {0} tokens{1} - the config asked for {2}" -f
                    $total, $detail, $asked) -ForegroundColor Yellow
    } else {
        Write-Host "  context: $total tokens$detail" -ForegroundColor DarkGray
    }
}



# HERE


function Get-ServerContextWindow {
    try {
        $props = Invoke-RestMethod "http://127.0.0.1:$port/props" -TimeoutSec 5
        $perSlot = $props.default_generation_settings.n_ctx
        if ($perSlot -isnot [int] -and $perSlot -isnot [long]) { return $null }
        return [int]$perSlot
    } catch { return $null }
}

function Add-OrSetProperty {
    param(
        [Parameter(Mandatory)]$Object,
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)]$Value
    )

    if ($Object.PSObject.Properties[$Name]) {
        $Object.$Name = $Value
    } else {
        $Object | Add-Member -NotePropertyName $Name -NotePropertyValue $Value
    }
}

function ConvertTo-TwoSpaceJson {
    param([Parameter(Mandatory)][string]$Json)

    # Windows PowerShell 5.1's ConvertTo-Json has no -Indent option and its
    # default whitespace is alignment padding rather than simple nesting.
    $compact = ($Json | ConvertFrom-Json | ConvertTo-Json -Depth 20 -Compress)
    $out = [Text.StringBuilder]::new()
    $indent = 0
    $inString = $false
    $escaped = $false
    $previous = [char]0

    for ($i = 0; $i -lt $compact.Length; $i++) {
        $ch = $compact[$i]

        if ($inString) {
            [void]$out.Append($ch)
            if ($escaped) {
                $escaped = $false
            } elseif ($ch -eq '\') {
                $escaped = $true
            } elseif ($ch -eq '"') {
                $inString = $false
                $previous = $ch
            }
            continue
        }

        if ([char]::IsWhiteSpace($ch)) { continue }

        if ($ch -eq '"') {
            [void]$out.Append($ch)
            $inString = $true
            continue
        }

        if ($ch -eq '{' -or $ch -eq '[') {
            [void]$out.Append($ch)
            $indent++
            $next = if ($i + 1 -lt $compact.Length) { $compact[$i + 1] } else { [char]0 }
            if (($ch -eq '{' -and $next -ne '}') -or ($ch -eq '[' -and $next -ne ']')) {
                [void]$out.Append([Environment]::NewLine)
                [void]$out.Append((('  ' * $indent) -join ''))
            }
        } elseif ($ch -eq '}' -or $ch -eq ']') {
            $indent = [Math]::Max(0, $indent - 1)
            if ($previous -ne '{' -and $previous -ne '[') {
                [void]$out.Append([Environment]::NewLine)
                [void]$out.Append((('  ' * $indent) -join ''))
            }
            [void]$out.Append($ch)
        } elseif ($ch -eq ',') {
            [void]$out.Append($ch)
            [void]$out.Append([Environment]::NewLine)
            [void]$out.Append((('  ' * $indent) -join ''))
        } elseif ($ch -eq ':') {
            [void]$out.Append(': ')
        } else {
            [void]$out.Append($ch)
        }

        $previous = $ch
    }

    return $out.ToString()
}

function Update-PiModelConfig {
    param(
        [Parameter(Mandatory)][string]$ModelId,
        [Parameter(Mandatory)][string]$Description,
        [Parameter(Mandatory)][bool]$SupportsImages,
        [int]$ContextWindow = 131072
    )

    $path = Join-Path $env:USERPROFILE '.pi\agent\models.json'
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "pi model configuration not found: $path"
    }

    try {
        $root = Get-Content -LiteralPath $path -Raw -ErrorAction Stop | ConvertFrom-Json
    } catch {
        throw "could not parse pi model configuration '$path': $($_.Exception.Message)"
    }

    $providers = $root.providers
    $llamacpp = if ($providers) { $providers.llamacpp } else { $null }
    if (-not $llamacpp) {
        throw "pi model configuration has no providers.llamacpp object: $path"
    }

    $inputTypes = [System.Collections.Generic.List[string]]::new()
    [void]$inputTypes.Add('text')
    if ($SupportsImages) { [void]$inputTypes.Add('image') }

    $existing = @($llamacpp.models | Where-Object { $_.id -eq $ModelId } | Select-Object -First 1)
    if ($existing.Count) {
        $model = $existing[0]
    } else {
        $model = [pscustomobject][ordered]@{
            id = $ModelId
            name = $Description
            input = $inputTypes
            contextWindow = $ContextWindow
            maxTokens = 131072
            reasoning = $true
            cost = [ordered]@{ input = 0; output = 0; cacheRead = 0; cacheWrite = 0 }
        }
    }

    Add-OrSetProperty -Object $model -Name 'id' -Value $ModelId
    if (-not $model.PSObject.Properties['name']) {
        Add-OrSetProperty -Object $model -Name 'name' -Value $Description
    }
    # Keep this as a JSON array even when it contains only "text". Existing
    # models may have been written by an older version with a scalar value.
    Add-OrSetProperty -Object $model -Name 'input' -Value $inputTypes
    if ($ContextWindow -gt 0) {
        Add-OrSetProperty -Object $model -Name 'contextWindow' -Value $ContextWindow
    }

    # The selected server is the only local model available on this port.
    $llamacpp.models = @($model)

    $json = ConvertTo-TwoSpaceJson -Json ($root | ConvertTo-Json -Depth 20)
    $tempPath = "$path.$PID.tmp"
    try {
        [IO.File]::WriteAllText($tempPath, $json, [Text.UTF8Encoding]::new($false))
        Move-Item -LiteralPath $tempPath -Destination $path -Force
    } catch {
        throw "could not update pi model configuration '$path': $($_.Exception.Message)"
    } finally {
        if (Test-Path -LiteralPath $tempPath) {
            Remove-Item -LiteralPath $tempPath -Force -ErrorAction SilentlyContinue
        }
    }

    Write-Host "Updated pi model configuration: llamacpp/$ModelId" -ForegroundColor DarkGray
}

function Invoke-SearxngCompose {
    param([Parameter(Mandatory)][ValidateSet('up', 'down')][string]$Action)

    $docker = Get-Command 'docker' -ErrorAction SilentlyContinue
    if (-not $docker) { throw 'docker command not found' }
    $searxngDir = Join-Path $env:USERPROFILE 'dev\searxng'
    if (-not (Test-Path -LiteralPath (Join-Path $searxngDir 'docker-compose.yml') -PathType Leaf)) {
        throw "SearXNG compose file not found: $searxngDir"
    }

    $composeArgs = if ($Action -eq 'up') { @('compose', 'up', '-d') } else { @('compose', 'down') }
    Push-Location -LiteralPath $searxngDir
    try {
        $output = @(& $docker.Source @composeArgs 2>&1)
        $exitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    if ($exitCode -ne 0) {
        $detail = if ($output) { ": $($output -join "`n")" } else { '' }
        throw "docker compose $Action failed$detail"
    }
}

function Start-Searxng {
    Write-Host 'Starting SearXNG web search...' -ForegroundColor Cyan
    try {
        Invoke-SearxngCompose -Action up
        Write-Host 'SearXNG web search is running.' -ForegroundColor DarkGray
        return $true
    } catch {
        Write-Warning "SearXNG could not be started: $($_.Exception.Message)"
        try { Invoke-SearxngCompose -Action down } catch { }
        return $false
    }
}

function Stop-Searxng {
    Write-Host 'Stopping SearXNG web search.' -ForegroundColor Cyan
    try {
        Invoke-SearxngCompose -Action down
    } catch {
        Write-Warning "SearXNG could not be stopped: $($_.Exception.Message)"
    }
}

# Validate every menu entry up front. Missing experimental models are retained
# in the menu but cannot be selected; valid entries remain usable.
$configStatus = [ordered]@{}
foreach ($name in $cfg.Keys) {
    $entry = $cfg[$name]
    $errors = @()
    $warnings = @()
    $paths = @{}

    foreach ($required in @('Desc', 'File', 'Args')) {
        if (-not $entry.Contains($required) -or $null -eq $entry[$required] -or
            ($entry[$required] -is [string] -and [string]::IsNullOrWhiteSpace($entry[$required]))) {
            $errors += "missing $required"
        }
    }

    if (-not $errors.Count) {
        $templateFlag = [Array]::IndexOf([object[]]@($entry.Args), '--chat-template-file')
        if ($templateFlag -ge 0) {
            if ($templateFlag + 1 -ge @($entry.Args).Count) {
                $errors += '--chat-template-file has no path'
            } else {
                $templatePath = @($entry.Args)[$templateFlag + 1]
                if (-not (Test-Path -LiteralPath $templatePath -PathType Leaf)) {
                    $errors += "chat template not found: $templatePath"
                }
            }
        }
        foreach ($spec in @(
            @{ Property = 'File';   Label = 'model file' },
            @{ Property = 'Draft';  Label = 'draft file' },
            @{ Property = 'Mmproj'; Label = 'mmproj file' }
        )) {
            if (-not ($entry.Contains($spec.Property) -and $entry[$spec.Property])) { continue }
            try {
                $paths[$spec.Property] = Resolve-ModelFile -Name $entry[$spec.Property] -What $spec.Label
            } catch {
                if ($spec.Property -eq 'File') {
                    $errors += $_.Exception.Message
                } else {
                    # MTP and mmproj files are optional; argument expansion
                    # removes their flag/placeholder pairs when unresolved.
                    $warnings += $_.Exception.Message
                }
            }
        }
    }

    $configStatus[$name] = @{
        Available = ($errors.Count -eq 0)
        Errors = $errors
        Warnings = $warnings
        Paths = $paths
    }
}

# ---------------------------------------------------------------------- menu

function Select-Model {
    $names = @($cfg.Keys)
    # Pad names to the longest one so the description column stays aligned
    $longestNameWidth = (@($names | ForEach-Object { $_.Length }) | Measure-Object -Maximum).Maximum + 1
    $selectable = @($names | Where-Object { $configStatus[$_].Available })
    if (-not $selectable.Count) { throw 'no valid model configurations are available' }
    $default = $selectable[0]

    # Arrow keys need a real console; fall back to the old prompt otherwise
    if ([Console]::IsInputRedirected) {
        Write-Host ''
        for ($i = 0; $i -lt $names.Count; $i++) {
            $suffix = if ($configStatus[$names[$i]].Available) { '' } else { ' [unavailable]' }
            Write-Host ("  {0}. {1} {2}{3}" -f ($i + 1), $names[$i].PadRight($longestNameWidth), $cfg[$names[$i]].Desc, $suffix)
        }
        Write-Host '  q. quit'
        while ($true) {
            $sel = Read-Host "model (number/name, Enter = $default)"
            if ($null -eq $sel -or $sel -eq '') { return $default }
            $sel = $sel.Trim().ToLower()
            if ($sel -eq 'q') { exit 0 }
            if ($cfg.Contains($sel) -and $configStatus[$sel].Available) { return $sel }
            $n = 0
            if ([int]::TryParse($sel, [ref]$n) -and $n -ge 1 -and $n -le $names.Count) {
                $candidate = $names[$n - 1]
                if ($configStatus[$candidate].Available) { return $candidate }
            }
            Write-Host 'enter a number, a model name, or q'
        }
    }

    $sel = 0
    $lineCount = $names.Count + 2   # Model rows + blank line + hint line
    Write-Host ''
    # Reserve the menu area once so redraws can overwrite it in place
    for ($i = 0; $i -lt $lineCount; $i++) { Write-Host '' }
    [Console]::CursorVisible = $false
    try {
        while ($true) {
            $top = [Console]::CursorTop - $lineCount
            [Console]::SetCursorPosition(0, $top)
            for ($i = 0; $i -lt $names.Count; $i++) {
                $marker = if ($i -eq $sel) { '>' } else { ' ' }
                $suffix = if ($configStatus[$names[$i]].Available) { '' } else { ' [unavailable]' }
                $line = ("  {0} {1}. {2} {3}{4}" -f $marker, ($i + 1), $names[$i].PadRight($longestNameWidth), $cfg[$names[$i]].Desc, $suffix)
                if ($i -eq $sel) {
                    Write-Host $line -ForegroundColor Black -BackgroundColor Cyan
                } else {
                    Write-Host $line
                }
            }
            Write-Host ''
            Write-Host '  up/down move, Enter select, 1-9 jump, q/Esc quit' -ForegroundColor DarkGray

            $key = [Console]::ReadKey($true)
            switch ($key.Key) {
                'UpArrow'   { $sel = ($sel - 1 + $names.Count) % $names.Count }
                'DownArrow' { $sel = ($sel + 1) % $names.Count }
                'Enter'     { if ($configStatus[$names[$sel]].Available) { return $names[$sel] } }
                'Escape'    { exit 0 }
                'Q'         { exit 0 }
                default {
                    $n = 0
                    if ([int]::TryParse($key.KeyChar, [ref]$n) -and $n -ge 1 -and $n -le $names.Count) {
                        $candidate = $names[$n - 1]
                        if ($configStatus[$candidate].Available) { return $candidate }
                    }
                }
            }
        }
    } finally {
        [Console]::CursorVisible = $true
    }
}

# ------------------------------------------------------------------ server I/O

function Get-LoadedModel {
    try {
        $r = Invoke-RestMethod "http://127.0.0.1:$port/v1/models" -TimeoutSec 2
        return $r.data[0].id
    } catch { return $null }
}

# ------------------------------------------------------------- model + config

$alreadyRunningServer = $false
$loadedBeforeSelection = Get-LoadedModel
$portOwnerBeforeSelection = (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
                             Select-Object -First 1).OwningProcess

if ($loadedBeforeSelection -and $portOwnerBeforeSelection) {
    $runningModelName = @($cfg.Keys | Where-Object {
        $id = $_
        $loadedBeforeSelection -eq $id -or $loadedBeforeSelection -like "*$id*"
    } | Select-Object -First 1)

    if (-not $runningModelName -or -not $configStatus[$runningModelName[0]].Available) {
        throw "a llama server is already running on port $port with model '$loadedBeforeSelection', but that model is not one of the available pillama configurations"
    }

    $Model = $runningModelName[0]
    $alreadyRunningServer = $true
} else {
    $Model = Select-Model
}

if ($alreadyRunningServer) {
    Write-Host "Server left running on port $port (stop it with: Get-Process llama | Stop-Process)"
    Write-Host "Connecting to server on port $port..." -ForegroundColor Cyan
} else {
    if (-not [Console]::IsOutputRedirected) {
        Clear-Host
    }
    Write-Host "Selected model: $Model" -ForegroundColor Green
    Write-Host 'Starting llama server...' -ForegroundColor Cyan
}
if (-not $configStatus[$Model].Available) {
    throw "model '$Model' is unavailable:`n  $($configStatus[$Model].Errors -join "`n  ")"
}
foreach ($warning in $configStatus[$Model].Warnings) {
    Write-Warning "$warning; starting without that optional feature"
}
$c    = $cfg[$Model]
$modelId = $Model
$gguf = $configStatus[$Model].Paths.File
$draft = $configStatus[$Model].Paths.Draft
$mmproj = $configStatus[$Model].Paths.Mmproj

$embeddedMtp = if ($c.Contains('EmbeddedMtp')) { [bool]$c.EmbeddedMtp } else { $false }
$modelArgs = Build-ArgPlaceholders -Arguments $c.Args -DraftPath $draft -MmprojPath $mmproj -EmbeddedMtp:$embeddedMtp

# ------------------------------------------------------------------ server I/O

function Wait-ServerReady {
    param($Proc, [int]$TimeoutSec, [string]$Label)

    $start = Get-Date
    $deadline = $start.AddSeconds($TimeoutSec)
    $ready = $false
    $detailHint = if ($Silent) { '; retry without -silent for details' } else { '; see the llama console for details' }

    # The countdown repaints one line in place, which needs a real console: a
    # redirected host has no cursor to rewind, so it would scroll a line per second.
    $live = -not ([Console]::IsOutputRedirected -or [Console]::IsInputRedirected)
    if ($Silent -and -not $live) {
        Write-Host 'Waiting for server health...'
    } else {
        Write-Host 'Waiting for server health; detailed startup output is in the llama window.'
    }

    $width = 0
    $painted = ''
    $nextCheck = $start
    try {
        while ($true) {
            $now = Get-Date
            if ($now -ge $deadline) { break }
            if ($Proc.HasExited) {
                throw "$llamaExe $serveCmd exited during startup (exit code $($Proc.ExitCode))$detailHint"
            }
            if ($live) {
                $el = $now - $start
                # Derive both displays from the same floored elapsed-second count.
                # Flooring elapsed and remaining TimeSpans independently makes the
                # countdown appear one second short (e.g. 0:10 + 9:49 for 600 sec).
                $elapsedSec = [int][Math]::Floor($el.TotalSeconds)
                $remainingSec = [Math]::Max(0, $TimeoutSec - $elapsedSec)
                $line = (" loading {0} {1}:{2:d2} elapsed {3}:{4:d2} until timeout" -f
                $Label, [int][Math]::Floor($elapsedSec / 60), ($elapsedSec % 60),
                [int][Math]::Floor($remainingSec / 60), ($remainingSec % 60))
                # Only the seconds change, so repaint on change instead of every poll.
                if ($line -ne $painted) {
                    Write-Host ("`r" + $line.PadRight($width)) -NoNewline -ForegroundColor DarkGray
                    $width = $line.Length
                    $painted = $line
                }
            }
            # Polling is deliberately slower than the repaint: the timer stays smooth
            # while the health endpoint is still asked roughly every two seconds.
            if ($now -ge $nextCheck) {
                $nextCheck = $now.AddSeconds(2)
                try {
                    $h = Invoke-RestMethod "127.0.0.1:$port/health" -TimeoutSec 2
                    if ($h.status -eq 'ok') { $ready = $true; break }
                } catch { }
            }
            Start-Sleep -Milliseconds 250
        }
    } finally {
        # Clear the countdown so the outcome line (or an exception) starts clean.
        if ($width) { Write-Host ("`r" + (' ' * $width) + "`r") -NoNewline }
    }

    $el = (Get-Date) - $start
    if (-not $ready) {
        throw "server did not become healthy within $TimeoutSec seconds$detailHint"
    }
    Write-Host ("  {0} ready in {1}:{2:d2}" -f $Label, [int]$el.TotalMinutes, $el.Seconds) -ForegroundColor Green
}

function Show-StartupMemorySummary {
    param(
        [string]$LogPath,
        [string]$FitTarget,
        [double]$KeepFreeGib
    )

    $layers = $null
    if ($LogPath -and (Test-Path -LiteralPath $LogPath -PathType Leaf)) {
        try { $log = Get-Content -LiteralPath $LogPath -Raw -ErrorAction Stop } catch { $log = '' }
        if ($log) {
            $layers = [regex]::Matches($log, '(?im)^\s*[^:\r\n]*load_tensors:\s+offloaded\s+(\d+)\s*/\s*(\d+)\s+layers\s+to\s+GPU') |
                      Select-Object -First 1
        }
    }

    Write-Host 'Startup memory summary:' -ForegroundColor Cyan
    if ($layers) {
        Write-Host ("  GPU layers: {0}/{1} offloaded" -f $layers.Groups[1].Value, $layers.Groups[2].Value)
    }
    if ($FitTarget) {
        $targets = @($FitTarget -split ',')
        for ($i = 0; $i -lt $targets.Count; $i++) {
            Write-Host ("  GPU {0} VRAM reserved for headroom: {1} MiB ({2:0.00} GiB)" -f
                        $i, $targets[$i], ([double]$targets[$i] / 1024))
        }
    }
    if ($KeepFreeGib -gt 0) {
        Write-Host ("  System RAM reserved for headroom: {0:0.00} GiB" -f $KeepFreeGib)
    }
}

# ------------------------------------------------------------------ start/reuse

$proc = $null
$startedByUs = $false
$serverOwnerPid = $null
$startupLogPath = $null
$searxngStartedByUs = $false
$loaded = Get-LoadedModel
$portOwner = (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
              Select-Object -First 1).OwningProcess

if ($loaded -and ($loaded -eq $modelId -or $loaded -like "*$modelId*")) {
    if (-not $alreadyRunningServer) {
        Write-Host "Reusing already-running server ($loaded on port $port)" -ForegroundColor Green
    }
    $searxngStartedByUs = Start-Searxng
} else {
    if ($portOwner) {
        $what = if ($loaded) { "a different model ($loaded)" } else { 'an unknown process' }
        $ownerProcess = Get-Process -Id $portOwner -ErrorAction SilentlyContinue
        $ownerName = if ($ownerProcess) { $ownerProcess.ProcessName } else { 'unknown' }
        if (-not $forceReplacePortOwner) {
            throw "port $port is already owned by $ownerName (pid $portOwner), serving $what. Stop it manually or set `$forceReplacePortOwner = `$true."
        }
        Write-Host "Port $port is serving $what - stopping $ownerName (pid $portOwner)" -ForegroundColor Yellow
        Stop-Process -Id $portOwner -Force
        Start-Sleep -Seconds 2
        $remainingOwner = (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
                           Select-Object -First 1).OwningProcess
        if ($remainingOwner) { throw "port $port is still held by pid $remainingOwner" }
    }

    $serveArgs = @('serve', '-m', $gguf, '--alias', $modelId) + $common + $modelArgs

    # Warn rather than strip: a config may have a reason to override the fitter, but it
    # should be a deliberate one, because the fallback placement has no memory margin.
    $fitBreakers = @($serveArgs | Where-Object {
        $_ -in @('-ngl', '--gpu-layers', '--n-gpu-layers', '-ts', '--tensor-split',
                 '-ncmoe', '--n-cpu-moe', '--cpu-moe', '-cmoe') })
    $fitIndex = [Array]::IndexOf([object[]]$serveArgs, '--fit')
    $fitDisabled = ($fitIndex -ge 0 -and $fitIndex + 1 -lt $serveArgs.Count -and
                    $serveArgs[$fitIndex + 1] -eq 'off')
    if ($fitBreakers.Count -and -not $fitDisabled) {
        Write-Warning ("{0} in this config aborts llama's memory fitter; placement falls back to a free-VRAM ratio with no margin." -f
                       (($fitBreakers | Select-Object -Unique) -join ', '))
    }
    if ($fitDisabled) {
        Write-Host '  llama memory fitting disabled; this profile requires all requested layers to fit on GPU' -ForegroundColor DarkGray
    }

    $fitTarget = Get-FitTarget -ReserveGib $vramReserve
    $fitTargetApplied = $null
    if ($fitTarget -and -not $fitDisabled) {
        if (Test-LlamaFlag '--fit-target') {
            $serveArgs += @('--fit-target', $fitTarget)
            $fitTargetApplied = $fitTarget
            Write-Host "  reserving $fitTarget MiB of VRAM, per device" -ForegroundColor DarkGray
        } else {
            Write-Warning 'this llama build has no --fit-target option; -reserve ignored'
        }
    }

    Test-HostMemoryHeadroom -Paths @($gguf, $draft, $mmproj) -ReserveMib $fitTarget -KeepFreeGib $RamThreshold

    if (Test-LlamaFlag '--log-file') {
        $startupLogPath = (New-TemporaryFile).FullName
        $serveArgs += @('--log-file', $startupLogPath)
    }

    $env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'

    $commandLine = ConvertTo-CommandLine -Arguments $serveArgs
    Write-Host "Starting: llama $commandLine" -ForegroundColor Cyan
    if ($Silent) {
        $proc = Start-Process 'llama' -ArgumentList $commandLine -WindowStyle Hidden -PassThru
    } else {
        $proc = Start-Process 'llama' -ArgumentList $commandLine -PassThru
    }
    $startedByUs = $true
    $searxngStartedByUs = Start-Searxng

    try {
        Wait-ServerReady -Proc $proc -TimeoutSec $startupTimeoutSec -Label $modelId
        $serverOwnerPid = (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
                           Select-Object -First 1).OwningProcess
        Show-StartupMemorySummary -LogPath $startupLogPath -FitTarget $fitTargetApplied -KeepFreeGib $RamThreshold
    } catch {
        if ($searxngStartedByUs) {
            Stop-Searxng
            $searxngStartedByUs = $false
        }
        throw
    }
}

try {
    Show-ServerContext -Arguments $modelArgs
    $contextWindow = Get-ServerContextWindow
    if (-not $contextWindow) {
        $ctxIndex = [Array]::IndexOf([object[]]@($modelArgs), '--ctx-size')
        if ($ctxIndex -ge 0 -and $ctxIndex + 1 -lt @($modelArgs).Count) {
            [int]::TryParse($modelArgs[$ctxIndex + 1], [ref]$contextWindow) | Out-Null
        }
    }
    if (-not $contextWindow) { $contextWindow = 131072 }
    Update-PiModelConfig -ModelId $modelId -Description $c.Desc `
                         -SupportsImages ([bool]$mmproj) -ContextWindow $contextWindow

    # provider/model comes from ~\.pi\agent\models.json
    pi --model "llamacpp/$modelId"
} finally {
    if ($startedByUs -and $Silent) {
        if ($proc -and -not $proc.HasExited) {
            Write-Host 'Stopping llama server.' -ForegroundColor Cyan
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
        Start-Sleep -Milliseconds 500
        if ($serverOwnerPid -and $serverOwnerPid -ne $proc.Id) {
            $stillUp = (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
                        Where-Object OwningProcess -eq $serverOwnerPid | Select-Object -First 1)
            if ($stillUp) {
                Write-Host "  stopping recorded server pid $serverOwnerPid" -ForegroundColor DarkYellow
                Stop-Process -Id $serverOwnerPid -Force -ErrorAction SilentlyContinue
            }
        }
    } elseif (-not $alreadyRunningServer) {
        Write-Host "Server left running on port $port (stop it with: Get-Process llama | Stop-Process)"
    }
    if ($searxngStartedByUs) {
        Stop-Searxng
        $searxngStartedByUs = $false
    }
    if ($startupLogPath -and (Test-Path -LiteralPath $startupLogPath)) {
        Remove-Item -LiteralPath $startupLogPath -Force -ErrorAction SilentlyContinue
    }
}

<#
Orphaned server? If the terminal window was closed outright (or the shell was
killed), the finally block above never ran, so llama is still holding VRAM.
Nothing can clean that up automatically - kill it by hand from a new shell:

  # precise: stop whatever is listening on the port ($port above)
  Get-NetTCPConnection -LocalPort 8080 -State Listen |
      ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }

  # blunt: stop every llama process
  Get-Process llama | Stop-Process -Force

  # check first, if you want to see what you are about to kill
  Get-NetTCPConnection -LocalPort 8080 -State Listen |
      ForEach-Object { Get-Process -Id $_.OwningProcess }

Re-running pillama also clears it: a server on the port that is not the model
you asked for gets stopped during startup. If it *is* the same model, the
orphan is reused as-is and nothing needs killing.

To exit cleanly instead, quit pi normally (or Ctrl+C) rather than closing the
window - then the finally block stops the server for you. After -server the
server is left up on purpose, so it is always yours to stop.
#>
