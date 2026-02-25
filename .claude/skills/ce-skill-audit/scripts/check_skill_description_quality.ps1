param(
    [string]$Root = ".claude/skills",
    [int]$MinLength = 70,
    [int]$MaxLength = 160,
    [int]$MaxCommas = 4
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-SkillDescription {
    param([string]$Path)

    $lines = Get-Content $Path
    $descIdx = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match '^description:\s*>\s*$') {
            $descIdx = $i
            break
        }
    }
    if ($descIdx -lt 0) {
        return $null
    }

    $buf = @()
    for ($j = $descIdx + 1; $j -lt $lines.Count; $j++) {
        if ($lines[$j] -eq '---') { break }
        if ($lines[$j] -match '^\s+') {
            $buf += $lines[$j].Trim()
        }
    }
    return ($buf -join " ").Trim()
}

$violations = @()

Get-ChildItem $Root -Directory | Sort-Object Name | ForEach-Object {
    $skill = $_.Name
    $skillPath = Join-Path $_.FullName "SKILL.md"
    if (-not (Test-Path $skillPath)) {
        $violations += [PSCustomObject]@{
            Skill = $skill
            Rule = "missing_skill_file"
            Detail = "SKILL.md not found"
        }
        return
    }

    $desc = Get-SkillDescription -Path $skillPath
    if ([string]::IsNullOrWhiteSpace($desc)) {
        $violations += [PSCustomObject]@{
            Skill = $skill
            Rule = "missing_description"
            Detail = "description block not found or empty"
        }
        return
    }

    $len = $desc.Length
    if ($len -lt $MinLength -or $len -gt $MaxLength) {
        $violations += [PSCustomObject]@{
            Skill = $skill
            Rule = "length_out_of_bounds"
            Detail = "length=$len (expected $MinLength..$MaxLength)"
        }
    }

    if ($desc -notmatch '^[A-Z]') {
        $violations += [PSCustomObject]@{
            Skill = $skill
            Rule = "must_start_with_uppercase"
            Detail = $desc
        }
    }

    if ($desc -notmatch '\.$') {
        $violations += [PSCustomObject]@{
            Skill = $skill
            Rule = "must_end_with_period"
            Detail = $desc
        }
    }

    $sentenceStops = ([regex]::Matches($desc, '[.!?]')).Count
    if ($sentenceStops -gt 1) {
        $violations += [PSCustomObject]@{
            Skill = $skill
            Rule = "too_many_sentences"
            Detail = "sentence_stops=$sentenceStops"
        }
    }

    $commaCount = ([regex]::Matches($desc, ',')).Count
    if ($commaCount -gt $MaxCommas) {
        $violations += [PSCustomObject]@{
            Skill = $skill
            Rule = "too_many_commas"
            Detail = "comma_count=$commaCount (max=$MaxCommas)"
        }
    }

    if ($desc -match "'[^']+'") {
        $violations += [PSCustomObject]@{
            Skill = $skill
            Rule = "quoted_phrase_dump"
            Detail = "Contains quoted trigger phrases; move to references/trigger_phrases.md"
        }
    }
}

if ($violations.Count -gt 0) {
    Write-Host "Skill description quality check failed:" -ForegroundColor Red
    $violations | Format-Table -AutoSize
    exit 1
}

Write-Host "Skill description quality check passed for $Root." -ForegroundColor Green
exit 0
