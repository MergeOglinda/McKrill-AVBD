<#
.SYNOPSIS
    Runs the performance tests of two git refs several times each, alternating between them, and compares the results.

.DESCRIPTION
    For each ref the script keeps a git worktree (a separate Unity project folder with its own Library) under -WorkRoot, so both sides are built from
    their branch and neither touches the working copy you are editing. It then:

      1. resolves both refs to commits and checks them out in the worktrees (the worktrees are scratch folders: local changes in them are discarded);
      2. opens each project once in batch mode to import it and compile the shaders (no tests);
      3. runs the performance tests -Runs times per side, alternating the order every round (A B, B A, A B, ...) so that a drift in GPU
         temperature or clocks does not favour one side. Every run is one Unity process and writes its own NUnit result file and log;
      4. calls multi_compare_perf.py on the two folders of result files.

    Everything goes to <OutRoot>\<timestamp>\ : baseline\runNN.xml and .log, candidate\runNN.xml and .log, meta.json, comparison.csv.

    The tests need a GPU, so the script never passes -nographics. Close other Unity editors that have one of the worktrees open, and keep the machine
    otherwise idle while it runs.

.PARAMETER Baseline
    Git ref of the baseline, for example origin/main. A branch name without a remote means the local branch, which can be older than the remote one.

.PARAMETER Candidate
    Git ref of the candidate, for example origin/primal-lds-accumulate.

.PARAMETER Runs
    Result files per side (default 8). Each is a run of the whole suite, which is what multi_compare_perf.py expects.

.PARAMETER RepoPath
    The repository to take the refs from. Default: the repository this script is in.

.PARAMETER WorkRoot
    Where the two worktrees and the results go. Default: a folder "mckrill-avbd-perf" next to the repository. Keep it between invocations: the
    Library folders in it make the next invocation start quickly.

.PARAMETER OutRoot
    Where the result folders go. Default: <WorkRoot>\results.

.PARAMETER UnityPath
    Unity.exe to use for both sides. Default: the editor in the Unity Hub folder that matches the project's version. The two refs must use the same
    editor version (the script stops if they do not, unless this is given).

.PARAMETER TestFilter
    Optional Unity -testFilter, for example "Scenario1_Shots2m_PlasticYieldOff;Scenario2_Shots2m_PlasticYieldOn" (names separated by semicolons).

.PARAMETER ExpectedTests
    How many test cases a result file should have (default 5). A run with another count gets a warning. Not checked when -TestFilter is given.

.PARAMETER RunTimeoutMinutes
    A Unity process that runs longer is killed and its run counts as failed (default 240).

.PARAMETER Warmup
    Adds one run of the suite per side before the measured runs, whose results are discarded.

.PARAMETER NoFetch
    Do not run "git fetch" first.

.PARAMETER Threshold
    Percent change of the median that multi_compare_perf.py reports as better or worse (default 5).

.EXAMPLE
    .\Tools\PerfCompare\run_perf_ab.ps1 -Baseline origin/main -Candidate origin/primal-lds-accumulate

.EXAMPLE
    .\Tools\PerfCompare\run_perf_ab.ps1 -Baseline origin/gpu-trace-dual-primal -Candidate origin/primal-lds-accumulate -Runs 4 -Warmup `
        -TestFilter "Scenario5_BrickRing110k"
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Baseline,
    [Parameter(Mandatory = $true)][string]$Candidate,
    [int]$Runs = 8,
    [string]$RepoPath,
    [string]$WorkRoot,
    [string]$OutRoot,
    [string]$UnityPath,
    [string]$TestFilter,
    [string]$TestCategory = 'Performance',
    [string]$AssemblyName = 'McKrill.Avbd.Tests.Performance',
    [int]$ExpectedTests = 5,
    [int]$RunTimeoutMinutes = 240,
    [double]$Threshold = 5,
    [string]$Python = 'python',
    [switch]$Warmup,
    [switch]$NoFetch
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2

if ($Runs -lt 1) { throw '-Runs must be at least 1.' }

# ---------------------------------------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------------------------------------

function Write-Step([string]$text) {
    Write-Host ('[{0}] {1}' -f (Get-Date -Format 'HH:mm:ss'), $text)
}

# Runs git in a folder and returns its standard output. Standard error is left on the console (git writes its progress there).
function Invoke-Git([string]$dir, [string[]]$gitArgs) {
    $out = & git -C $dir @gitArgs
    if ($LASTEXITCODE -ne 0) { throw ('git {0} failed in {1}' -f ($gitArgs -join ' '), $dir) }
    return $out
}

function Resolve-Commit([string]$ref) {
    $sha = & git -C $RepoPath rev-parse --verify --quiet "${ref}^{commit}"
    if ($LASTEXITCODE -ne 0 -or -not $sha) {
        throw ("Cannot resolve '{0}' to a commit in {1}. Use origin/<branch> for a branch of the remote, and do not use -NoFetch for a new one." -f $ref, $RepoPath)
    }
    return ([string]$sha).Trim()
}

# Creates the worktree on first use, then moves it to the commit. The worktree is detached, so the branch itself stays free in the main repository.
function Initialize-Worktree([string]$path, [string]$sha) {
    if (Test-Path -LiteralPath (Join-Path $path '.git')) {
        Invoke-Git $path @('checkout', '--force', '--detach', $sha) | Out-Null
    }
    else {
        if (Test-Path -LiteralPath $path) { throw ("'{0}' exists but is not a git worktree. Remove it or choose another -WorkRoot." -f $path) }
        Invoke-Git $RepoPath @('worktree', 'add', '--detach', $path, $sha) | Out-Null
    }
}

function Get-UnityVersion([string]$project) {
    $file = Join-Path $project 'ProjectSettings\ProjectVersion.txt'
    $match = Select-String -LiteralPath $file -Pattern '^m_EditorVersion:\s*(\S+)' | Select-Object -First 1
    if (-not $match) { throw ('No m_EditorVersion in {0}' -f $file) }
    return $match.Matches[0].Groups[1].Value
}

function Get-UnityExe([string]$project) {
    if ($UnityPath) {
        if (-not (Test-Path -LiteralPath $UnityPath)) { throw ('-UnityPath not found: {0}' -f $UnityPath) }
        return $UnityPath
    }
    $version = Get-UnityVersion $project
    $exe = Join-Path $env:ProgramFiles ('Unity\Hub\Editor\{0}\Editor\Unity.exe' -f $version)
    if (-not (Test-Path -LiteralPath $exe)) { throw ("Unity {0} not found at '{1}'. Install it or pass -UnityPath." -f $version, $exe) }
    return $exe
}

# Starts Unity with the arguments, waits for it and returns its exit code (-1 after a timeout, when the process tree is killed).
function Invoke-Unity([string]$exe, [string[]]$unityArgs) {
    $argLine = ($unityArgs | ForEach-Object {
            if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
        }) -join ' '
    $process = Start-Process -FilePath $exe -ArgumentList $argLine -PassThru -NoNewWindow
    $null = $process.Handle   # keeps the handle, so that ExitCode is available after the process has ended
    if (-not $process.WaitForExit($RunTimeoutMinutes * 60 * 1000)) {
        & taskkill /PID $process.Id /T /F | Out-Null
        return -1
    }
    $process.WaitForExit()
    return $process.ExitCode
}

# Opens the project in batch mode without tests: imports assets, compiles scripts and shaders. Stops the script when that fails.
function Import-Project($side) {
    $log = Join-Path $side.Dir 'import.log'
    Write-Step ('Importing {0} ({1}); the first import of a new worktree takes a while' -f $side.Name, $side.Sha.Substring(0, 9))
    $code = Invoke-Unity $side.Unity @('-batchmode', '-quit', '-projectPath', $side.Path, '-logFile', $log)
    if ($code -ne 0) { throw ('Unity could not import {0} (exit code {1}). See {2}' -f $side.Name, $code, $log) }
}

# Runs the suite once for a side. Returns $true when a result file was written. Failed runs are added to $failures.
function Invoke-Suite($side, [string]$label, [string]$resultPath) {
    $log = [IO.Path]::ChangeExtension($resultPath, '.log')
    $unityArgs = @('-batchmode', '-projectPath', $side.Path, '-runTests', '-testPlatform', 'PlayMode',
        '-testResults', $resultPath, '-logFile', $log)
    if ($AssemblyName) { $unityArgs += @('-assemblyNames', $AssemblyName) }
    if ($TestCategory) { $unityArgs += @('-testCategory', $TestCategory) }
    if ($TestFilter) { $unityArgs += @('-testFilter', $TestFilter) }

    $clock = [Diagnostics.Stopwatch]::StartNew()
    Write-Step ('{0}: {1} ...' -f $label, $side.Name)
    $code = Invoke-Unity $side.Unity $unityArgs
    $clock.Stop()
    $minutes = [math]::Round($clock.Elapsed.TotalMinutes, 1)

    # Unity exits with 0 when every test passed and with 2 when some tests failed; both write the result file.
    if (-not (Test-Path -LiteralPath $resultPath)) {
        $failures.Add(('{0} {1}: no result file (exit code {2}); see {3}' -f $label, $side.Name, $code, $log))
        Write-Warning ('{0}: {1} wrote no result file (exit code {2}, {3} min). See {4}' -f $label, $side.Name, $code, $minutes, $log)
        return $false
    }

    $note = ''
    try {
        $doc = New-Object Xml.XmlDocument
        $doc.Load($resultPath)
        $cases = @($doc.SelectNodes('//test-case'))
        $failed = @($cases | Where-Object { $_.GetAttribute('result') -notlike 'Passed*' })
        if (-not $TestFilter -and $cases.Count -ne $ExpectedTests) {
            $note += (' WARNING: {0} test cases, expected {1}.' -f $cases.Count, $ExpectedTests)
            $failures.Add(('{0} {1}: {2} test cases, expected {3}' -f $label, $side.Name, $cases.Count, $ExpectedTests))
        }
        if ($failed.Count -gt 0) {
            $note += (' WARNING: {0} test case(s) did not pass.' -f $failed.Count)
            $failures.Add(('{0} {1}: {2} test case(s) did not pass' -f $label, $side.Name, $failed.Count))
        }
    }
    catch {
        $note += (' WARNING: the result file could not be read: {0}' -f $_.Exception.Message)
        $failures.Add(('{0} {1}: result file unreadable' -f $label, $side.Name))
    }
    Write-Step ('{0}: {1} finished in {2} min (exit code {3}).{4}' -f $label, $side.Name, $minutes, $code, $note)
    return $true
}

# ---------------------------------------------------------------------------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------------------------------------------------------------------------

if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'git was not found on the PATH.' }

if (-not $RepoPath) { $RepoPath = (Invoke-Git $PSScriptRoot @('rev-parse', '--show-toplevel') | Select-Object -First 1).Trim() }
$RepoPath = [IO.Path]::GetFullPath($RepoPath)
if (-not $WorkRoot) { $WorkRoot = Join-Path (Split-Path $RepoPath -Parent) 'mckrill-avbd-perf' }
$WorkRoot = [IO.Path]::GetFullPath($WorkRoot)
if (-not $OutRoot) { $OutRoot = Join-Path $WorkRoot 'results' }
$OutRoot = [IO.Path]::GetFullPath($OutRoot)

$outDir = Join-Path $OutRoot (Get-Date -Format 'yyyyMMdd-HHmmss')
$failures = New-Object 'System.Collections.Generic.List[string]'

if (-not $NoFetch) {
    Write-Step 'git fetch'
    Invoke-Git $RepoPath @('fetch', '--all', '--prune') | Out-Null
}

$sides = @()
foreach ($entry in @(@('baseline', $Baseline), @('candidate', $Candidate))) {
    $name = $entry[0]
    $ref = $entry[1]
    $sha = Resolve-Commit $ref
    $subject = (Invoke-Git $RepoPath @('log', '-1', '--format=%h %s', $sha) | Select-Object -First 1).Trim()
    $path = Join-Path $WorkRoot $name
    Write-Step ('{0}: {1} = {2}' -f $name, $ref, $subject)
    Initialize-Worktree $path $sha

    $dir = Join-Path $outDir $name
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $sides += [pscustomobject]@{
        Name    = $name
        Ref     = $ref
        Sha     = $sha
        Subject = $subject
        Path    = $path
        Dir     = $dir
        Version = (Get-UnityVersion $path)
        Unity   = $null
    }
}

if (-not $UnityPath -and $sides[0].Version -ne $sides[1].Version) {
    throw ('The refs use different Unity versions ({0} and {1}); the comparison would include the editor. Pass -UnityPath to use one editor for both.' -f
        $sides[0].Version, $sides[1].Version)
}
foreach ($side in $sides) { $side.Unity = Get-UnityExe $side.Path }

foreach ($side in $sides) { Import-Project $side }

# What was run, for the record next to the results.
$gpus = @()
try { $gpus = @(Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }) } catch { }
[ordered]@{
    started      = (Get-Date).ToString('o')
    machine      = $env:COMPUTERNAME
    gpus         = $gpus
    unity        = $sides[0].Unity
    runs         = $Runs
    warmup       = [bool]$Warmup
    testCategory = $TestCategory
    testFilter   = $TestFilter
    baseline     = @{ ref = $Baseline; commit = $sides[0].Sha; subject = $sides[0].Subject }
    candidate    = @{ ref = $Candidate; commit = $sides[1].Sha; subject = $sides[1].Subject }
} | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $outDir 'meta.json') -Encoding UTF8

# ---------------------------------------------------------------------------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------------------------------------------------------------------------

$baselineSide = $sides[0]
$candidateSide = $sides[1]

if ($Warmup) {
    $warmDir = Join-Path $outDir 'warmup'
    New-Item -ItemType Directory -Force -Path $warmDir | Out-Null
    foreach ($side in @($baselineSide, $candidateSide)) {
        Invoke-Suite $side 'warm-up' (Join-Path $warmDir ('{0}.xml' -f $side.Name)) | Out-Null
    }
    # A warm-up that failed says nothing about the measured runs, and its result is not compared.
    $failures.Clear()
}

for ($run = 1; $run -le $Runs; $run++) {
    # Alternate who goes first: A B, B A, A B, ...
    if ($run % 2 -eq 1) { $order = @($baselineSide, $candidateSide) } else { $order = @($candidateSide, $baselineSide) }
    foreach ($side in $order) {
        $label = 'run {0}/{1}' -f $run, $Runs
        Invoke-Suite $side $label (Join-Path $side.Dir ('run{0:D2}.xml' -f $run)) | Out-Null
    }
}

# ---------------------------------------------------------------------------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------------------------------------------------------------------------

$compareScript = Join-Path $PSScriptRoot 'multi_compare_perf.py'
$csv = Join-Path $outDir 'comparison.csv'
$compareArgs = @($compareScript, (Join-Path $outDir 'baseline'), (Join-Path $outDir 'candidate'),
    '-o', $csv, '--threshold', $Threshold.ToString([Globalization.CultureInfo]::InvariantCulture))

Write-Host ''
if (Get-Command $Python -ErrorAction SilentlyContinue) {
    Write-Step 'Comparing (median over the runs of each side)'
    & $Python @compareArgs
    if ($LASTEXITCODE -ne 0) { Write-Warning ('multi_compare_perf.py failed (exit code {0}).' -f $LASTEXITCODE) }
}
else {
    $quoted = ($compareArgs | ForEach-Object { '"' + $_ + '"' }) -join ' '
    Write-Warning ("'{0}' was not found. Compare later with:`n  {0} {1}" -f $Python, $quoted)
}

Write-Host ''
Write-Step ('Results: {0}' -f $outDir)
if ($failures.Count -gt 0) {
    Write-Warning ('{0} problem(s):' -f $failures.Count)
    $failures | ForEach-Object { Write-Warning ('  ' + $_) }
    exit 1
}
