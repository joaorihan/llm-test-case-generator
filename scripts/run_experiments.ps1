param(
    [string]$Python = '.\.venv\Scripts\python.exe',
    [string]$Cache = '.\.cache\huggingface'
)

$ErrorActionPreference = 'Stop'
$env:HF_HOME = (New-Item -ItemType Directory -Force -Path $Cache).FullName
$env:PYTHONIOENCODING = 'utf-8'
$env:TORCH_BLAS_PREFER_HIPBLASLT = '0'

& $Python src\make_dataset.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python src\gpu_check.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python src\evaluate.py --name baseline --output reports\baseline.json
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$experiments = @(
    @{ Name = 'lora-r8-lr2e4'; Rank = 8; LearningRate = '0.0002' },
    @{ Name = 'lora-r16-lr2e4'; Rank = 16; LearningRate = '0.0002' },
    @{ Name = 'lora-r16-lr1e4'; Rank = 16; LearningRate = '0.0001' }
)

foreach ($experiment in $experiments) {
    $run = Join-Path 'runs' $experiment.Name
    & $Python src\train.py --name $experiment.Name --rank $experiment.Rank --lr $experiment.LearningRate --epochs 2 --run-dir $run
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Copy-Item -LiteralPath (Join-Path $run 'train_metrics.json') -Destination (Join-Path 'reports' ('training-' + $experiment.Name + '.json')) -Force
    & $Python src\evaluate.py --name $experiment.Name --adapter (Join-Path $run 'adapter') --output (Join-Path 'reports' ($experiment.Name + '.json'))
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

& $Python src\compare.py
exit $LASTEXITCODE
