# Reproduzir o projeto no Windows

Execute os comandos em PowerShell **na raiz do repositório**, onde estão `requirements.txt` e `src`. Neste PC, se o prompt mostrar `...\new-chat>`, entre na pasta com:

```powershell
Set-Location .\outputs\llm-test-cases-ptbr
```

O ambiente testado usa Python 3.13, PyTorch `2.12.0+rocm7.14.1` e a GPU AMD Radeon RX 9070 XT (`gfx1201`). Reserve cerca de 30 GB livres para downloads, ambiente e arquivos temporários. Consulte a [instalação da AMD para esta GPU](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html?fam=radeon&gfx=gfx1201&gpu=rx-9070-xt&i=pip&os=windows&pytorch-ver=2.12.0&w=compute) se o driver ou as versões mudarem.

## Usar o ambiente já instalado neste PC

O ambiente Python deste trabalho está em `new-chat\.venv`, duas pastas acima do repositório. O PyTorch e as dependências já foram instalados e a GPU foi testada. No terminal da raiz do repositório:

```powershell
$python = (Resolve-Path ..\..\.venv\Scripts\python.exe).Path
$env:HF_HOME = (New-Item -ItemType Directory -Force ..\..\.cache\huggingface).FullName
$env:PYTHONIOENCODING = 'utf-8'
$env:TORCH_BLAS_PREFER_HIPBLASLT = '0'
& $python src\gpu_check.py
```

Os indicadores `gpu_disponivel`, `resultado_na_gpu` e `resultado_finito` devem ser `true`. O resultado também é salvo em [`reports/gpu_check.json`](../reports/gpu_check.json).

## Instalação nova após clonar

Use esta sequência se o ambiente acima não existir. O modelo base será baixado na primeira avaliação ou execução da interface; seus pesos não são incluídos no GitHub.

```powershell
py -3.13 -m venv .venv
$python = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $python -m pip install --upgrade pip
& $python -m pip install --no-cache-dir --index-url https://repo.amd.com/rocm/whl-multi-arch/ 'rocm[libraries,device-gfx1201]==7.14.1'
& $python -m pip install --no-cache-dir --index-url https://repo.amd.com/rocm/whl-multi-arch/ 'torch[device-gfx1201]==2.12.0+rocm7.14.1'
& $python -m pip install --no-cache-dir -r requirements.txt
$env:HF_HOME = (New-Item -ItemType Directory -Force .\.cache\huggingface).FullName
$env:PYTHONIOENCODING = 'utf-8'
$env:TORCH_BLAS_PREFER_HIPBLASLT = '0'
& $python src\gpu_check.py
```

## Treinar e comparar

Depois de definir `$python` e as variáveis acima, execute a comparação completa:

```powershell
.\scripts\run_experiments.ps1 -Python $python -Cache $env:HF_HOME
```

O script gera os dados, avalia o modelo original, treina três combinações de rank e taxa de aprendizado na GPU, avalia os adaptadores e atualiza [`reports/summary.md`](../reports/summary.md). O adaptador com menor perda de validação é copiado para `models/best-adapter`. Os resultados já obtidos foram preservados em `reports/` para conferência sem repetir o treino.

## Abrir a interface local

```powershell
& $python src\app.py
```

Abra `http://127.0.0.1:7860` no navegador. Digite uma regra de validação em português e clique em **Gerar casos de teste**. O frontend carrega o adaptador incluído em `models/best-adapter` e exige uma GPU disponível.

## Conferir a evidência

A [captura no VS Code](../screenshots/gpu-vscode.jpg) mostra o relatório da GPU aberto no editor. Se for exigida a saída no terminal integrado, execute nele `& $python src\gpu_check.py` e capture a tela com a saída visível. A [auditoria qualitativa](../reports/audit-challenge.md) registra os erros semânticos observados nas respostas; o formato JSON correto, por si só, não garante casos de teste corretos.
