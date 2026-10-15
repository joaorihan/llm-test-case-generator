# Comparação dos experimentos

Todos os prompts desta tabela vêm de `data/test.jsonl`, que não foi usado no treino.
As métricas automáticas medem formato; correção e cobertura exigem revisão humana.

| Execução | JSON válido | 3 casos | Esquema completo | Perda de validação | Tempo de avaliação | Pico de VRAM |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 29.2% | 29.2% | 0.0% | — | 113.4 s | 3.14 GB |
| lora-r16-lr1e4 | 100.0% | 100.0% | 100.0% | 0.01879 | 144.0 s | 3.14 GB |
| lora-r16-lr2e4 | 100.0% | 100.0% | 100.0% | 0.00930 | 139.2 s | 3.14 GB |
| lora-r8-lr2e4 | 100.0% | 100.0% | 100.0% | 0.01544 | 137.2 s | 3.14 GB |

## Critérios de revisão humana

Para cada resposta, atribua 0, 1 ou 2 pontos a cada critério:

1. Correção: os resultados esperados obedecem ao requisito?
2. Cobertura: há cenário válido e cenários inválidos ou de fronteira relevantes?
3. Clareza: os casos podem ser executados por outra pessoa sem adivinhar detalhes?

A planilha `human_review.csv` contém pares de respostas sem nome do modelo. Preencha os pontos antes de consultar o arquivo de correspondência `review_key.json`.

## Modelo selecionado

`lora-r16-lr2e4` teve a menor perda de validação (0.00930); a seleção não usou o conjunto de teste. O teste serve para relatar o desempenho após a escolha.
