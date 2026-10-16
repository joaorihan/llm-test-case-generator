# Revisão qualitativa das 48 respostas do conjunto de teste

Esta é uma **avaliação inicial feita pelo assistente**, em 15/10/2026, das 24 respostas do modelo original e das 24 respostas do adaptador `lora-r16-lr2e4` preservadas em `baseline.json` e `lora-r16-lr2e4.json`. Ela **não substitui** a revisão independente dos quatro ou cinco integrantes do grupo. O avaliador conhecia a identidade dos modelos; portanto, as notas abaixo não são uma avaliação cega. A planilha `human_review.csv` continua sem notas para que o grupo possa preenchê-la antes de consultar `review_key.json`.

## Rubrica

Cada resposta recebe três notas de 0 a 2, na ordem **correção / cobertura / clareza**.

| Nota | Correção | Cobertura | Clareza |
|---|---|---|---|
| 0 | Há resultado esperado contraditório com o requisito ou a resposta não permite julgar os casos. | Não há contraste útil entre cenários. | A saída não pode ser usada no formato pedido e exige reconstrução substancial. |
| 1 | Parte dos resultados faz sentido, mas há erro, regra inventada ou conclusão ambígua. | Há algum contraste, mas falta um cenário importante. | Os casos são interpretáveis, mas exigem correção de formato ou concretização de entradas. |
| 2 | Os resultados dos três cenários respeitam o requisito. | Há cenário permitido e cenários inválidos ou de fronteira relevantes. | Entradas e resultados são suficientemente específicos para execução sem escolher valores ausentes. |

Um JSON inválido não zera automaticamente a correção: o conteúdo ainda pode revelar acertos e erros. Porém, reduz a clareza. Descrições como “campo com 4 caracteres” e “valor válido” não são entradas literais; mesmo com JSON correto, recebem clareza 1.

## Notas por requisito

| ID | Original | LoRA | Principal observação sobre o original |
|---|---:|---:|---|
| obrigatorio-01 | 2/2/0 | 2/2/1 | Cobre preenchido, vazio e espaço, mas o JSON tem fechamento incorreto. |
| comprimento-01 | 0/1/0 | 2/2/1 | Classifica uma entrada de quatro caracteres como longa, embora seja o mínimo aceito. |
| faixa_inteira-01 | 1/2/0 | 2/2/2 | Exibe os números como resultados, sem indicar aceitação ou rejeição. |
| email-01 | 1/1/1 | 2/2/2 | O título “sem @” não corresponde à entrada; as outras entradas não cobrem claramente os erros anunciados. |
| opcoes-01 | 1/0/0 | 2/2/2 | Usa só opções permitidas; falta opção desconhecida e campo vazio. |
| ordem_datas-01 | 0/2/0 | 2/2/2 | Rejeita datas iguais, contrariando “igual ou posterior”. |
| obrigatorio-11 | 2/2/1 | 2/2/1 | Cenários coerentes, mas o campo `entrada` é objeto, fora do esquema pedido. |
| comprimento-11 | 0/1/0 | 2/2/1 | Comprimentos dos textos não correspondem aos rótulos e resultados esperados. |
| faixa_inteira-11 | 0/2/1 | 2/2/2 | Aceita 12 apesar do máximo 11. |
| email-11 | 2/2/1 | 2/2/2 | Casos úteis, mas tipos de `entrada` e `resultado_esperado` diferem do esquema. |
| opcoes-11 | 1/0/0 | 2/2/2 | Repete somente valores permitidos. |
| ordem_datas-11 | 0/2/0 | 2/2/2 | Rejeita datas iguais e gera JSON inválido. |
| obrigatorio-21 | 1/1/0 | 2/2/1 | O resultado do valor preenchido não deixa claro que o registro deve ser aceito. |
| comprimento-21 | 2/2/0 | 2/2/1 | A lógica dos cenários é adequada, mas expressões `"a"*20` quebram o JSON. |
| faixa_inteira-21 | 0/2/0 | 2/2/2 | Aceita 9 apesar do máximo 8. |
| email-21 | 1/2/1 | 2/2/2 | Introduz exigência de `.br`, que o requisito não especifica. |
| opcoes-21 | 1/1/0 | 2/2/2 | Mistura rótulos e resultados ambíguos em JSON inválido. |
| ordem_datas-21 | 0/2/1 | 2/2/2 | Trata data posterior e data igual como erro. |
| obrigatorio-31 | 0/2/0 | 2/2/1 | Espera erro para departamento preenchido. |
| comprimento-31 | 2/2/0 | 2/2/1 | A lógica é adequada, mas o JSON usa multiplicação de string. |
| faixa_inteira-31 | 2/2/0 | 2/2/2 | Cenários coerentes em JSON inválido. |
| email-31 | 2/2/1 | 2/2/2 | Cenários coerentes, mas os tipos não seguem o esquema exigido. |
| opcoes-31 | 1/2/0 | 2/2/2 | Cobre permitido, inválido e vazio, mas a estrutura JSON está quebrada. |
| ordem_datas-31 | 0/1/0 | 2/2/2 | Rejeita igualdade e repete a mesma entrada como dois casos distintos. |

## Síntese

| Critério | Original | LoRA |
|---|---:|---:|
| Correção | 22/48 | 48/48 |
| Cobertura | 38/48 | 48/48 |
| Clareza | 7/48 | 40/48 |
| **Total** | **67/144** | **136/144** |

No teste de 24 requisitos, os três cenários de cada resposta LoRA mantêm a regra solicitada nesta inspeção. A nota de clareza cai em oito respostas de campos obrigatórios ou comprimento porque usam entradas descritivas em vez de valores literais. No desafio reformulado, a auditoria separada encontrou **7 corretas, 1 parcial e 4 incorretas** entre 12 respostas LoRA. Assim, o desempenho de formato e esta revisão do teste não autorizam afirmar que o modelo generaliza para requisitos novos sem supervisão.

## Como fazer a revisão do grupo

Cada integrante deve pontuar `human_review.csv` individualmente, sem abrir `review_key.json`. Depois de concluídas as notas, associem A e B aos modelos, comparem divergências e registrem consenso ou média. As notas deste arquivo podem servir como ponto de comparação posterior, mas não como votos dos integrantes. Revisem também exemplos reais de requisitos antes de usar a ferramenta fora da demonstração.
