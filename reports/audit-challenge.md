# Auditoria das respostas fora dos moldes de treino

Foram lidas as 12 respostas de `challenge-best.json`, geradas na GPU com o adaptador `lora-r16-lr2e4`. Cada requisito foi classificado conforme os resultados esperados e a utilidade dos três casos. Esta é uma inspeção inicial feita durante a preparação do projeto; a planilha `human_review.csv` permite uma revisão cega pelos integrantes do grupo.

| Requisito | Julgamento | Observação |
|---|---|---|
| `obrigatorio-90` | Correto | Cobre preenchido, vazio e apenas espaços; há um defeito de texto no título do terceiro caso. |
| `obrigatorio-91` | Correto | Cobre preenchido, vazio e apenas espaços. |
| `comprimento-90` | Correto | Usa 5, 4 e 19 caracteres para a faixa de 5 a 18. |
| `comprimento-91` | Correto | Usa 12, 11 e 81 caracteres para a faixa de 12 a 80. |
| `faixa_inteira-90` | Correto | Usa 2, 1 e 16 para a faixa de 2 a 15. |
| `faixa_inteira-91` | Incorreto | Trata 1 como menor que o mínimo 0, embora 1 seja permitido. |
| `email-90` | Correto | Inclui endereço válido, sem arroba e sem domínio. |
| `email-91` | Incorreto | Repete o mesmo email válido e dá resultados opostos; introduz regra de nome não solicitada. |
| `opcoes-90` | Correto | Inclui opção permitida, desconhecida e vazia. |
| `opcoes-91` | Parcial | O terceiro caso usa “menor que permitido” como valor, sem indicar uma opção concreta; também pressupõe hierarquia de acesso não descrita. |
| `ordem_datas-90` | Incorreto | O caso chamado “anterior” usa prazo em 11/03 depois da solicitação em 10/03. |
| `ordem_datas-91` | Incorreto | O caso rejeitado usa abertura em 09/03 e encerramento em 12/03, que deveria ser aceito. |

**Síntese:** 7 corretos, 1 parcial e 4 incorretos nesta inspeção. O modelo acertou o formato JSON em 12/12, mas isso não equivale a 12 respostas semanticamente corretas. É necessário revisar os casos antes de usá-los em testes reais. Os arquivos `challenge-baseline.json` e `challenge-best.json` preservam as respostas completas para conferência.
