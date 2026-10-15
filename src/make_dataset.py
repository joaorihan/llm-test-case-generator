"""Cria um conjunto reproduzível de requisitos e casos de teste em português.

Os exemplos são sintéticos e gerados por regras explícitas. Cada requisito
fica em apenas uma partição. O conjunto de teste nunca é usado no treino.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SYSTEM = (
    "Você é um analista de testes de software. A partir do requisito, gere "
    "exatamente três casos de teste: um cenário válido e dois cenários inválidos "
    "ou de fronteira. Responda somente JSON válido no formato "
    '{"casos":[{"titulo":"...","entrada":"...","resultado_esperado":"..."}]}. '
    "Não inclua explicações fora do JSON."
)

FIELDS = [
    "nome completo", "nome de usuário", "telefone", "endereço", "cidade",
    "bairro", "código do produto", "matrícula", "nome do projeto", "empresa",
    "cargo", "departamento", "número do pedido", "descrição", "título",
    "apelido", "identificador", "nome do cliente", "nome do fornecedor", "assunto",
]
SYSTEMS = [
    "portal de compras", "sistema acadêmico", "aplicativo de reservas", "intranet",
    "plataforma de cursos", "sistema de chamados", "portal de fornecedores", "loja virtual",
    "sistema de estoque", "aplicativo financeiro", "agenda de serviços", "portal de eventos",
    "sistema de contratos", "painel administrativo", "aplicativo de entregas", "portal de RH",
    "sistema de manutenção", "plataforma de projetos", "aplicativo de suporte", "sistema de inscrições",
]


def case(title: str, entry: str, expected: str) -> dict[str, str]:
    return {"titulo": title, "entrada": entry, "resultado_esperado": expected}


def record(kind: str, idx: int, requirement: str, cases: list[dict[str, str]]) -> dict:
    assert len(cases) == 3
    return {
        "id": f"{kind}-{idx:02d}",
        "tipo": kind,
        "requisito": requirement,
        "resposta": json.dumps({"casos": cases}, ensure_ascii=False, separators=(",", ":")),
    }


def build() -> list[dict]:
    records: list[dict] = []
    for idx in range(40):
        field = FIELDS[idx % len(FIELDS)]
        application = SYSTEMS[idx % len(SYSTEMS)]
        wording = idx // len(FIELDS)

        requirement = (
            f"No formulário de cadastro, o campo '{field}' é obrigatório."
            if wording == 0 else
            f"Ao salvar um registro, não permita que '{field}' fique sem conteúdo."
        )
        records.append(record("obrigatorio", idx, requirement, [
            case("Valor preenchido", f"{field}=valor válido", "Registro aceito."),
            case("Valor vazio", f"{field}=vazio", "Registro rejeitado; informar que o campo é obrigatório."),
            case("Apenas espaços", f"{field}=espaços em branco", "Registro rejeitado; espaços não contam como conteúdo."),
        ]))

        minimum = 3 + idx % 8
        maximum = minimum + 8 + idx % 6
        requirement = (
            f"O campo '{field}' deve ter entre {minimum} e {maximum} caracteres, inclusive."
            if wording == 0 else
            f"Valide '{field}': aceite no mínimo {minimum} e no máximo {maximum} caracteres."
        )
        records.append(record("comprimento", idx, requirement, [
            case("No limite mínimo", f"{field} com {minimum} caracteres", "Valor aceito."),
            case("Abaixo do mínimo", f"{field} com {minimum - 1} caracteres", f"Valor rejeitado; mínimo de {minimum} caracteres."),
            case("Acima do máximo", f"{field} com {maximum + 1} caracteres", f"Valor rejeitado; máximo de {maximum} caracteres."),
        ]))

        numeric_field = ["quantidade", "idade", "número de parcelas", "vagas", "limite de itens"][idx % 5]
        low = 1 + idx % 7
        high = low + 4 + idx % 9
        requirement = (
            f"O campo '{numeric_field}' aceita inteiros de {low} a {high}, incluindo os limites."
            if wording == 0 else
            f"Rejeite '{numeric_field}' quando o inteiro informado for menor que {low} ou maior que {high}."
        )
        records.append(record("faixa_inteira", idx, requirement, [
            case("Limite inferior", f"{numeric_field}={low}", "Valor aceito."),
            case("Abaixo da faixa", f"{numeric_field}={low - 1}", f"Valor rejeitado; mínimo permitido é {low}."),
            case("Acima da faixa", f"{numeric_field}={high + 1}", f"Valor rejeitado; máximo permitido é {high}."),
        ]))

        actor = ["cliente", "fornecedor", "aluno", "professor", "responsável", "usuário", "gestor", "técnico", "solicitante", "colaborador"][idx % 10]
        email_field = f"email do {actor}"
        requirement = (
            f"No {application}, o campo '{email_field}' deve conter um endereço de email válido."
            if wording == 0 else
            f"No {application}, antes de salvar, valide a sintaxe do endereço em '{email_field}'."
        )
        records.append(record("email", idx, requirement, [
            case("Endereço válido", f"{email_field}=ana{idx}@exemplo.com", "Endereço aceito."),
            case("Sem arroba", f"{email_field}=ana{idx}exemplo.com", "Endereço rejeitado por formato inválido."),
            case("Sem domínio", f"{email_field}=ana{idx}@", "Endereço rejeitado por formato inválido."),
        ]))

        option_field, choices = [
            ("prioridade", ["baixa", "média", "alta"]),
            ("status", ["aberto", "em andamento", "concluído"]),
            ("turno", ["manhã", "tarde", "noite"]),
            ("tipo de entrega", ["retirada", "correio", "transportadora"]),
            ("perfil", ["leitor", "editor", "administrador"]),
        ][idx % 5]
        allowed = ", ".join(choices)
        requirement = (
            f"No {application}, o campo obrigatório '{option_field}' aceita apenas: {allowed}."
            if wording == 0 else
            f"No {application}, valide '{option_field}' como obrigatório e restrito às opções {allowed}."
        )
        records.append(record("opcoes", idx, requirement, [
            case("Opção permitida", f"{option_field}={choices[0]}", "Valor aceito."),
            case("Opção desconhecida", f"{option_field}=outro", "Valor rejeitado; opção não permitida."),
            case("Sem seleção", f"{option_field}=vazio", "Valor rejeitado; seleção obrigatória."),
        ]))

        context = ["reserva", "projeto", "contrato", "curso", "campanha", "licença", "evento", "tarefa", "inspeção", "serviço"][idx % 10]
        requirement = (
            f"No {application}, na {context}, a data final deve ser igual ou posterior à data inicial."
            if wording == 0 else
            f"No {application}, ao cadastrar {context}, impeça que a data de término seja anterior à data de início."
        )
        records.append(record("ordem_datas", idx, requirement, [
            case("Término posterior", "início=10/03/2026; término=12/03/2026", "Período aceito."),
            case("Mesmo dia", "início=10/03/2026; término=10/03/2026", "Período aceito; datas iguais são permitidas."),
            case("Término anterior", "início=12/03/2026; término=10/03/2026", "Período rejeitado; término anterior ao início."),
        ]))

    ids = [item["id"] for item in records]
    assert len(ids) == len(set(ids))
    assert len({item["requisito"] for item in records}) == len(records)
    return records


def build_challenge() -> list[dict]:
    """Requisitos redigidos separadamente, com valores e frases fora dos moldes de treino."""
    return [
        record("obrigatorio", 90, "Para abrir um chamado, o código de rastreio precisa ser informado; espaços isolados não valem.", [
            case("Código informado", "código de rastreio=BR123", "Chamado aceito."),
            case("Código ausente", "código de rastreio=vazio", "Chamado rejeitado; código obrigatório."),
            case("Só espaços", "código de rastreio=espaços", "Chamado rejeitado; espaços não são conteúdo."),
        ]),
        record("obrigatorio", 91, "A justificativa da solicitação não pode ficar em branco ao enviar o formulário.", [
            case("Justificativa presente", "justificativa=necessidade operacional", "Solicitação aceita."),
            case("Justificativa vazia", "justificativa=vazio", "Solicitação rejeitada; justificativa obrigatória."),
            case("Espaços", "justificativa=espaços", "Solicitação rejeitada; espaços não são justificativa."),
        ]),
        record("comprimento", 90, "No cadastro, aceite apelidos com 5 a 18 caracteres, contando os extremos.", [
            case("Mínimo válido", "apelido com 5 caracteres", "Apelido aceito."),
            case("Curto demais", "apelido com 4 caracteres", "Apelido rejeitado; mínimo de 5."),
            case("Longo demais", "apelido com 19 caracteres", "Apelido rejeitado; máximo de 18."),
        ]),
        record("comprimento", 91, "Uma observação enviada pelo usuário precisa ter pelo menos 12 e no máximo 80 caracteres.", [
            case("No limite", "observação com 12 caracteres", "Observação aceita."),
            case("Abaixo", "observação com 11 caracteres", "Observação rejeitada; mínimo de 12."),
            case("Acima", "observação com 81 caracteres", "Observação rejeitada; máximo de 80."),
        ]),
        record("faixa_inteira", 90, "Permita registrar de 2 até 15 participantes por turma, sem números fora dessa faixa.", [
            case("Mínimo", "participantes=2", "Quantidade aceita."),
            case("Um a menos", "participantes=1", "Quantidade rejeitada; mínimo de 2."),
            case("Um a mais", "participantes=16", "Quantidade rejeitada; máximo de 15."),
        ]),
        record("faixa_inteira", 91, "O usuário pode configurar de zero a três tentativas de repetição, inclusive.", [
            case("Zero permitido", "tentativas=0", "Valor aceito."),
            case("Negativo", "tentativas=-1", "Valor rejeitado; mínimo de 0."),
            case("Acima", "tentativas=4", "Valor rejeitado; máximo de 3."),
        ]),
        record("email", 90, "Na conta do usuário, verifique se o email alternativo tem sintaxe de endereço eletrônico.", [
            case("Válido", "email alternativo=ana@exemplo.com", "Endereço aceito."),
            case("Falta arroba", "email alternativo=anaexemplo.com", "Endereço rejeitado."),
            case("Falta domínio", "email alternativo=ana@", "Endereço rejeitado."),
        ]),
        record("email", 91, "O contato técnico informado na abertura de incidente deve ser um email bem formado.", [
            case("Válido", "contato técnico=suporte@empresa.com", "Endereço aceito."),
            case("Sem arroba", "contato técnico=suporteempresa.com", "Endereço rejeitado."),
            case("Sem domínio", "contato técnico=suporte@", "Endereço rejeitado."),
        ]),
        record("opcoes", 90, "Na compra, o tipo de pagamento é obrigatório e só pode ser pix, boleto ou cartão.", [
            case("Pix", "pagamento=pix", "Compra aceita."),
            case("Tipo desconhecido", "pagamento=cheque", "Compra rejeitada; tipo não permitido."),
            case("Ausente", "pagamento=vazio", "Compra rejeitada; pagamento obrigatório."),
        ]),
        record("opcoes", 91, "Escolha obrigatoriamente um nível de acesso entre consulta, escrita e gestão.", [
            case("Consulta", "acesso=consulta", "Seleção aceita."),
            case("Outro nível", "acesso=superadmin", "Seleção rejeitada."),
            case("Sem nível", "acesso=vazio", "Seleção rejeitada; escolha obrigatória."),
        ]),
        record("ordem_datas", 90, "O prazo de entrega pode coincidir com a solicitação, mas nunca ser anterior a ela.", [
            case("Posterior", "solicitação=01/04/2026; entrega=03/04/2026", "Prazo aceito."),
            case("Mesmo dia", "solicitação=01/04/2026; entrega=01/04/2026", "Prazo aceito."),
            case("Anterior", "solicitação=03/04/2026; entrega=01/04/2026", "Prazo rejeitado."),
        ]),
        record("ordem_datas", 91, "Em uma manutenção, a data de encerramento não pode anteceder a data de abertura.", [
            case("Posterior", "abertura=10/05/2026; encerramento=11/05/2026", "Período aceito."),
            case("Igual", "abertura=10/05/2026; encerramento=10/05/2026", "Período aceito."),
            case("Anterior", "abertura=11/05/2026; encerramento=10/05/2026", "Período rejeitado."),
        ]),
    ]


def main() -> None:
    DATA.mkdir(exist_ok=True)
    partitions = {"train": [], "validation": [], "test": []}
    for item in build():
        suffix = int(item["id"].split("-")[-1]) % 10
        split = "validation" if suffix == 0 else "test" if suffix == 1 else "train"
        partitions[split].append(item)
    for split, rows in partitions.items():
        path = DATA / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} exemplos -> {path}")
    challenge = DATA / "challenge.jsonl"
    with challenge.open("w", encoding="utf-8") as handle:
        for row in build_challenge():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"challenge: {len(build_challenge())} exemplos -> {challenge}")


if __name__ == "__main__":
    main()
