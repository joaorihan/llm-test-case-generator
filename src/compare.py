"""Resume os experimentos e prepara uma revisão humana cega das respostas."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    paths = [REPORTS / "baseline.json"] + sorted(REPORTS.glob("lora-*.json"))
    reports = [load(path) for path in paths if path.exists()]
    reports = [report for report in reports if report["split"] == "test"]
    if not reports:
        raise RuntimeError("Nenhum relatório de avaliação encontrado")

    lines = [
        "# Comparação dos experimentos",
        "",
        "Todos os prompts desta tabela vêm de `data/test.jsonl`, que não foi usado no treino.",
        "As métricas automáticas medem formato; correção e cobertura exigem revisão humana.",
        "",
        "| Execução | JSON válido | 3 casos | Esquema completo | Perda de validação | Tempo de avaliação | Pico de VRAM |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        metrics = report["metrics"]
        training_path = REPORTS / f"training-{report['name']}.json"
        validation_loss = load(training_path)["history"][-1]["validation_loss"] if training_path.exists() else None
        loss_cell = f"{validation_loss:.5f}" if validation_loss is not None else "—"
        lines.append(
            f"| {report['name']} | {metrics['json_valido']:.1%} | "
            f"{metrics['tres_casos']:.1%} | {metrics['esquema_completo']:.1%} | "
            f"{loss_cell} | {report['elapsed_seconds']:.1f} s | {report['peak_vram_gb']:.2f} GB |"
        )
    lines.extend([
        "",
        "## Critérios de revisão humana",
        "",
        "Para cada resposta, atribua 0, 1 ou 2 pontos a cada critério:",
        "",
        "1. Correção: os resultados esperados obedecem ao requisito?",
        "2. Cobertura: há cenário válido e cenários inválidos ou de fronteira relevantes?",
        "3. Clareza: os casos podem ser executados por outra pessoa sem adivinhar detalhes?",
        "",
        "A planilha `human_review.csv` contém pares de respostas sem nome do modelo. "
        "Preencha os pontos antes de consultar o arquivo de correspondência `review_key.json`.",
        "",
    ])
    (REPORTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    if len(reports) < 2:
        return
    candidates = [
        (load(REPORTS / f"training-{report['name']}.json")["history"][-1]["validation_loss"], report)
        for report in reports[1:]
        if (REPORTS / f"training-{report['name']}.json").exists()
    ]
    if not candidates:
        raise RuntimeError("Copie os train_metrics.json para reports/training-<nome>.json")
    selected_loss, best = min(candidates, key=lambda pair: pair[0])
    with (REPORTS / "summary.md").open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n## Modelo selecionado\n\n`{best['name']}` teve a menor perda de validação "
            f"({selected_loss:.5f}); a seleção não usou o conjunto de teste. "
            "O teste serve para relatar o desempenho após a escolha.\n"
        )
    source_adapter = ROOT / "runs" / best["name"] / "adapter"
    if source_adapter.exists():
        bundled_adapter = ROOT / "models" / "best-adapter"
        bundled_adapter.mkdir(parents=True, exist_ok=True)
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            shutil.copy2(source_adapter / filename, bundled_adapter / filename)
    baseline_by_id = {item["id"]: item for item in reports[0]["items"]}
    best_by_id = {item["id"]: item for item in best["items"]}
    ordered_ids = sorted(set(baseline_by_id) & set(best_by_id))
    rows, key = [], []
    for index, item_id in enumerate(ordered_ids):
        pair = [baseline_by_id[item_id], best_by_id[item_id]]
        if index % 2:
            pair.reverse()
            labels = [best["name"], "baseline"]
        else:
            labels = ["baseline", best["name"]]
        for letter, item, label in zip("AB", pair, labels):
            rows.append({
                "id": item_id,
                "versao": letter,
                "tipo": item["tipo"],
                "requisito": item["requisito"],
                "resposta": item["resposta"],
                "correcao_0a2": "",
                "cobertura_0a2": "",
                "clareza_0a2": "",
                "comentario": "",
            })
            key.append({"id": item_id, "versao": letter, "modelo": label})
    with (REPORTS / "human_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (REPORTS / "review_key.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Resumo salvo em {REPORTS / 'summary.md'}")
    print(f"Revisão cega preparada para {len(ordered_ids)} requisitos")


if __name__ == "__main__":
    main()
