"""Avalia geração na GPU com prompts não vistos durante o treino."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from make_dataset import SYSTEM
from train import MODEL_ID, ROOT, read_jsonl


REQUIRED_KEYS = {"titulo", "entrada", "resultado_esperado"}


def inspect_answer(answer: str) -> dict:
    try:
        parsed = json.loads(answer)
    except json.JSONDecodeError:
        return {"json_valido": False, "tres_casos": False, "esquema_completo": False}
    cases = parsed.get("casos") if isinstance(parsed, dict) else None
    three = isinstance(cases, list) and len(cases) == 3
    complete = three and all(
        isinstance(item, dict)
        and REQUIRED_KEYS.issubset(item)
        and all(isinstance(item[key], str) and item[key].strip() for key in REQUIRED_KEYS)
        for item in cases
    )
    return {"json_valido": True, "tres_casos": three, "esquema_completo": complete}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--split", choices=["validation", "test", "challenge"], default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A avaliação exige GPU ROCm")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).to("cuda")
    model.eval()
    rows = read_jsonl(ROOT / "data" / f"{args.split}.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    results = []
    for index, row in enumerate(rows, start=1):
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["requisito"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(generated[0, inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()
        result = {
            "id": row["id"],
            "tipo": row["tipo"],
            "requisito": row["requisito"],
            "referencia": row["resposta"],
            "resposta": answer,
            **inspect_answer(answer),
        }
        results.append(result)
        print(f"{args.name}: {index}/{len(rows)} {row['id']} JSON={result['json_valido']}", flush=True)
    torch.cuda.synchronize()
    keys = ("json_valido", "tres_casos", "esquema_completo")
    metrics = {key: round(sum(bool(r[key]) for r in results) / len(results), 4) for key in keys}
    report = {
        "name": args.name,
        "split": args.split,
        "base_model": MODEL_ID,
        "adapter": str(args.adapter) if args.adapter else None,
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "rocm_hip": torch.version.hip,
        "total": len(results),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "metrics": metrics,
        "items": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "items"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
