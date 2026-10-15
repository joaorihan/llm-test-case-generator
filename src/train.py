"""Ajuste fino supervisionado com LoRA, inteiramente executado na GPU.

Exemplo: python src/train.py --name lora-r8-lr2e4 --rank 8 --lr 0.0002
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from make_dataset import SYSTEM


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class TestCaseDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.examples = []
        for row in rows:
            prompt = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": row["requisito"]},
            ]
            prefix = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
            if hasattr(prefix, "input_ids"):
                prefix = prefix["input_ids"]
            if prefix and isinstance(prefix[0], list):
                prefix = prefix[0]
            response_ids = tokenizer.encode(row["resposta"], add_special_tokens=False)
            full = prefix + response_ids + [tokenizer.eos_token_id]
            if len(full) > max_length:
                raise ValueError(f"{row['id']} usa {len(full)} tokens; aumente --max-length")
            labels = [-100] * len(prefix) + full[len(prefix) :]
            self.examples.append({"input_ids": full, "labels": labels})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]


def collator(pad_id: int):
    def pad(batch: list[dict]) -> dict[str, torch.Tensor]:
        width = max(len(item["input_ids"]) for item in batch)
        ids, masks, labels = [], [], []
        for item in batch:
            count = width - len(item["input_ids"])
            ids.append(item["input_ids"] + [pad_id] * count)
            masks.append([1] * len(item["input_ids"]) + [0] * count)
            labels.append(item["labels"] + [-100] * count)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long, device="cuda"),
            "attention_mask": torch.tensor(masks, dtype=torch.long, device="cuda"),
            "labels": torch.tensor(labels, dtype=torch.long, device="cuda"),
        }

    return pad


@torch.inference_mode()
def validation_loss(model, loader: DataLoader) -> float:
    model.eval()
    losses = []
    for batch in loader:
        losses.append(float(model(**batch).loss.item()))
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-steps", type=int, default=0, help="0 = treino completo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Este experimento exige GPU ROCm")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("A configuração requer BF16 na GPU")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train = TestCaseDataset(read_jsonl(ROOT / "data" / "train.jsonl"), tokenizer, args.max_length)
    validation = TestCaseDataset(read_jsonl(ROOT / "data" / "validation.jsonl"), tokenizer, args.max_length)
    pad = collator(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=pad, generator=generator)
    val_loader = DataLoader(validation, batch_size=1, collate_fn=pad)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.rank * 2,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    optimizer_steps = 0
    history = []
    stop = False
    for epoch in range(args.epochs):
        model.train()
        running = []
        for batch_index, batch in enumerate(train_loader):
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss não finita no passo {global_step}")
            (loss / args.accumulation).backward()
            running.append(float(loss.detach().item()))
            global_step += 1
            if global_step % args.accumulation == 0 or batch_index + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            if global_step % 20 == 0:
                print(f"{args.name}: época {epoch + 1}, passo {global_step}, loss {sum(running[-20:]) / min(20, len(running)):.4f}", flush=True)
            if args.max_steps and global_step >= args.max_steps:
                stop = True
                break
        val = validation_loss(model, val_loader)
        row = {
            "epoch": epoch + 1,
            "train_loss": round(sum(running) / len(running), 5),
            "validation_loss": round(val, 5),
            "steps": global_step,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if stop:
            break

    torch.cuda.synchronize()
    adapter_dir = args.run_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    report = {
        "name": args.name,
        "base_model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "rocm_hip": torch.version.hip,
        "seed": args.seed,
        "rank": args.rank,
        "learning_rate": args.lr,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.accumulation,
        "max_length": args.max_length,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "trainable_parameters": trainable,
        "optimizer_steps": optimizer_steps,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "history": history,
    }
    (args.run_dir / "train_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
