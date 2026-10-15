"""Teste inicial de geração com o modelo original na GPU."""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from make_dataset import SYSTEM


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU não detectada pelo PyTorch ROCm")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda")
    model.eval()
    prompt = "O campo 'idade' aceita inteiros de 18 a 65, incluindo os limites."
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=250, do_sample=False)
    answer = tokenizer.decode(ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    report = {
        "modelo": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0),
        "prompt": prompt,
        "resposta": answer,
        "pico_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
