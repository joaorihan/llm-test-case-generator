"""Interface local para gerar casos de teste usando o adaptador treinado."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from make_dataset import SYSTEM
from train import MODEL_ID, ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=ROOT / "models" / "best-adapter")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A interface exige GPU ROCm")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda")
    model = PeftModel.from_pretrained(model, args.adapter).to("cuda")
    model.eval()

    def respond(requirement: str, max_new_tokens: int) -> str:
        if not requirement.strip():
            return "Digite um requisito para gerar os casos de teste."
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": requirement.strip()}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(rendered, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=int(max_new_tokens), do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(generated[0, inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()
        try:
            return json.dumps(json.loads(answer), ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            return answer

    with gr.Blocks(title="Gerador de casos de teste") as demo:
        gr.Markdown("# Gerador de casos de teste\nDescreva uma regra de validação de formulário em português.")
        gr.Markdown(f"**Modelo:** {MODEL_ID} + LoRA  ·  **GPU:** {torch.cuda.get_device_name(0)}")
        requirement = gr.Textbox(
            label="Requisito",
            lines=4,
            placeholder="Ex.: O campo idade aceita inteiros de 18 a 65, incluindo os limites.",
        )
        length = gr.Slider(128, 512, value=384, step=64, label="Máximo de tokens gerados")
        button = gr.Button("Gerar casos de teste", variant="primary")
        output = gr.Code(label="Resposta JSON", language="json", lines=18)
        button.click(respond, inputs=[requirement, length], outputs=output)
    demo.launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
