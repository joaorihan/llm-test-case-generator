"""Comprove que o PyTorch executa operações na GPU AMD.

Execute no terminal integrado do VS Code e capture a saída para a entrega.
"""

import json
import sys

import torch


def main() -> int:
    report = {
        "pytorch": torch.__version__,
        "rocm_hip": torch.version.hip,
        "gpu_disponivel": torch.cuda.is_available(),
    }
    if not report["gpu_disponivel"]:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    device = torch.device("cuda:0")  # A API CUDA do PyTorch também é usada pelo ROCm.
    properties = torch.cuda.get_device_properties(device)
    matrix = torch.randn((1024, 1024), device=device)
    result = matrix @ matrix.T
    torch.cuda.synchronize(device)

    report.update(
        {
            "gpu": properties.name,
            "vram_gb": round(properties.total_memory / 1e9, 2),
            "operacao": "multiplicação de matrizes 1024x1024",
            "resultado_na_gpu": result.is_cuda,
            "resultado_finito": bool(torch.isfinite(result).all().item()),
            "memoria_alocada_mb": round(torch.cuda.memory_allocated(device) / 1e6, 1),
        }
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["resultado_na_gpu"] and report["resultado_finito"] else 1


if __name__ == "__main__":
    sys.exit(main())
