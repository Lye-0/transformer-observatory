import os

import torch


def configure_runtime():
    # Eager CUDA BF16 reduced-precision partial sums diverged from the independent
    # CPU reference at Qwen3's final MLP. Use float32 accumulation, retaining BF16
    # inputs/outputs and the same published tolerances.
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    # On this Windows/PyTorch build, parallel CPU LayerNorm raises handled native
    # access violations. A single CPU thread was measured to avoid that path.
    # CUDA execution is unaffected. Keep an explicit, opt-in override for other PCs.
    default = "1" if os.name == "nt" else "4"
    torch.set_num_threads(max(1, int(os.environ.get("OBSERVATORY_CPU_THREADS", default))))
