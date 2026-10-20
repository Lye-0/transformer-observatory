import os

import torch


def configure_runtime():
    # On this Windows/PyTorch build, parallel CPU LayerNorm raises handled native
    # access violations. A single CPU thread was measured to avoid that path.
    # CUDA execution is unaffected. Keep an explicit, opt-in override for other PCs.
    default = '1' if os.name == 'nt' else '4'
    torch.set_num_threads(max(1, int(os.environ.get('OBSERVATORY_CPU_THREADS', default))))
