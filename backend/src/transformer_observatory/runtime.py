import os
from pathlib import Path

import torch


class InstanceLock:
    """One backend per catalog; OS releases the lock even after a crash."""

    def __init__(self, data_root):
        root = Path(data_root)
        root.mkdir(parents=True, exist_ok=True)
        self.file = (root / "backend.lock").open("a+b")
        self.file.seek(0, 2)
        if self.file.tell() == 0:
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            raise RuntimeError(
                "同じデータフォルダーを使用するバックエンドが起動済みです。既存のUIを使用するか、先に終了してください。"
            ) from exc

    def close(self):
        if self.file.closed:
            return
        self.file.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        self.file.close()


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
