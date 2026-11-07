# torchrun entry point for speculators.train with each rank pinned to one physical GPU:
# rank r -> GPU r + RANK_GPU_OFFSET (default 2, leaving GPUs 0-1 for the vLLM extraction server).
import os
import runpy
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = str(
    int(os.environ.get("LOCAL_RANK", "0")) + int(os.environ.get("RANK_GPU_OFFSET", "2"))
)

import torch._dynamo  # noqa: E402

# As in the reference run. Required only together with TORCHINDUCTOR_MAX_AUTOTUNE=1, which must
# not be used with --gradient-checkpointing (see README).
torch._dynamo.config.optimize_ddp = False

# Spawned DataLoader workers re-run this file as __mp_main__; only torchrun's direct call trains.
if __name__ == "__main__":
    sys.argv[0] = "-m speculators.train"
    runpy.run_module("speculators.train", run_name="__main__")
