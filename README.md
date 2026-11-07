# H-Spec drafter for Qwen3.8-27B (speculators)

[vllm-project/speculators](https://github.com/vllm-project/speculators) branch `hspec_qwen_27B` at `577a974`
(Megan Flynn: Qwen3.8-27B support and the partial-RoPE fix), plus one change: `init_from_verifier` also
initializes the drafter's `q_proj` from the target, taking the query half of Qwen3.8's gated
`q_proj` (`[query | gate]` per head), as in the H-Spec paper recipe (`--init-q-proj verifier`).

## Training command

```bash
torchrun --standalone --nproc_per_node 2 examples/qwen3.8-27b-hspec/rank_launch.py \
    --speculator-type mamba_attn_hybrid \
    --verifier-name-or-path Qwen/Qwen3.8-27B \
    --draft-config examples/qwen3.8-27b-hspec/draft_config.json \
    --data-path /path/to/prepared_data \
    --hidden-states-path /path/to/hidden_states \
    --hidden-states-backend file_kv \
    --vllm-endpoint http://localhost:8000/v1 --on-missing generate \
    --target-layer-ids 3 19 35 51 63 \
    --attn-kv-layer-ids 35 51 63 \
    --block-pattern mamba mlp mamba attention mlp mamba attention mlp mamba attention mlp \
    --mamba-num-heads 80 --mamba-head-dim 64 --mamba-d-state 4 --mamba-n-groups 4 \
    --block-size 8 --draft-vocab-size 64000 --max-anchors 512 --markov-rank 256 \
    --loss-fn '{"ce": 0.1, "tv": 0.9}' \
    --optimizer adamw --lr 0.0006 --weight-decay 0.01 \
    --scheduler-type cosine --scheduler-warmup-ratio 0.01 \
    --epochs 1 --checkpoint-freq 0.25 --gradient-checkpointing \
    --save-path /path/to/checkpoints
```

Before running it:

- Environment, `speculators prepare-data`, and the hidden-state/KV extraction server are as in
  [`docs/hspec_qwen_27B.md`](docs/hspec_qwen_27B.md) (vLLM: `MeganEFlynn/vllm` branch `hspec_qwen_27B`),
  except that the server's `kv_cache_layer_ids` must be `[35, 51, 63]` to match `--attn-kv-layer-ids`.
- `rank_launch.py` pins training rank r to GPU `r + RANK_GPU_OFFSET` (default 2). The reference run had the
  extraction server on GPUs 0-1 (`--data-parallel-size 2`) and the two ranks on GPUs 2-3 (4x A100 80GB).

## Notes

- **Do not set `TORCHINDUCTOR_MAX_AUTOTUNE=1` together with `--gradient-checkpointing`.** With both, H-Spec's
  gradients are ~1e5x too large (step-0 grad norm 1.36e6, vs 18.3 with checkpointing alone and 10.6 with
  autotune alone, same weights and batch) and the drafter collapses to an input-independent output within
  ~50 steps, with no error or NaN. DSpark with both settings is unaffected (50.8).
- `draft_config.json` sets the target's RoPE (`rope_theta` 1e7, `partial_rotary_factor` 0.25) and a
  2,048-token sliding window on all three attention layers. With `--draft-config`, the trainer takes the
  window from this file, not from `--sliding-window`.
- Architecture (4M3A): three attention sublayers read the target's KV at layers 35/51/63; the Mamba state is
  initialized from the fused last-token hidden states of layers 3/19/35/51/63. 80 Mamba heads give 1.92B
  trainable parameters, matched to a 5-layer DSpark (1.915B) at the same widths.
- Reference run (100K rows of `shanjiaz/OpenPerfectBlend-Qwen38-27B-regenerated`, 1 epoch): at step 600 the
  train loss / acceptance is 0.802 / 0.290, vs 0.856 / 0.209 for DSpark with the same recipe. That run used
  an equivalent partial-RoPE patch on `f165c95` instead of `577a974`'s implementation; both produce the
  target's rotation exactly (max difference vs vLLM's rotation: 0).
