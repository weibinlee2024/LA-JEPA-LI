[Uploading README.md…]()
# LA-JEPA-LI

Independent reproduction of **La-JEPA** (*Compact Robot Policies with Latent World Modeling*, [arXiv:2611.08156](https://arxiv.org/abs/2611.08156)).

This repository is **not** the official `mt-clemente/lajepa` release. It reimplements the paper recipe for training and evaluation on LIBERO / LIBERO-Plus / RoboTwin under the name **LA-JEPA-LI**.

## Features

- Frozen V-JEPA 2.1 ViT-L features + FLAN-T5-large language conditioning
- Dual-stream World / Action DiT with tripartite attention masks
- Flow-matching action generation (EMA eval, 10 Euler steps, 10-action chunks on LIBERO)
- Hydra configs for model sizes `s` / `b` / `l` and recipes `baseline` / `final` / ablations
- Feature precompute store, DDP training, and closed-loop LIBERO eval entry points

## Repository layout

```
├── lajepa/
│   ├── features/                 # Stage 1: Feature precomputation, fully decoupled from training
│   │   ├── vjepa_encoder.py      # Wrapper for frozen V-JEPA 2.1 (tube-motion slicing)
│   │   ├── precompute.py         # Batch-encode future targets to build feature bank
│   │   └── store.py              # Feature bank I/O (sharded + memmap, dual granularity support)
│   ├── models/
│   │   ├── encoders.py           # Observation encoder (LoRA wrapped), FLAN-T5, propriojection head
│   │   ├── world_dit.py          # World DiT with RoPE and pluggable token reduction layer
│   │   ├── action_dit.py         # Action DiT
│   │   ├── masking.py            # Tripartite attention mask (core correctness component)
│   │   └── lajepa.py             # Top-level model assembly & forward pass
│   ├── flow_matching.py          # Separate sampling for τ_f/τ_a, noise injection, velocity objective, dual MSE loss
│   ├── dropouts.py               # Conditional dropout for camera, proprioception and text
│   ├── augment.py                # Observation-only augmentation (0.3 bypass gate + independent transform gate)
│   ├── data/
│   │   ├── dataset.py            # Window sampling: 2 obs frames + 16 future frames + 32-action chunk
│   │   └── normalize.py          # Percentile statistics & feature normalization; stats saved with checkpoint
│   ├── train.py                  # DDP training loop with bf16, EMA and gradient checkpointing
│   └── inference.py              # Standalone inference pipeline: KV cache, τ_ctx=0, Euler sampler

```
The `lajepa/` folder contains the full implementation of LAJEPA. It adopts a two-stage paradigm: offline feature precomputation decoupled from the subsequent flow-matching training. It implements World DiT and Action DiT with tripartite attention masking, conditional dropout and observation augmentation. The dataloader samples fixed-length trajectory windows with percentile normalization. The training script supports DDP, bf16, EMA and gradient checkpointing; inference is implemented separately with KV cache and Euler sampling.

```
├── eval/
│   ├── libero_eval.py                # Closed-loop LIBERO / LIBERO-Plus evaluation
│   ├── libero_plus_instructions.py   # Instruction correction for polluted LIBERO-Plus variants
│   ├── robotwin_eval.py              # RoboTwin 2.0 evaluation adapter
│   └── latency.py                    # Policy-call latency benchmark (Appendix C)
```

The `eval/` directory contains evaluation pipelines for LIBERO-Plus and RoboTwin, together with a latency benchmark script. Latency is measured at batch size 1 with 20 warmup iterations; the reported value is the median of 100 runs to match the evaluation protocol in the paper.

```
├── scripts/
│   ├── prepare_libero.py             # Build LIBERO manifest from HDF5 demos
│   ├── precompute_features.py/.sh    # Offline V-JEPA feature bank
│   ├── compute_norm_stats.py         # Action / state / feature normalization stats
│   ├── train.py / train.sh           # Hydra + torchrun DDP training entry
│   ├── run_libero_s.sh               # One-shot prepare → precompute → train (model S)
│   └── eval.sh                       # Dispatch to libero / latency evaluators
```

The `scripts/` folder contains bash entry scripts to trigger feature precomputation, training and evaluation workflows conveniently.

## Checkpoints (download separately)

Weights are **not** included in this repo. See [`checkpoints/README.md`](checkpoints/README.md) for Chinese/English details.

| Checkpoint | Required? | Download |
|---|---|---|
| V-JEPA 2.1 ViT-L distilled (`vjepa2_1_vitl_dist_vitG_384.pt`, ~5.15 GB) | Yes (precompute + train) | https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt |
| FLAN-T5-large | Yes (auto via Hugging Face) | https://huggingface.co/google/flan-t5-large |
| Trained LA-JEPA-LI policies | Optional | Not released yet — train locally |

```bash
mkdir -p checkpoints
curl -fL https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt \
  -o checkpoints/vjepa2_1_vitl_dist_vitG_384.pt
```

## Requirements

- Python ≥ 3.10
- CUDA GPU(s); paper defaults assume 4× RTX 4090 for full runs
- Distilled **V-JEPA 2.1 ViT-L** weights (link above)
- LIBERO demonstration HDF5 (four suites)
- Optional: LIBERO / LIBERO-Plus / RoboTwin simulators for closed-loop eval

```bash
# Create an env, then install a CUDA build of PyTorch for your platform.
pip install -r requirements.txt
pip install -e .
```

Copy `.env.example` and set paths:

```bash
export VJEPA_CHECKPOINT=/path/to/vjepa2_1_vitl_dist_vitG_384.pt
export LIBERO_DATA=/path/to/libero_hdf5   # contains libero_spatial|object|goal|10
```

## Quick start (LIBERO, model S)

```bash
# 1) Build a frame-budgeted manifest from the four suites
python scripts/prepare_libero.py \
  --hdf5-root "$LIBERO_DATA" \
  --output data/libero \
  --max-frames 24000

# 2) Offline V-JEPA feature store (2×2 pooled for LIBERO)
python scripts/precompute_features.py \
  --manifest data/libero/manifest.json \
  --output data/libero/features \
  --checkpoint "$VJEPA_CHECKPOINT" \
  --pool-factor 2 \
  --batch-windows 4 \
  --device cuda

# 3) Action / state / feature normalization stats
python scripts/compute_norm_stats.py \
  --manifest data/libero/manifest.json \
  --feature-store data/libero/features \
  --output data/libero/stats.pt

# 4) Train (1 GPU example; global batch 128 via accumulation)
NPROC=1 bash scripts/train.sh model=s data=libero recipe=final batch_size=32 num_workers=8

# Or the all-in-one helper:
# bash scripts/run_libero_s.sh
```

Multi-GPU (paper-style 4-GPU DDP):

```bash
NPROC=4 bash scripts/train.sh model=s data=libero recipe=final
```

## Evaluation

`eval/` provides LIBERO-Plus / RoboTwin closed-loop runners and a latency benchmark. Latency uses **batch size 1**, **20 warmup** iterations, and reports the **median of 100** timed runs (paper Appendix C).

```bash
# Closed-loop LIBERO suite (requires libero installed)
bash scripts/eval.sh libero --checkpoint runs/lajepa/libero_s_final/final.pt --suite libero_spatial

# Denoising latency (batch=1, 20 warmup, median of 100)
bash scripts/eval.sh latency --size s --steps 10
```

Inference defaults match Appendix B: **EMA weights**, **noise scale 1.0** redrawn every replan, **10** denoising steps, **10**-action execution chunks.

> Full paper reporting (checkpoint grid at 30/60/90/120/150k, shared step across objectives, all 10,030 LIBERO-Plus episodes) is not automated yet; use the saved `step_*.pt` checkpoints and `eval/libero_eval.py` manually.

## Configs

| Key | Options |
|---|---|
| `model` | `s` (0.10B), `b` (0.34B), `l` (1.14B), `tiny` (tests) |
| `data` | `libero`, `libero_plus`, `robotwin` |
| `recipe` | `final`, `baseline`, `ablation_*` |

Override any Hydra field on the CLI, for example `model=b recipe=baseline learning_rate=5e-4`.

## Tests

```bash
python -m unittest tests.test_lajepa
```

## Citation

If you use this code, please cite the original La-JEPA paper:

```bibtex
@article{lajepa2026,
  title   = {La-JEPA: Compact Robot Policies with Latent World Modeling},
  journal = {arXiv preprint arXiv:2611.08156},
  year    = {2026}
}
```

And refer to this reproduction as **LA-JEPA-LI**.

## License

MIT — see [LICENSE](LICENSE).

Upstream assets (V-JEPA weights, LIBERO / LIBERO-Plus / RoboTwin, FLAN-T5) remain under their own licenses.
