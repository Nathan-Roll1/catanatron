# Quantization notes (C inference)

## Shipped in this repo

- **`export_nn.py --weight-format fp16`**: header field `version=2`, weight tensors as IEEE-754 half precision on disk. [`csrc/nn.c`](../csrc/nn.c) `nn_load` expands to **float32** in memory; compute is still FP32 (smaller file, same runtime math cost).
- **`export_nn.py --weight-format int8`**: header field `version=3`, symmetric int8 **per `nn_load` tensor block** (composite C structs such as `EdgeConvWeights` are one quantized blob + one scale so the file stays aligned with `read_tensor_block`). `nn_load` expands to fp32 and also packs a per-output-row int8 view for compute.
- **`CATAN_NN_COMPUTE=int8`**: experimental only. It uses NEON dot-product int8 matvecs for M=1 dense layers while keeping Accelerate fp32 SGEMM for batched GNN/spatial work. It was fast in microbenchmarks but caused bad game-level behavior in a 1v3 smoke test, so fp32 compute remains the default even for version-3 int8 storage files.
- **`CATAN_NN_INT8_BATCH=1`**: opt-in all-int8 batched FC path. On M2 this was slower than Accelerate SGEMM and is currently for experiments only.
- `policy_top_k` now calls `nn_policy_only`, and `policy_head` skips sub-action heads with no legal actions in the mask. This preserves logits for legal groups while avoiding unnecessary policy work.
- Best current quality/speed tradeoff: keep fp32 compute and use the safe structural optimizations (`policy_top_k` -> `nn_policy_only`, mask-gated policy heads, policy cache). Version-3 int8 storage is useful for file size, but int8 compute needs a stronger game-level quality gate before use.
- **Unstructured weight pruning** does not speed up the dense fp32 paths in `nn.c`. For real FLOPs reduction use a smaller compile-time geometry (see `MAC_INFERENCE_SMALL.md`) or future sparse/int8 kernels.

## Current M2 benchmark

- `human_bot/bench_nn_compute.py --reps 2000` on rebuilt `csrc/libnn.dylib` + `nn_weights_m2.bin`: fp32 ~224 us value-only / ~213 us policy-only / ~241 us full forward; hybrid int8 ~172 us value-only / ~180 us policy-only / ~190 us full forward, but int8 failed the single-game quality smoke test and should not be used for play.
- 360 generated states, fp32 vs hybrid int8: top legal policy move agreement 99.7%. Logit magnitudes drift more than top moves, so use move agreement and game eval as the gate.
- `CATAN_NN_INT8_BATCH=1` improved with the blocked kernel but was still slower (~306 us value-only / ~319 us policy-only / ~316 us full forward) and 96.4% top legal agreement in one run, so the fastest usable path keeps batched GNN GEMMs on Accelerate.

## Future work

- FP16 **compute** would need a separate half-weight packed view plus ARM FP16 matvec/GEMM kernels or a BNNS/Accelerate half GEMM path. Current `fp16` is storage-only.
- Full int8 GNN speedup needs a faster batched kernel than the current simple NEON-dot implementation.

Reference: `human_bot/policy_net.py` uses `torch.quantization.quantize_dynamic` for a **PyTorch** policy; that path is not `nn_weights_*.bin` + `libdeep`.
