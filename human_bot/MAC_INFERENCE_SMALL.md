# Smaller HumanBotNet for faster C inference (structured shrink)

The C runtime in [`csrc/nn.c`](../csrc/nn.c) compiles **one** geometry via macros in [`csrc/nn.h`](../csrc/nn.h):

- `NN_GNN_HIDDEN` (64 = small, 80 = large M2-style)
- `NN_TRUNK_CH` (128 = small, 192 = large)

Export must match: train or distill a checkpoint with the same `SmallNetworkConfig` / `HumanBotNet` dimensions, then:

```bash
python -m human_bot.export_nn --checkpoint checkpoints/your_small.pt \
  --output csrc/nn_weights_small.bin
```

Rebuild native libs that link `nn.c`, e.g.:

```bash
# from repo root; example Darwin line (see human_bot/MODELS.md)
cc -shared -O3 -march=native -flto -fPIC -I csrc -DNN_GNN_HIDDEN=64 -DNN_TRUNK_CH=128 \
  -o csrc/libnn.dylib csrc/nn.c -lm -framework Accelerate
# libdeep: same -D flags as deploy_local_super_m2.sh deep_lib recipe
```

Then point `WEIGHTS=` / `--weights` at `nn_weights_small.bin`.

**Note:** This is the practical “pruning” path for this stack: **dense** Accelerate/NEON GEMMs and fused BN, not unstructured sparse weights.
