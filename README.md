# radiance · DARTree

This is the `dartree` branch of a [radiance](https://codeberg.org/StillDeadcode/radiance) fork: the
C++/HIP inference server for AMD GPUs, with a set of speculative-decoding and serving
optimisations built and tuned on **Qwen3.8-Flash-Next** (IQR-V4) on two Radeon AI PRO R9700 cards
(gfx1201, 32 GB each).

It is a working branch, not a release: about 50 commits on top of upstream radiance, some opt-in,
some measured and kept as negative results. Several of its pieces have since landed upstream; the
rest live only here.

## What is in it

- **DARTree:** adaptive draft trees for the MTP / block drafters. Trees are grown best-first over
  the drafter's candidates and verified in one pass, with confidence pruning, lossless tree
  coupling, dynamic verify rows and a chain fast path. Kernel work covers GDN tree recurrence,
  QSA tree indexer and M-RoPE tree steps.
- **Coupled drafting:** a coupling draft law, block and traversal verification, n-gram leaves,
  and adaptive, load-aware draft depth.
- **Suffix decoding** (`--draft-suffix`): a per-request suffix selector beside the MTP chain, a
  global cache of earlier responses (`--draft-suffix-global-mib`), and an index built one prefill
  chunk at a time that is taken over across turns.
- **Prefix store and caches:** a persistent prefix store with LRFU scores, admission and a write
  budget, plus a tail snapshot inside the prefill step so a resume needs no extra step.
- **Scheduler:** `--decode-headroom` (decode rows get room on top of the prefill budget),
  `prefill-span-first`, one routed MoE pass over a spanned step (`prefill-span-pass: step`), and
  concurrent prefills taking the span one at a time.
- **Kernels (libr4d):** MoE prefill GEMMs (packed 128-row run map for gate_up, 64-row blocks for
  the w4nl64 down), a staged coded form of the two-rank fused all-reduce for large prefill
  messages, a prefill form of `hc_read`, and a tiled prefill form of the QSA sparse attention.

## Measured (Qwen3.8-Flash-Next IQR-V4, 2x R9700, TP2, BetterBench)

| | 210 W a card | 300 W a card |
|---|---|---|
| prefill 2K / 8K / 16K / 32K / 47K tokens | 6970 / 7648 / 7618 / 7506 / 7680 tok/s | 7763 / 8379 / 8336 / 8483 / 8401 tok/s |
| decode, weighted single stream | 257.5 tok/s | 268.1 tok/s |
| concurrency 1 / 2 / 4 / 8 clients | 240 / 322 / 430 / 555 tok/s | 249 / 345 / 452 / 583 tok/s |

Coupled drafting with traversal verify and adaptive depth, against the plain MTP chain (T 1.0):
decode +28.7 % (code +29.4 %, file_edit +28.9 %), concurrency +30 / +29 / +21 / +17 % at
1 / 2 / 4 / 8 clients, prefill +4–9 %.

## Example config

```yaml
server:
  host: 127.0.0.1
  port: 8011
models:
  flashnext:
    model: /srv/models/qwen3.8-next-flash-iqr-v4/qwen3.8-next-flash-iqr-v4.rad
    tp: 2
    max-model-len: 200000
    kv-cache-dtype: i8
    max-num-seqs: 8
    placement: expert_tiered
    host-pool-mib: 20480
    max-num-batched-tokens: 8192
    prefill-span: 4
    prefill-span-pass: step
    prefill-span-min: 0
    prefill-span-first: on
    prefix-cache-mib: 16384
    prefix-cache-ttl: 0
    busy-prefill-chunk: 1088
    prefix-resume: nearest
    num-speculative-tokens: 7
    draft-coupling: on
    draft-verify: traversal
    draft-depth: adaptive
    draft-depth-load: on
    draft-suffix: on
    draft-tree: off
```

The container is [`StillDeadcode/qwen3.8-next-flash-iqr-v4`](https://huggingface.co/StillDeadcode/qwen3.8-next-flash-iqr-v4).

## Build from source

You need Linux, git, a C++20 compiler, CMake 3.21+, Ninja and ROCm (`hipcc`).

```sh
git clone --branch dartree https://github.com/Code000-211/RadianceDARTree.git && cd RadianceDARTree
./build.sh                                   # configure, build for the cards present, run the fast tests
```

Set `RADIANCE_HOME=build/radiance_home` when running from the build tree. For everything else
(other models, Docker, the full option list) see the upstream README and [`docs/GUIDE.md`](docs/GUIDE.md).

## License

[Apache-2.0](LICENSE), copyright Deadcode and the radiance contributors. Keep [`NOTICE`](NOTICE)
with any copy or derivative. It credits the authors and lists the MIT-licensed code the tree
includes. The model weights are not covered: each keeps the license of the model it was converted
from.
