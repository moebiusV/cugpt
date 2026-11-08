# cugpt — C Translation of Karpathy's microgpt

Index: [All Projects](https://moebiusv.github.io/) | Git Repository: [github.com/moebiusV/cugpt](https://github.com/moebiusV/cugpt) | `git clone https://github.com/moebiusV/cugpt`

Two C implementations of Andrej Karpathy's [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95): a minimal GPT that trains a character-level language model on a names dataset and generates hallucinated names.

| File | Description |
|---|---|
| [`microgpt.py`](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) | Original pure-Python reference by Karpathy (benchmark baseline, not in repo) |
| `microgpt.c` | C translation — Wengert tape autograd, zero dependencies beyond libc/libm |
| `cugpt.c` | C translation — manual backward pass, OpenBLAS-accelerated BLAS |

## Quick Start

```sh
make
./microgpt    # 452× faster than Python
./cugpt       # 2476× faster than Python
```

Both download `input.txt` automatically on first run.
The original Python reference is at the [gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) (not included here).

## Model

1-layer GPT-2 variant · vocab=27 · n\_embd=16 · n\_head=4 · block\_size=16 · **4,192 params**
RMSNorm · ReLU · no biases · Adam (lr=0.01, β₁=0.85, β₂=0.99, linear decay) · 1,000 steps

## Benchmark

Machine: Ubuntu 22.04 / WSL2, gcc 14.2.0, Python 3.13.3 — 1,000 training steps on names dataset.

### Python baseline

| | Time |
|---|---|
| `python3 microgpt.py` | 146.11 s |

### microgpt.c — compiler flag sweep

| Flags | Time (s) | Speedup vs Python |
|---|---|---|
| `-O0` | 1.592 | 92× |
| `-O1` | 0.760 | 192× |
| `-O2` | 0.718 | 203× |
| `-O3` | 0.729 | 200× |
| `-O2 -march=native` | 0.588 | 248× |
| `-O3 -march=native` | 0.637 | 229× |
| `-O3 -march=native -ffast-math` | 0.337 | 434× |
| `-O3 -march=native -ffast-math -funroll-loops` | **0.323** | **452×** ← default |
| `-O3 -march=native -ffast-math -flto` | 0.336 | 435× |
| `-O3 -march=native -ffast-math -funroll-loops -flto` | 0.332 | 440× |

### cugpt.c — compiler flag sweep

| Flags | Time (s) | Speedup vs Python |
|---|---|---|
| `-O0` | 0.189 | 773× |
| `-O1` | 0.090 | 1623× |
| `-O2` | 0.078 | 1873× |
| `-O3` | 0.073 | 2001× |
| `-O2 -march=native` | 0.077 | 1897× |
| `-O3 -march=native` | 0.082 | 1782× |
| `-O3 -march=native -ffast-math` | 0.061 | 2395× |
| `-O3 -march=native -ffast-math -funroll-loops` | 0.078 | 1873× |
| `-O3 -march=native -ffast-math -flto` | **0.059** | **2476×** ← default |
| `-O3 -march=native -ffast-math -funroll-loops -flto` | 0.064 | 2283× |

### Summary

| Implementation | Best time | Speedup vs Python | vs microgpt.c |
|---|---|---|---|
| `microgpt.py` (Python) | 146.11 s | 1× | — |
| `microgpt.c` (tape autograd) | 0.323 s | 452× | 1× |
| `cugpt.c` (BLAS + manual grad) | 0.059 s | 2476× | **5.5×** |

### Key observations

- **`-ffast-math` is the biggest single-flag win (~2×)** — it lets the compiler reassociate FP ops and use faster math approximations. Safe here since we only need gradient descent, not bit-exact reproducibility.
- **`-funroll-loops` helps `microgpt.c`** (tight loops over tape entries) but *hurts* `cugpt.c` (BLAS already handles inner loops).
- **`-flto` wins for `cugpt.c`** by inlining the thin BLAS wrapper functions across translation units.
- **Even unoptimized `cugpt.c -O0` (0.189 s) beats the best `microgpt.c` build (0.323 s)** — eliminating the autograd tape is an algorithmic win, not a compiler win.

## Implementation Notes

**`microgpt.c`** translates the Python scalar-`Value` autograd graph to a flat Wengert tape (a pre-allocated 4M-entry arena, reset each step). Backward is a single reverse scan — no pointer chasing, no heap allocation, no recursive topo-sort.

**`cugpt.c`** discards the tape entirely. Forward activations are saved in a struct array (`Activations act[BLOCK_SIZE]`). The backward pass is written analytically. Every matrix-vector product is dispatched through `cblas_dgemv`; weight gradient accumulation uses `cblas_dger` (rank-1 outer-product update).

## Build

```sh
./configure && make        # detects compiler, OpenBLAS, cblas.h
make check                 # smoke test (10 steps each)
make bench                 # full flag sweep + Python baseline
make install               # to /usr/local (or ./configure --prefix=~/.local)
make dist                  # creates cugpt-1.0.tar.gz
```

**Dependencies:**
- `microgpt.c` — C compiler + libm (always present)
- `cugpt.c` — C compiler + libopenblas-dev (`apt install libopenblas-dev`)

See [INSTALL](INSTALL) for manual build instructions.

## License

MIT — see [COPYING](COPYING).
Original `microgpt.py` by Andrej Karpathy, shared under the MIT convention used across his educational ML projects ([micrograd](https://github.com/karpathy/micrograd), [makemore](https://github.com/karpathy/makemore)).
