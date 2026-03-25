# Speed Experiments Log

Branch: `experiment/speed-optimization` (both repos)
Goal: prefill speed closer to q8_0 (currently 1074 vs 2694 tok/s) while PPL stays at 6.19 +/- 0.1

## Baseline (before experiments)

| Config | Prefill tok/s | PPL | Notes |
|--------|-------------|-----|-------|
| q8_0 | 2694 | 5.41 | target |
| turbo3 fp16 WHT | 1074 | 5.47 | current top-of-tree (32 chunks) |
| turbo3 fp16 WHT | — | 6.195 | 8-chunk PPL reference |
| turbo3 no rotation | 1577 | — | speed ceiling (wrong quality) |

## Experiment 1: Vectorized half4 WHT + packed centroid lookup

**Hypothesis:** The WHT butterfly and centroid unpacking can be vectorized with half4 operations for 4x wider SIMD throughput. Also optimizes memory access patterns for qs/signs bytes.

**Changes:**
- `turbo_fwht_128_half4()`: WHT butterfly on 32 x half4 vectors instead of 128 x half scalars
  - h=1,2: intra-vector swizzle (no loop over pairs)
  - h=4..64: inter-vector butterfly with computed stride
- Centroid lookup: process 4 elements per qs byte (natural byte boundary)
- Sign application: vectorized half4 multiply
- Final conversion: float4 output with fused norm scale

**Results:**

| Config | Prefill tok/s | PPL (32-chunk) | PPL (8-chunk) |
|--------|-------------|----------------|---------------|
| Baseline (scalar fp16 WHT) | 1074 | 5.47 | 6.195 |
| **half4 vectorized WHT** | **1411** | **5.47** | **6.195** |
| q8_0 | 2694 | 5.41 | — |

**+31% speedup, PPL unchanged.** Gap to q8_0: 1.91x (was 2.51x).

**Codex review:** No correctness bugs found. Butterfly pairing, centroid unpacking, and sign application all verified correct.

**Status:** COMPLETE — committed

---

## Experiment 2: Reduced centroid lookup overhead

**Hypothesis:** The 3-bit index unpacking does 3 loads + 2 shifts + 1 OR per element. Pre-combining indices during quantize into a single packed array would reduce dequant to 1 load + mask.

**Status:** PENDING

---

## Experiment 3: RoPE-aware pre-rotate-queries

**Hypothesis:** The earlier pre-rotate-queries failed because WHT and RoPE don't commute. Fix: apply WHT immediately AFTER RoPE in the model code (not in build_attn_mha). This eliminates the WHT from dequant entirely.

**Status:** PENDING
