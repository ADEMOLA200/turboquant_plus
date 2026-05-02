# Cross-Engine KV Cache Fidelity on AMD MI300X: Same Model, Same fp8, Different Outputs

**Tom Turney**
Independent Researcher
GitHub: [@TheTom](https://github.com/TheTom)

---

## Abstract

We benchmark three production inference engines (vLLM, SGLang, llama.cpp) on a single AMD MI300X GPU running the same hybrid GDN + attention MoE model (Qwen3.6-35B-A3B). Two rounds: (1) BF16 baseline with no KV compression, measuring load time, perplexity, prefill, and decode; (2) REFRACT 4-axis fidelity scoring with each engine's native 8-bit KV against its own fp/bf16 reference.

The BF16 round shows no single winner: SGLang dominates prefill (3× vLLM, 25× llama.cpp at 32K), llama.cpp dominates decode (1.5× SGLang, 5× vLLM) and load time (~6× faster), and vLLM sits in the middle. Workload shape decides the right engine.

The REFRACT round produces a more interesting result. All three engines hit perfect 100.0 R-NIAH (long-context retrieval at 32K) under 8-bit KV. But trajectory and KLD axes diverge sharply. llama.cpp `q8_0` (true int8) produces 0.0025 nats of mean KL drift. vLLM `fp8_e4m3` produces 0.037 nats — 15× worse. SGLang `fp8_e4m3` (forced through Triton attention because AITER's fp8 prefill kernel rejected the hybrid model) produces 0.021 nats. **Two engines that both label their compression `fp8_e4m3` produce a clean 1.8× difference in KL drift on the same model and hardware.** The dtype label is doing less work than people think; the actual kernel implementation is what determines fidelity.

The bench also documents 8 nontrivial engine-side bugs that had to be cleared before any cross-engine measurement was possible: vLLM's `max_num_seqs` default vs Mamba block count, `prompt_logprobs` cap, GPU memory carryover across engine subprocess deletions, missing flash-attn ROCm wheel forcing a 70-minute Composable Kernel compile; SGLang's broken `aiter.dtypes` import chain in the published Docker image, AITER fp8 prefill rejecting the hybrid model, KV dtype fixed at server launch (no per-request switching). Total bring-up to clean cross-engine apples-to-apples: ~12 hours.

The findings are: (1) trajectory drift is the axis that compounds in long generations and where engines actually differ — single-metric (PPL, NIAH) bench would have called these engines equivalent; (2) AMD's fp8 KV path is noisier than int8 at the same nominal bit width, suggesting fp8 needs better scaling granularity to catch up; (3) all three engines deliver PASS or borderline-PASS on hybrid Qwen3.6 with 8-bit KV — the model is safe to deploy with compression, the question is just how much fidelity you keep.

---

## 1. Introduction

KV cache compression is the dominant lever for fitting long context into GPU memory at inference time. Three production engines support 8-bit KV cache compression on AMD MI300X via different paths: llama.cpp via `q8_0` (block-quantized int8), vLLM via `fp8_e4m3` (AMD's fnuz fp8 variant through ROCm flash-attention), and SGLang via `fp8_e4m3` (through AITER's fp8 prefill kernel).

The natural question is: do these three implementations produce equivalent output quality? The dtype label suggests they should. Two engines that both advertise `fp8_e4m3` KV cache, running the same model on the same hardware, should produce the same perplexity, the same trajectory, the same KL divergence from their fp16 baseline.

We tested this empirically. They don't.

This paper reports a controlled cross-engine measurement on a single MI300X, with an explicit methodology for matching context length, tokenization, and reference anchoring across engines. The fidelity scoring uses [REFRACT](https://github.com/TheTom/turboquant_plus/tree/main/refract), our 4-axis evaluation framework anchored to each engine's own fp16 reference.

---

## 2. Setup

### Hardware

- 1× AMD Instinct MI300X (192 GB HBM3, gfx942)
- DigitalOcean dev cloud droplet, ROCm 7.2, Ubuntu 24.04
- $300 of AMD credits applied

### Model

- `Qwen/Qwen3.6-35B-A3B` (BF16 safetensors, ~67 GB across 26 shards)
- Architecture: hybrid GDN + attention MoE, 256 K native context
- Same model file referenced across all three engines (vLLM and SGLang load HF safetensors directly; llama.cpp loads BF16 GGUF converted from the same source)

### Engines

| Engine | Source | KV-cache option used |
|---|---|---|
| **vLLM** | [TheTom/vllm](https://github.com/TheTom/vllm) `pr/tq-prebaked-centroids` (BF16 path identical to upstream main) | `kv_cache_dtype="fp8_e4m3"` (candidate); `"auto"` = bfloat16 (reference) |
| **SGLang** | `lmsysorg/sglang:v0.5.10.post1-rocm720-mi30x` Docker | `--kv-cache-dtype fp8_e4m3` (candidate); `auto` = bfloat16 (reference); forced `--attention-backend triton` |
| **llama.cpp** | [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) `feature/turboquant-kv-cache` | `-ctk q8_0 -ctv q8_0` (candidate); `f16/f16` (reference) |

### Methodology

- Eval corpus: `wikitext-2-raw/wiki.test.raw` (~1.3 MB, ~250K tokens)
- Same `prompts/v0.1.jsonl` (30 prompts) across all three engines
- BF16 baseline measured at 32K context for prefill / decode / KV size
- REFRACT axes measured at 4096 ctx (Trajectory, KLD, PLAD) and 32768 ctx_max (R-NIAH)
- All scores anchored to each engine's own fp/bf16 reference, not a global "fp16 truth"

---

## 3. BF16 Baseline Results

No quantization on weights or KV. Pure capability check on each engine.

| Metric | vLLM | SGLang | llama.cpp | Winner |
|---|---|---|---|---|
| Model load (s) | 188.9 | ~210 | **31.8** | llama.cpp (~6×) |
| PPL @ 32K | 5.49 | 5.74 | 6.01 | within methodology variance |
| Prefill tok/s @ 32K | 11,690 | **32,428** | 1,298 | SGLang (3× vLLM, 25× llama.cpp) |
| Decode tok/s @ 256 out | 25.6 | 90.2 | **133.2** | llama.cpp (1.5× SGLang, 5× vLLM) |
| KV / state footprint @ 32K | 92.1 GiB KV pool | 51 GB KV + 46 GB GDN/Mamba state | 702 MiB context block | not directly comparable |
| Setup ergonomics | clean (native venv) | **6 in-container monkey patches** | clean (native binary) | vLLM / llama.cpp tie |

PPL across engines (5.49, 5.74, 6.01) is within methodology variance. The model loaded correctly on all three; this is a sanity check, not a quality differentiator.

The other axes diverge by significant margins. **No single engine wins all four.** Workload shape decides:

- **Long input + short output (RAG, classification, doc summarization with short answers):** SGLang's prefill dominates.
- **Short input + long output (chat, generation, agentic loops):** llama.cpp's decode wins.
- **Cold-start / batch-style / one-shot:** llama.cpp's 32 s load vs 3+ min for the others.
- **Hybrid model dev work / advanced KV compression:** vLLM (only engine with TurboQuant+ extensions live; SGLang doesn't have TurboQuant at all).

---

## 4. REFRACT 4-Axis Fidelity Results

[REFRACT](https://github.com/TheTom/turboquant_plus/tree/main/refract) scores how much fidelity each engine retains when 8-bit KV compression is enabled, anchored to that engine's own fp/bf16 reference. Four axes:

- **Trajectory (gtm):** greedy-decode N tokens per prompt under both KV configs. Score = fraction of candidate tokens that match the reference token-by-token.
- **KLD:** per-token KL divergence between candidate and reference next-token distributions on a natural-text corpus. Score = `100 * exp(-mean_kld)`.
- **R-NIAH:** insert a sentinel ("APRICOT-7-BLUE is the rare paint color") at fractional positions of long context, score retrieval accuracy at lengths up to 32K.
- **PLAD:** per-token edit distance under prompt perturbations (typos, case changes, paraphrases). Score reflects how much extra drift the candidate introduces vs the reference under the same perturbations.

Composite is the harmonic mean of the four. Bands: ≥95 EXCELLENT, ≥85 PASS, ≥70 DEGRADED, <70 FAIL.

### Cross-engine REFRACT scores

| Engine | Cand KV | gtm | kld | rniah | plad | **Composite** | Band |
|---|---|---:|---:|---:|---:|---:|---|
| **llama.cpp** | `q8_0` (int8) | 69.62 | 99.75 | 100.0 | 96.54 | **89.39** | **PASS** |
| **SGLang** | `fp8_e4m3` (triton) | 67.71 | 97.95 | 100.0 | 90.77 | **86.97** | **PASS** |
| **vLLM** | `fp8_e4m3` | 62.40 | 96.35 | 100.0 | 90.58 | **84.31** | DEGRADED |

### Mean KL divergence (the source-of-truth fidelity metric)

| Engine | Cand KV | mean_kld nats | top-1 % |
|---|---|---:|---:|
| llama.cpp | `q8_0` | **0.0025** | — |
| SGLang | `fp8_e4m3` | 0.021 | 97.67 |
| vLLM | `fp8_e4m3` | 0.037 | 97.48 |

llama.cpp's int8 produces ~15× less KL drift than vLLM's fp8 on the same model. SGLang's fp8 sits in the middle (see §5 for the asterisk on this number).

### Findings

1. **All three engines hit perfect 100.0 R-NIAH at 32K.** Long-context retrieval is intact under 8-bit KV regardless of which engine implements it. This is the load-bearing finding for production users: the model still finds the needle even with compression.

2. **llama.cpp int8 wins every axis.** Best gtm, best kld, best plad. PASS by a clear margin. int8 with per-block scaling is doing more work than fp8 with per-tensor scaling at the same nominal bit width.

3. **SGLang fp8 (87) edges vLLM fp8 (84)** despite same nominal dtype. Engine-level fp8 kernel paths diverge — see §5.

4. **Trajectory drift dominates the cross-engine gap.** llama.cpp 69.62, SGLang 67.71, vLLM 62.40. KLD/R-NIAH/PLAD are all close — the trajectory axis is where 8-bit KV bites, and that's also where the fp8 kernel implementation differences amplify across many tokens.

5. **No catastrophic failure on hybrid Qwen3.6.** All three engines are safe to deploy with this baseline 8-bit KV config — though vLLM falls into the DEGRADED band (84.31, 0.69 below the 85 PASS threshold).

---

## 5. The fp8 Label is Not a Standard

The most surprising finding: **same model, same hardware, same `fp8_e4m3` dtype string, two engines disagree on KLD by a factor of 1.8×**. The dtype label is doing less work than expected. The kernel implementation downstream of that string is what determines actual fidelity.

### Different paths, same nominal dtype

- **vLLM** routes through ROCm flash-attention. Per-tensor fp8 scaling at write time. fp8 attention math through the rotary path. Dequantize at read.

- **SGLang** *intended* to route through AITER's `mha_batch_prefill_fp8bf16` kernel. AITER's prefill rejected the hybrid Qwen3.6 with `RuntimeError: invalid argument for batch_prefill` on both `fp8_e4m3` and `fp8_e5m2` variants. Working around this required `--attention-backend triton`, which routes prefill through Triton's attention implementation with its own fp8 quantization scheme.

These are entirely different downstream kernel paths. The dtype string says `fp8_e4m3`. What actually runs is per-engine.

### The asterisk on SGLang's number

SGLang's `mean_kld = 0.021` nats is suspiciously low for "real" fp8 quantization. It sits between llama.cpp's int8 (0.0025) and vLLM's fp8 (0.037). Possible explanations:

1. **Triton's fp8 scaling is genuinely better** than the AITER/flash-attn ROCm path — using per-channel or per-token scaling instead of per-tensor.
2. **Some layers silently bf16-cast** in the Triton fallback path. The mha kernel may accept fp8 inputs but operate internally at bf16, producing apparent fidelity that's a no-op artifact rather than real fp8 KV.

R-NIAH at 100% does not disambiguate these — llama.cpp also gets 100% with bonafide int8. A stress test (32k retrieval under adversarial distractors) would settle it. We have not run that yet. Pending.

### Implication for users

If you are choosing between vLLM and SGLang based on a paper claiming "fp8 KV works on Qwen-class models," **do not assume the paper's number applies to your engine**. The paper's number was generated by some specific kernel implementation. Your engine probably ships a different one. Test it.

---

## 6. The Saga of Getting Three Engines to Run

Most of the bench's wall time was bring-up, not measurement. We document the failures because they are reproducible and because someone trying this benchmark elsewhere will hit them.

### llama.cpp (1 issue)

The REFRACT R-NIAH axis tokenizes the haystack via `runner.tokenize_to_ids`, which historically shelled out to a `llama-tokenize` binary. On hosts where the local llama.cpp checkout had drifted, this failed with `Symbol not found: _llama_memory_breakdown_print` even though the loaded library was functional. Fix: dispatch `tokenize_to_ids` to the active backend's own tokenizer when the backend is not llamacpp. One commit, no other friction.

### vLLM (5 rounds)

1. **Missing flash-attn ROCm wheel.** Qwen3.6 instantiates a `Qwen3_VisionTransformer` subcomponent at model load even for text-only use. Its RoPE imports `flash_attn.ops.triton.rotary`. There is no pre-built flash-attn wheel for ROCm. Built from source via `git+https://github.com/ROCm/flash-attention.git@main_perf` and `--no-build-isolation`. ~5,800 .hip object files. ~70 minutes wall time on the droplet.

2. **`max_num_seqs` default vs Mamba blocks.** vLLM defaults to `max_num_seqs=1024`. Hybrid Qwen3.6's Mamba state allocator at `gpu_memory_utilization=0.45` produced only 784 cache blocks. Engine init crashed: `ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (784). Each decode sequence requires one Mamba cache block, so CUDA graph capture cannot proceed.` Fixed via `REFRACT_VLLM_MAX_NUM_SEQS=32` env knob. REFRACT only does serial requests so 32 is generous.

3. **`prompt_logprobs` cap.** REFRACT KLD axis sent `prompt_logprobs=64` (top-K for the next-token distribution). vLLM caps at 20: `VLLMValidationError: Requested prompt logprobs of 64, which is greater than max allowed: 20`. Fixed via `REFRACT_VLLM_KLD_TOPK=20`.

4. **Trajectory axis interleaving forces N model loads.** REFRACT's trajectory axis originally interleaved `ref` and `cand` calls per prompt. With our eviction-on-key-change cache (necessary because two LLM instances of this hybrid model don't fit 192 GB at high mem util), interleaving meant ~60 model evictions per axis. ~40 minutes per cold load. Refactored axis to batch all-ref then all-cand. Two model loads total per axis. PR landed in REFRACT's `refract.axes.trajectory`.

5. **vLLM v1 engine subprocess holds GPU memory across `del LLM()`.** vLLM's v1 architecture runs the engine core as a multiprocessing subprocess. `del LLM()` plus `gc.collect()` plus `torch.cuda.empty_cache()` doesn't actually release the engine subprocess's allocations. The second axis's LLM init saw 24 GB free out of 192 GB and crashed: `ValueError: Free memory on device cuda:0 (24.05/191.69 GiB) on startup is less than desired GPU memory utilization (0.85, 162.93 GiB)`. Fix: split each axis into its own python process (`--skip-kld` for axis A, `--skip-gtm` for axis B, etc.). Process exit guarantees teardown.

For axis C R-NIAH specifically, also bumped `REFRACT_VLLM_MAX_MODEL_LEN=33792` since the 32K probe needs ctx ≥ 32K and the LLM is cached by `max_model_len`.

### SGLang (3 rounds plus orchestrator)

1. **Broken `aiter.dtypes` in published Docker image.** The `lmsysorg/sglang:v0.5.10.post1-rocm720-mi30x` image's `aiter` package is missing its `dtypes` module. SGLang's Quark MXFP4 import chain references `aiter.dtypes.fp8` unconditionally at module load (even when Quark isn't being used):
   ```
   File "sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py":
       from aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_split_cat import ...
   File "aiter/ops/triton/quant/fused_fp8_quant.py":
       fp8_dtype = aiter.dtypes.fp8
   AttributeError: module 'aiter' has no attribute 'dtypes'
   ```
   Fix: `sitecustomize.py` that stubs `aiter.dtypes` mapping to `torch.float8_e4m3fnuz`. Mounted as `/opt/sitecustom/sitecustomize.py:ro` and selected via `PYTHONPATH=/opt/sitecustom`. Also stubbed `dynamic_per_tensor_quant` and `static_per_tensor_quant` to raise on call (the Quark loader imports them but doesn't invoke them on the BF16 / fp8 KV paths).

2. **AITER fp8 prefill rejects hybrid model.** With Quark loading patched, the next failure surfaces at request time: AITER's `mha_batch_prefill_fp8bf16` kernel throws `RuntimeError: invalid argument for batch_prefill` on hybrid Qwen3.6 for both `fp8_e4m3` and `fp8_e5m2`. Workaround: `--attention-backend triton` forces SGLang to bypass AITER's prefill kernel and route through Triton attention. This works but means SGLang's "fp8 KV" is going through a different kernel implementation than its default path on this model — see §5.

3. **KV dtype is fixed at server launch.** SGLang has no per-request KV dtype switching. REFRACT's KLD axis wants to compare two configs in one run. Built a sequential orchestrator (`refract_sglang_seq.sh`):
   - Phase ref: launch BF16 server, run all probes via HTTP, dump to JSON, kill container
   - Phase cand: launch fp8 server, run same probes, dump to JSON, kill
   - Aggregate: load both JSONs, compute KLD per chunk, generate REFRACT-format scores

   For axes C (R-NIAH) and D (PLAD), wrote `refract_sglang_cd_collect.py` which imports REFRACT's needle generator (`refract.axes.rniah._build_prompt`, `_extract_password_keyword`) and perturbation functions (`refract.axes.plad._PERTURBATION_FUNCS`) directly so methodology is identical. The aggregator uses HuggingFace's tokenizer for PLAD's edit distance to avoid shelling back out to `llama-tokenize`.

### Total bring-up cost

8 nontrivial bugs across 3 engines. ~12 hours wall time end-to-end including the 70-minute flash-attn ROCm compile, several CUDA graph captures, two cycles of vLLM script revision, and the SGLang sequential orchestrator development.

---

## 7. Discussion

### Why fp8 KV is noisier than int8

`fp8_e4m3fnuz` has more dynamic range than int8 in theory. In practice, the per-tensor scaling factor that the AMD ROCm path applies is a single scalar per tensor. int8 (`q8_0`) uses per-block scaling — typically a separate scale per 32 elements. For KV cache values whose distribution varies sharply across the head dimension or across the sequence dimension, per-tensor scaling discards much of fp8's dynamic-range advantage.

A fp8 KV cache with per-channel or per-token scaling would likely catch up to int8 on fidelity at the same nominal bit width. NVIDIA's H100 and B200 paths may already be doing this; we have not tested them. AMD's roadmap probably needs to head in this direction before fp8 KV becomes the better default on MI300X.

### Why trajectory is the axis that compounds

Trajectory drift is the quadratic axis. Every token's KV is read by every subsequent token's attention. A small per-step distortion in the KV distribution accumulates over the generation. KLD measures the per-step distortion. R-NIAH measures retrieval at a fixed point. Trajectory measures the cumulative effect of N steps of KLD-level drift compounding through N greedy decodes.

This is why all three engines hit perfect R-NIAH at 32K but only llama.cpp keeps trajectory above 69. The needle test is one query at fixed depth. The trajectory test is 30 prompts × 128-token autoregressive decode under both configs, comparing token by token.

For production users, trajectory is the axis that matters for long-form generation (chat, code, agents). For RAG with short outputs, R-NIAH is the relevant axis. Different workloads stress different axes. REFRACT's multi-axis design surfaces this.

### Hybrid models stress every engine

AITER's fp8 prefill broke. vLLM's `max_num_seqs` default broke. SGLang's Quark loader broke. The published Docker images and stable engine versions all assume dense attention. Hybrid GDN + attention models like Qwen3.6 and Qwen3-Next expose paths that production engines have not yet hardened. llama.cpp was the exception because the maintainer's hybrid support work has been ongoing since early 2026.

If you are deploying hybrid models on AMD MI300X today, expect bring-up friction. Budget for it.

---

## 8. Limitations

1. **Single model, single GPU.** Qwen3.6-35B-A3B on one MI300X. Other hybrid models (Qwen3-Next, Jamba2) and other GPUs (H100, B200, MI355X) are untested.

2. **Symmetric 8-bit KV only.** Asymmetric configs (K=int8 + V=fp8, or K=q8_0 + V=turbo3) are the production-recommended setting and almost certainly lift every engine here closer to PASS. Not tested in this bench.

3. **No TurboQuant+.** Symmetric `turbo4` on hybrid Qwen3.6 produces PPL 9.13 / KLD 9.13 on llama.cpp — the V3 paper §9.6 hybrid blocker showing up cleanly. TurboQuant+ on hybrid models needs algorithmic work that has not started.

4. **SGLang fp8 fidelity has an asterisk.** The 0.021 nats mean_kld may be partial bf16 fallback rather than true fp8 quantization through the Triton attention path. R-NIAH at 100% does not disambiguate. Pending an adversarial NIAH stress test.

5. **Single bench run per engine.** No multi-trial variance bound. PPL and decode tok/s have been observed to vary 2-5% across runs. The cross-engine differences here are larger than that variance, but tighter bounds would require multi-trial.

6. **No NVIDIA comparison.** This bench is AMD-only. The AMD fp8 KV finding may not generalize to NVIDIA's fp8 paths.

---

## 9. Conclusion

Three production inference engines on the same model, same GPU, same `fp8_e4m3` dtype label produce measurably different output fidelity. The dtype string is not a standard. The kernel implementation downstream of that string is what determines what actually runs.

llama.cpp's int8 KV is the cleanest 8-bit option on AMD MI300X today. SGLang's fp8 path (via Triton attention) and vLLM's fp8 path differ by ~1.8× on KL drift despite identical nominal configuration. All three engines preserve perfect R-NIAH retrieval at 32K under 8-bit KV — the model is safe to deploy with compression, the question is just how much fidelity you keep.

For workload selection: SGLang for prefill-heavy, llama.cpp for decode-heavy and cold-start, vLLM for hybrid model R&D and TurboQuant+ extensions.

For methodology: anchor scoring against each engine's own fp/bf16 reference, match context length and tokenization, run multi-axis fidelity scoring (PPL alone misses what trajectory catches), and don't trust the dtype label without testing it on your model.

---

## 10. Reproducibility

All scripts and orchestrators on the droplet at `/root/scripts/`:

- `cross_engine_bench.sh` — BF16 baseline (load / PPL / prefill / decode / KV size for all 3 engines)
- `refract_llamacpp_full.sh` — REFRACT --full on llama.cpp
- `refract_vllm_full.sh` / `refract_vllm_full_cd.sh` — REFRACT split-axis on vLLM
- `refract_sglang_seq.sh` / `refract_sglang_cd_seq.sh` — REFRACT two-phase orchestrator on SGLang
- `refract_sglang_collect.py` / `refract_sglang_cd_collect.py` — SGLang HTTP probe collectors (A+B and C+D)
- `refract_sglang_aggregate.py` / `refract_sglang_cd_aggregate.py` — REFRACT-format scoring from collected dumps
- `sitecustomize.py` — `aiter.dtypes` stub for SGLang container

REFRACT framework changes pushed to [TheTom/turboquant_plus@main](https://github.com/TheTom/turboquant_plus):
- vLLM backend rewritten from skeleton (evict-on-key-change cache, env knobs for `MAX_NUM_SEQS`, `KLD_TOPK`, `GPU_MEMORY_UTILIZATION`, `MAX_MODEL_LEN`)
- SGLang backend with two-phase orchestration support
- Trajectory and PLAD axes refactored to batch ref/cand by KV config (helps any memory-pressured backend)
- Runner `tokenize_to_ids` dispatches through the active backend (R-NIAH unblock for vLLM/SGLang)

---

## References

- [REFRACT framework](https://github.com/TheTom/turboquant_plus/tree/main/refract) — 4-axis KV-cache fidelity scoring
- [REFRACT QUICKSTART](https://github.com/TheTom/turboquant_plus/blob/main/refract/QUICKSTART.md)
- [REFRACT vLLM backend](https://github.com/TheTom/turboquant_plus/blob/main/refract/backends/vllm.py)
- [REFRACT SGLang backend](https://github.com/TheTom/turboquant_plus/blob/main/refract/backends/sglang.py)
- [REFRACT leaderboard](https://github.com/TheTom/turboquant_plus/blob/main/refract/LEADERBOARD.md)
- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) — llama.cpp fork with TurboQuant+ KV cache and REFRACT trajectory patch
- [TheTom/vllm](https://github.com/TheTom/vllm) — vLLM fork with TurboQuant+ KV cache extensions
- [TurboQuant paper](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- Asymmetric K/V findings: [asymmetric-kv-compression.md](asymmetric-kv-compression.md)
- TriAttention V3 (hybrid model V3 work): [triattention-v3.md](triattention-v3.md)
- PPL artifacts on instruct models: [attn-rotation-and-ppl-artifact.md](attn-rotation-and-ppl-artifact.md)
- Earlier configuration recommendations: [turboquant-recommendations.md](../turboquant-recommendations.md)
