# LM_Dec_Base_LaMDA

LaMDA - Base family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Smoke-Tested**
- Failure class: Hardware Infeasible: T5 relative bias blocks flash attention (12.3 GB at length 2048); 17M of 200M tokens

## Provenance

- Source repository: LM_2022_LaMDA-pytorch
- Upstream / license: vendored from RichardMinsooGo/LM_2022_LaMDA-pytorch (see LICENSE in vendor folder if present)

## Differences from the source ([Unified] protocol)

- T5 relative position bias (upstream)

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- Parameters: 22.39 M
- Causality check: pass (largest change of earlier outputs 0.0e+00)
- Overfit check (4 x 256 bytes): 5.762 -> 0.001 nats, pass
- LM-02 enwik8 (200M tokens): valid 1.946 / test 1.934 bits per byte (time-capped)
- Peak GPU memory at length 2048 (batch 1): train 12307 MB, inference 3698 MB
- Peak GPU memory at length 8192 (batch 1): train None, inference None

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy
python run.py check lamda
python run.py lm02 lamda      # enwik8 is downloaded to data/ on first use
```
