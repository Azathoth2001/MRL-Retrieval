# MRL-Retrieval

**Matryoshka Representation Learning (MRL) for CLIP image–text retrieval.** A 1.05 M-parameter
residual adapter is trained with a **nested InfoNCE** loss so that *prefixes* of the embedding are
themselves usable retrieval representations. You can then truncate to whatever dimension your
latency and storage budget allows — without training a separate model per target dimension.

**At 64 dimensions the model keeps 95% of full-dimension recall while cutting single-query latency by
94%. Cascade retrieval reaches a 22× speed-up while preserving ≥99% of exact full-dimension recall.**

Full method, all result tables and figures: **[RESULTS.md](RESULTS.md)**.

![recall vs dim](outputs/figures/r1_vs_dim.png)

---

## Method

For each image–text pair the adapter emits a 512-d vector `e`. For every level
`m ∈ {32, 64, 128, 256, 512}` we take the **prefix `e[:m]` first, then L2-normalise**, compute a
symmetric InfoNCE, and sum the terms with weights `c_m`:

```
L = Σ_m  c_m · InfoNCE( normalize(e_img[:m]), normalize(e_txt[:m]), τ )
```

> ⚠️ **The order matters.** Normalising the full vector and *then* slicing yields prefixes with
> norm < 1 and silently corrupts recall. `src/metrics.py` asserts unit norm to catch exactly this.

Four conditions, compared on the same recall-vs-dimension curve:

| | Name | What it does | Trained |
| --- | --- | --- | --- |
| ① | Truncate | native CLIP embedding cut to *m* | no |
| ② | PCA | PCA fitted on train (one projection shared by both modalities) | no |
| ③ | Adapter-plain | adapter trained with InfoNCE **at full dimension only**, then truncated | yes |
| ④ | **Adapter-MRL** | adapter trained with the **nested** loss | yes |

③ exists to isolate the contribution of *nesting*. ③ and ④ share architecture, parameter count,
data, epochs, learning rate, seed, temperature mode and best-checkpoint criterion — **the only
difference is whether the loss has one term or five nested terms.** At m = 32, ④ beats ③ by
**+24.7 (t2i) / +26.1 (i2t)** points.

---

## Environment

Requires a CUDA-capable GPU. The whole pipeline, including the Phase-2 backbone fine-tuning, fits in
**8 GB of VRAM**.

```bash
conda create -n mrl-retrieval python=3.12 -y
conda activate mrl-retrieval
# Install torch from the official index matching your CUDA version; cu124 shown as an example
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python scripts/00_check_env.py          # verifies CUDA availability and dependencies
```

Reference versions used to produce the results in this repository: Python 3.12 · torch 2.6.0+cu124 ·
open_clip 3.3.0 · faiss-cpu 1.14.3.

## Data

Both datasets are fetched to local disk by scripts; **nothing touches the network at run time**.
Destinations are set by `paths.data_root` / `paths.coco_root` in `configs/default.yaml` — **those two
lines are all you need to change on a new machine.**

| Dataset | Contents | Download | Script |
| --- | --- | --- | --- |
| **Flickr30k** | Karpathy split: 29,000 train / 1,014 val / 1,000 test, 5 captions per image | 4.4 GB | `00_fetch_data.py` |
| **MS-COCO** (optional) | the val2014 part of the Karpathy split: 5,000 test / 5,000 val / 30,504 restval | 6.6 GB | `09_fetch_coco.py` |

COCO is only needed for the cross-domain experiments. Its Karpathy `train` split (82,783 images,
inside `train2014.zip`) is **deliberately not downloaded** — it is reserved as a training pool that
never doubles as a distractor set.

## Running

```bash
# ---- Core pipeline (Flickr30k only) ----
python scripts/00_fetch_data.py         # one-off download, resumable
python scripts/01_sanity_check.py       # reproduce the official baseline: i2t 86.30 / t2i 69.84
python scripts/02_extract_features.py   # cache frozen-CLIP features (everything downstream reuses these)
python scripts/03_eval_baseline_pca.py  # ① and ② need no training — two curves immediately
python scripts/04_train_adapter.py      # ④ Adapter-MRL
python scripts/04_train_adapter.py --set loss.nested=false --set experiment.name=adapter_plain   # ③ ablation
python scripts/05_evaluate_all.py       # four-way table + figures
```

**Do not proceed until `01_sanity_check.py` passes.** It checks against OpenCLIP's published
retrieval numbers; a mismatch means the preprocessing or the i2t/t2i protocol is wrong, which would
invalidate every comparison downstream.

```bash
# ---- Optional experiments ----
python scripts/07_ablate_loss.py                       # 3x3 ablation over temperature and dim weights
python scripts/08_cascade.py --gallery full31k         # cascade retrieval
python scripts/06_train_lora.py                        # Phase 2: unfreeze the backbone via LoRA
python scripts/06_train_lora.py --name ctrl --set "lora.towers=[]"   # control: same budget, backbone still frozen

# Require COCO
python scripts/09_fetch_coco.py
python scripts/01_sanity_check.py --dataset coco       # second baseline: i2t 59.44 / t2i 42.31
python scripts/10_extract_coco.py
python scripts/08_cascade.py --gallery cross_domain    # cascade on a clean 31,504-image gallery
python scripts/11_eval_cross_domain.py                 # switch to COCO's own retrieval task

# Same-domain clean gallery: first train a held-out pair of adapters, otherwise the
# distractors are images the adapter was trained on and the measurement is contaminated
python scripts/04_train_adapter.py --set data.use_holdout=true --set experiment.name=adapter_mrl_ho
python scripts/04_train_adapter.py --set data.use_holdout=true --set loss.nested=false --set experiment.name=adapter_plain_ho
python scripts/08_cascade.py --gallery same_domain
```

Every hyper-parameter lives in `configs/default.yaml`; override any of them from the command line
with `--set key.path=value`. **No code edits required.**

### Timings (single RTX 4070 Laptop, 8 GB)

| Step | Time |
| --- | --- |
| Feature extraction (31 k images + 155 k captions) | ~4 min |
| Adapter training (frozen backbone, 20 epochs) | ~1 min |
| 3×3 loss ablation (9 runs) | ~20 min |
| LoRA (5 epochs, batch 256) | ~32 min |

Adapter training takes a minute because features are cached and the loss runs on 512-d vectors alone.
LoRA changes the backbone at every step, so **cached features cannot be reused** — every step must
run the full forward pass through both encoders, which is an order of magnitude more expensive.

---

## Repository layout

```
configs/default.yaml    single source of configuration (edit YAML, not code)
src/                    library, one responsibility per module
  data / model / features      dataset loading, frozen CLIP, feature caching
  adapter / losses             MRLAdapter; nested InfoNCE with all τ and c_m modes
  reducers                     one interface that flattens all four conditions to transform(emb, m)
  metrics / retrieval          bidirectional R@K; faiss indexing, latency and storage measurement
  cascade                      two-stage retrieval (low-dim shortlist + full-dim re-rank)
  lora                         hand-written LoRA injection
  evaluate / train             evaluation harness, training loop
scripts/                entry points; the number is the execution order, orchestration only
outputs/
  results/  figures/           committed — this is the evidence behind the claims
  features/  checkpoints/      generated at run time, not committed (large and recomputable)
```

`outputs/features/` and `outputs/checkpoints/` are outside version control; the scripts create them
as needed. Running the pipeline in the order above regenerates everything.

## Known limitations

- **Single seed, no error bars.** The 2–3 point effects reported for LoRA are large relative to
  typical run-to-run variance on a 1,000-image test set, but multi-seed replication has not been done.
- **Latency has ~20% cross-run measurement noise** — the same 256-dimension configuration measured
  1.25–1.51 ms across runs. Compare dimension scaling *within* a run, not absolute values across runs.
- **Cascade recall is measured on a 31 k gallery** and is therefore **not comparable** to the
  1 k-test numbers reported in the literature; it is only used to compare conditions on the same gallery.
- **Full-dimension cross-domain ability regresses.** The adapter gains 0.3–1.0 points on Flickr30k but
  loses 4.3–6.0 points on MS-COCO. Low dimensions (m ≤ 128) still transfer far better than the frozen
  baseline, but if what you need is a drop-in replacement at full dimension, this adapter is the wrong
  tool. The cause has been traced to the adapter's domain specialisation rather than backbone drift —
  see [RESULTS.md](RESULTS.md) §7.

## If Hugging Face downloads fail

Datasets and model weights come from Hugging Face. If `huggingface.co` is unreliable from your
network, point `huggingface_hub` at a mirror (no code change needed):

```bash
export HF_ENDPOINT=https://hf-mirror.com     # Windows PowerShell: $env:HF_ENDPOINT="https://hf-mirror.com"
```

`HF_HOME` relocates the model and dataset cache to a larger disk: `export HF_HOME=/path/to/cache`.

## Resources

- Backbone: [OpenCLIP](https://github.com/mlfoundations/open_clip) ViT-B/16, `laion2b_s34b_b88k`
- Flickr30k (Karpathy split): [`nlphuji/flickr30k`](https://huggingface.co/datasets/nlphuji/flickr30k)
- MS-COCO (Karpathy split annotations): [`yerevann/coco-karpathy`](https://huggingface.co/datasets/yerevann/coco-karpathy); images from [cocodataset.org](https://cocodataset.org)
- Method: Kusupati et al., *Matryoshka Representation Learning*, NeurIPS 2022

## License

MIT — see [LICENSE](LICENSE).
