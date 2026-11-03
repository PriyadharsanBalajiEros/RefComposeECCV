# RefCompose: Multi-Reference Image Generation via LoRA-Conditioned Diffusion

Official PyTorch / Diffusers codebase for **"RefCompose: Multi-Reference Image Generation via LoRA-Conditioned Diffusion"** (`arXiv:2609.32389v1`).

**Authors:** Sai Sri Teja Kuppa¹, Parth Shinde¹,², Priyadharsan Balaji S¹,², Jinka Harshavardhan¹, and Sriprabha Ramanarayanan¹  
*¹ Eros Innovation, India · ² Indian Institute of Technology Madras, India*

---

## Overview

Existing multi-reference diffusion frameworks tokenize each reference image independently, causing memory and attention complexity to scale linearly with reference count $N$ (going Out-Of-Memory at $N=6$ on an 80GB H100 at $1280 \times 720$). **RefCompose** decouples *where* objects appear (structure) from *what* they look like (appearance) at strictly **constant token and memory cost** regardless of $N$:

1. **Stage 1 — Prompt-Induced Layout Generation (Section 3.1):**  
   A structured compositional prompt $\mathbf{p}$ induces a coarse layout image $L = \mathcal{F}_\theta(\mathbf{p}) \in \mathbb{R}^{H \times W \times 3}$ over $T=40$ flow-matching steps from frozen **FLUX.2 Dev**, localized via **Grounding DINO** ($\mathcal{B} = \mathcal{G}(L)$) and geometrically scaffolded via **Depth Anything 3** ($D_s = \Psi(L)$) without requiring any LLM or dedicated layout generator.
2. **Stage 2 — Correspondence-Guided Reference Canvas (Sections 3.2 & 3.3):**  
   All $N$ references $\mathcal{R} = \{I_1, \dots, I_N\}$ are resized and composited in pixel space onto a single fixed-resolution $1280 \times 720$ **Reference Canvas** $\mathcal{C}$ using far-to-near depth ordering ($\bar{d}_k = \mathrm{mean}(D_s \mid b_k)$), encoded once by the VAE into constant-size latents $z_{\mathcal{C}} = \mathcal{E}(\mathcal{C})$ and $z_{D_s} = \mathcal{E}(D_s)$.
3. **Stage 3 — Decoupled Structure-Appearance (DSA) LoRA (Sections 3.4 & 4.3):**  
   Extends **EasyControl**'s Condition Injection LoRA with two orthogonal rank-$32$ LoRA streams ($\Delta W_a = B_a A_a$ for $z_{\mathcal{C}}$, loaded before $\Delta W_s = B_s A_s$ for $z_{D_s}$) in the double-stream MMDiT attention blocks, kept disentangled via **block-causal attention masking** and **Step-0 `bank_kv` KV caching**.

---

## Repository Structure

```text
Refcompose/
├── configs/
│   ├── default.yaml               # FLUX.2-dev, 1280x720, T=40, LoRA r=32 defaults
│   ├── data_prep.yaml             # 5-stage dataset curation & synthetic corpus config
│   ├── train_dsa_lora.yaml        # 3-stage 50,000-iteration training curriculum config
│   ├── eval_denselayout.yaml      # DenseLayout (5,000 test composites, N=5 & N>5) config
│   └── benchmark_memory.yaml      # Peak GPU VRAM scaling benchmark (Table 1) config
├── refcompose/
│   ├── config.py                  # Hierarchical typed config loader
│   ├── prompts/template.py        # Section 4.2 structured template & priority scale steering
│   ├── canvas/
│   │   ├── composer.py            # Eq. (3)-(7) pixel-space Reference Canvas & depth overwrite
│   │   └── augmentations.py       # Stage-3 photometric jitter, affine warps & box shifts
│   ├── models/
│   │   ├── dsa_lora.py            # Eq. (8)-(12) DSA LoRA, causal mask & bank_kv cache
│   │   ├── flux2_wrapper.py       # FLUX.2 Dev wrapper, VAE pack/unpack & T=40 Euler sampler
│   │   ├── detector.py            # Grounding DINO localization + Hungarian assignment
│   │   ├── depth_estimator.py     # Depth Anything 3 (DA3) monocular geometric scaffold
│   │   └── vlm_annotator.py       # Qwen3-VL object enumeration, slot annotation & VQA
│   ├── pipeline/
│   │   └── refcompose_pipeline.py # End-to-end 3-stage pipeline + multi-ref swap & depth control
│   ├── data/
│   │   ├── curation_pipeline.py   # Section 4.1 5-stage cinematic curation (>4 dets, DINO >0.90)
│   │   ├── synthetic_generator.py # FLUX.2 diverse pose/orientation synthetic scene generator
│   │   ├── dataset.py             # 50%/50% training Dataset with dynamic 3-stage curriculum
│   │   └── denselayout_loader.py  # InstanceAssemble DenseLayout test split loader & canonicalizer
│   ├── training/
│   │   ├── curriculum.py          # Stage 1 (0-10k), Stage 2 (10k-30k), Stage 3 (30k-50k)
│   │   ├── losses.py              # Eq. (12) Conditional Rectified Flow-Matching loss L_FM
│   │   └── trainer.py             # Accelerate bf16 trainer, safetensors checkpointing & resume
│   ├── evaluation/
│   │   ├── metrics_region.py      # LayoutSAM-Eval: Spatial, Color, Texture, Shape
│   │   ├── metrics_global.py      # PICK, DINO, DINO_refs, CLIP-I, CLIP-T
│   │   └── evaluator.py           # Full Table 2 benchmark runner & Markdown/LaTeX exporters
│   └── benchmarks/
│       ├── baselines.py           # CreatiLayout, InstanceAssemble, UNO, XVerse, RefCompose
│       ├── memory_profiler.py     # Table 1 peak GPU VRAM & latency profiler (with OOM capture)
│       └── ablations.py           # Figures 4, 5, 6, 7 & Section 4.6 ablation experiments
├── scripts/
│   ├── prepare_dataset.py         # CLI for 5-stage dataset curation
│   ├── train.py                   # CLI for 3-stage 50k-step DSA LoRA training
│   ├── infer.py                   # CLI for multi-reference generation
│   ├── evaluate.py                # CLI for DenseLayout quantitative evaluation (Table 2)
│   ├── benchmark_memory.py        # CLI for GPU memory scaling benchmark (Table 1)
│   └── run_ablations.py           # CLI for Figures 4-7 & ablation studies
└── tests/                         # Unit & integration test suite
```

---

## Installation

```bash
cd Refcompose
pip install -e .
pip install -r requirements.txt

# Install Depth Anything 3 (DA3, arXiv:2511.10647)
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
pip install -e Depth-Anything-3
```

---

## 1. Five-Stage Dataset Preparation (Section 4.1)

Curates raw $1280 \times 720$ cinematic movie shots alongside a 50%/50% synthetic FLUX.2 corpus:
1. **Qwen3-VL** enumerates all actors, props, and landmarks.
2. **Grounding DINO** localizes objects, retaining only shots with **$> 4$ detections**.
3. **FLUX.2 Dev** canonicalizes each crop to neutral pose and lighting.
4. **DINO** cosine similarity filters canonicalized crops (**$> 0.90$**).
5. **Qwen3-VL** generates structured scene/crop annotations & **Depth Anything 3** computes $D_s$.

```bash
python scripts/prepare_dataset.py \
    --config configs/data_prep.yaml \
    --raw-cinematic-dir ./data/raw/cinematic_frames \
    --generate-synthetic
```

---

## 2. Three-Stage DSA LoRA Training (Section 4.3)

Trains rank $r=32$ DSA LoRA adapters on frozen FLUX.2 Dev for **50,000 iterations** at $1280 \times 720$:
- **Stage 1 (`0 – 10,000`):** Depth $D_s$ + black canvas $\mathcal{C} = \mathbf{0}$ (initializes structure branch $\Delta W_s$).
- **Stage 2 (`10,000 – 30,000`):** Canonicalized crops placed at clean Grounding DINO boxes on ground-truth image.
- **Stage 3 (`30,000 – 50,000`):** Adds photometric jitter, mild affine warps, and small box translations.

```bash
accelerate launch scripts/train.py --config configs/train_dsa_lora.yaml
```

---

## 3. Multi-Reference Inference

```bash
python scripts/infer.py \
    --config configs/default.yaml \
    --lora-dir ./outputs/checkpoints/refcompose_dsa_lora/final \
    --references assets/bed.png assets/lamp.png assets/rug.png assets/backpack.png assets/toy.png \
    --descriptions "wooden bed" "vintage table lamp" "woven rug" "leather backpack" "wooden toy" \
    --output-dir ./outputs/demo_bedroom
```

---

## 4. Quantitative Evaluation & Benchmarking (Section 4.4–4.6)

### Table 2: Dense Layout Benchmark ($N=5$ and $N>5$)
Evaluates all 9 metrics (`Spatial`, `Color`, `Texture`, `Shape`, `PICK`, `DINO`, `DINO_refs`, `CLIP-I`, `CLIP-T`) and exports `.json`, `.csv`, `.md`, and `.tex` tables:

```bash
python scripts/evaluate.py --config configs/eval_denselayout.yaml
```

### Table 1: Peak GPU Memory Scaling ($N = 2, 4, 6, \dots, 15$)
```bash
python scripts/benchmark_memory.py --config configs/benchmark_memory.yaml
```

### Figures 4–7: Qualitative Grids, Multi-Reference Swap, Depth Control & Ablations
```bash
python scripts/run_ablations.py \
    --config configs/eval_denselayout.yaml \
    --lora-dir ./outputs/checkpoints/refcompose_dsa_lora/final \
    --output-dir ./outputs/ablations
```

---

## Running Unit & Dry-Run Verification Tests

All components can be tested offline without GPU or model downloads using `--mock` or `unittest`:

```bash
python -m unittest discover -s tests -v
```

---

## Citation

```bibtex
@article{kuppa2026refcompose,
  title   = {RefCompose: Multi-Reference Image Generation via LoRA-Conditioned Diffusion},
  author  = {Kuppa, Sai Sri Teja and Shinde, Parth and Balaji S, Priyadharsan and Harshavardhan, Jinka and Ramanarayanan, Sriprabha},
  journal = {arXiv preprint arXiv:2609.32389},
  year    = {2026}
}
```

