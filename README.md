# Open-Vocabulary Referring Expression Segmentation

Zero-shot referring-expression segmentation: given an image and a free-text expression (e.g. *"bowl behind the others can only see part"*), localize and segment the described object. Built with **Grounding DINO** for text-conditioned detection and **SAM 2** for mask refinement. No training on RefCOCO.

## Pipeline

```
image + text expression
      │  Grounding DINO (text_threshold=0.25, top-scoring box)
      ▼
   bounding box ──► SAM 2 (box prompt, multimask_output=False, >0.5) ──► binary mask
```

Per-sample IoU is computed against ground-truth masks rasterized from RefCOCO's polygon annotations. One expression per sample (`answer[0]`).

## Results — RefCOCO validation set (8,811 samples)

| Method | mIoU | P@0.5 | P@0.7 | P@0.9 |
|---|---|---|---|---|
| DINO box-only baseline | 0.4910 | 0.576 | 0.174 | 0.008 |
| DINO + SAM 2 (full pipeline) | 0.7263 | 0.817 | 0.757 | 0.381 |

On the full set, SAM 2 contributes **+0.2353 mIoU** (0.4910 → 0.7263).

**Ablation (100-sample pilot, 100 valid / 0 failed):**

| Method | mIoU | P@0.5 | P@0.7 | P@0.9 |
|---|---|---|---|---|
| DINO box-only | 0.4994 | 0.610 | 0.160 | 0.010 |
| DINO + SAM 2 | 0.7549 | 0.850 | 0.820 | 0.410 |

SAM 2 contributes **+0.2555 mIoU**. The most telling jump is P@0.7 (0.16 → 0.82): boxes get the location roughly right, but rectangles have an IoU ceiling against irregular object shapes — SAM 2 turns "roughly right" into "shape-accurate".

## What the numbers mean

Rough scale on RefCOCO val: 0.5 ≈ right region (this is where box-only methods plateau); 0.7+ = shape-accurate; 0.9 = near pixel-perfect.

Context: trained baselines report ~0.62–0.81 cIoU (MCN 0.624, CRIS 0.705, LAVT 0.727, ReLA 0.738, LISA-7B 0.741, Prompt-RIS 0.781, One-Ref 0.798) — all trained on RefCOCO. This pipeline is **zero-shot** (no RefCOCO training) and lands in the same range. Two honest caveats:

1. Literature reports **cIoU** (intersection/union aggregated over the whole dataset — large objects dominate); this project reports **per-sample mIoU** (each sample weighted equally). Same magnitude, not directly interchangeable.
2. The headline figure (0.7263) is the full-validation number; the 0.7549 figure is the 100-sample pilot — small samples run optimistic.

## Error analysis

Failure rate (mask IoU < 0.5): **1,614 / 8,811 = 18.3%**. Breaking failures down by pipeline stage shows the bottleneck is detection, not segmentation:

- **91% of failures** occur when the grounded box already overlaps poorly (box IoU < 0.5); in 398 failures (24.7%) the box has zero overlap with the target at all.
- Pure mask failures — box IoU ≥ 0.5 yet the final mask still scores < 0.5 — are rare: 143 samples, **1.6%** of all samples. When DINO localizes well, SAM 2 almost always delivers.

An early hypothesis from the pilot — that spatial-relation expressions fail more often — did **not** hold on the full set: expressions containing spatial keywords fail at 17.7% vs 19.2% for the rest. Individual cases do show it (for *"bowl behind the others can only see part"*, the model segments the prominent metal bowl in front, IoU = 0.0), but it is not the systematic driver.

Even with SAM 2, P@0.9 stays modest (0.381 on the full set) — boundary precision remains the headroom.

## Demo

Qualitative results (original image + DINO box | ground truth | prediction):

