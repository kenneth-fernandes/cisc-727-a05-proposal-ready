# CISC-727 Assignment A05 — ARE-II Ready Package

**Author:** Kenneth Peter Fernandes  
**Course:** CISC-727 — Advanced Research Explorations (ARE) - I  
**Instructor:** Professor Majid Shaalan, Ph.D.  
**Institution:** Harrisburg University of Science and Technology  
**Date:** April 25, 2026

This repository is the integrated bundle for Assignment A05. It consolidates the work of CISC-727 — Advanced Research Explorations (ARE) - I into a single advisor-ready package and proposes the next-semester (CISC-733, ARE-II) execution plan.

---

## What's in this repository

```
.
├── README.md                       (this file)
├── requirements.txt                (Python dependencies)
├── proposal/
│   ├── AREII_Ready_Package.tex     (the main bundle: proposal + method + pilot + ARE-II plan)
│   ├── AREII_Ready_Package.pdf     (compiled output)
│   └── references.bib              (BibTeX, 22 sources)
├── related_work/
│   ├── related_work.tex            (standalone literature map + gap matrix + annotated bibliography)
│   └── related_work.pdf            (compiled output)
├── configs/
│   ├── baseline.yaml               (A03 baseline: patch_size=16, seq_len=5)
│   ├── variant.yaml                (A04 strong-result variant: patch_size=4, seq_len=65)
│   └── are_ii/
│       ├── sprint1_kernel_attribution.yaml
│       ├── sprint2_microbenchmark.yaml
│       ├── sprint3_full_model.yaml
│       └── sprint4_a100_stretch.yaml
├── src/
│   └── dist_train.py               (cleaned distributed-training harness)
└── results/
    ├── primary_figure.png          (A04 primary figure — pilot strong result)
    ├── supporting_breakdown.png    (A04 forward/backward time decomposition)
    ├── summary.csv                 (A04 summary table)
    └── all_results.json            (A04 full per-replicate data)
```

`knowledge/` and `notes/` are present locally but excluded by `.gitignore`. They contain reference materials from prior assignments (A01–A04) and personal planning notes.

---

## How to reproduce the pilot strong result (A04)

The pilot was run on Kaggle Notebooks with two NVIDIA Tesla T4 GPUs.

### 1. Set up a Kaggle notebook

1. Create a new Kaggle notebook.
2. Under **Settings → Accelerator**, choose **GPU T4 x2**.
3. Add the [`cifar-10-python` Kaggle dataset](https://www.kaggle.com/datasets/pankrzysiu/cifar10-python) as input. It mounts at `/kaggle/input/cifar-10-python/`.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The harness uses only stable PyTorch and torchvision APIs. Kaggle's default Python image typically already contains a compatible PyTorch.

### 3. Run the four pilot configurations

Each cell below produces one JSON output file. Run from the repository root.

```bash
# Baseline at sequence length 5 (patch_size=16), math attention
torchrun --nproc_per_node=2 src/dist_train.py 16 0 results/r1_baseline_std.json

# Baseline at sequence length 5, requested FlashAttention (NB: see Part IV of the bundle)
torchrun --nproc_per_node=2 src/dist_train.py 16 1 results/r1_baseline_flash.json

# Variant at sequence length 65 (patch_size=4), math attention
torchrun --nproc_per_node=2 src/dist_train.py 4 0 results/r2_variant_std.json

# Variant at sequence length 65, requested FlashAttention
torchrun --nproc_per_node=2 src/dist_train.py 4 1 results/r2_variant_flash.json
```

The argument signature is: `PATCH_SIZE  USE_FLASH (0 or 1)  OUTPUT_FILE`.

### 4. Aggregate and re-render the primary figure

The pilot's figure-generation cells live in the original Kaggle notebook (preserved under `knowledge/a04/notebook/a04_strong_result.ipynb`). The output `results/primary_figure.png` and `results/summary.csv` in this repository are the artifacts produced by that notebook on April 24, 2026.

For ARE-II, the harness is being extended to read the configurations under `configs/are_ii/` directly and produce the kernel-attribution table and crossover plot described in [`proposal/AREII_Ready_Package.tex`](proposal/AREII_Ready_Package.tex), Part III.

---

## How to compile the LaTeX bundle

The repository's PDFs are produced from the LaTeX sources in `proposal/` and `related_work/`. To rebuild on a machine with TeX Live and `biber`:

```bash
# Main bundle
cd proposal
pdflatex AREII_Ready_Package.tex
biber AREII_Ready_Package
pdflatex AREII_Ready_Package.tex
pdflatex AREII_Ready_Package.tex

# Standalone related-work package
cd ../related_work
pdflatex related_work.tex
biber related_work
pdflatex related_work.tex
pdflatex related_work.tex
```

Both documents share the bibliography at `proposal/references.bib`.

---

## Hardware on which this work was developed

| Component | Specification |
|---|---|
| Platform | Kaggle Notebooks (free tier) |
| GPU | NVIDIA Tesla T4 (16 GB HBM2), 2 GPUs available |
| Compute capability | sm_75 (Turing) |
| Interconnect | PCIe |
| Framework | PyTorch 2.10.0+cu128 |
| Dataset | CIFAR-10 (170.5 MB) via Kaggle input |

A documented caveat (see Part IV of the bundle): PyTorch's `SDPBackend.FLASH_ATTENTION` path requires sm_80 or newer. Tesla T4 is sm_75. Pilot runs that requested the FlashAttention backend on Tesla T4 likely fell back to a different kernel; resolving this attribution is the first deliverable of ARE-II.

---

## ARE-II execution plan (summary)

The full plan is in [`proposal/AREII_Ready_Package.tex`](proposal/AREII_Ready_Package.tex), Section 8 ("ARE-II Execution Plan"). In one sentence: four sprints across the May 9 – first-week-of-August window, with weekend-only working hours, treating Sprint 4 (an Ampere-class spot check) as a stretch goal that can be dropped without remorse.

| Sprint | Deliverable | Weekend budget |
|---|---|---|
| 1 — Kernel attribution | Per-backend kernel signatures and warning logs on Tesla T4 | 1 |
| 2 — Microbenchmark sweep | Crossover map across kernels × precision × sequence length | 3 |
| 3 — Full-model validation | End-to-end ViT runs at extended sequence length | 2–3 |
| 4 — A100 spot check (stretch) | Cross-architecture comparison on Ampere | 1 (drop if late) |
| Writing | Committee report + figure polish + reproducibility cleanup | 2–3 |

---

## Submission package contents (Option A: PDF + repo link)

The advisor-ready submission is:

1. The compiled PDF: [`proposal/AREII_Ready_Package.pdf`](proposal/AREII_Ready_Package.pdf)
2. A link to this repository

---

## Acknowledgments

This package builds on materials from CISC-727 Assignments A01 through A04. AI tools used in its preparation are itemized in the AI Usage Log appendix of the bundle.
