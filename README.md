# ACIM Low-Precision Ensemble


<!-- PAPER_METADATA_START -->

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TVLSI-00629B)](https://doi.org/10.1109/TVLSI.2026.3666118)
![Publication](https://img.shields.io/badge/Publication-2026-green)
![Framework](https://img.shields.io/badge/Framework-NeuroSim%20V1.4-blue)

## Associated Publication

**Exploiting Hardware Nonidealities for Robust
Low-Precision Ensembles on ACIM**

Donghyuck Lee and Tae-Hyeon Kim

IEEE Transactions on Very Large Scale Integration
(VLSI) Systems, Vol. 34, No. 6, June 2026.

**Paper:** https://doi.org/10.1109/TVLSI.2026.3666118

This repository provides the research implementation
and available experimental results associated with
the published article, including recovered historical
results and subsequent evaluations.

<!-- PAPER_METADATA_END -->


Research artifact associated with [Exploiting Hardware Nonidealities for Robust Low-Precision Ensembles on ACIM](https://doi.org/10.1109/TVLSI.2026.3666118), IEEE TVLSI (2026).

## Method
Two independently perturbed low-precision replicas exploit conductance variation; their outputs are aggregated for classification. The hardware evaluation extends NeuroSim V1.4 for SRAM, RRAM and FeFET.

## Repository layout
- `accuracy/`: surviving VGG-8/CIFAR-10, VGG-8/CIFAR-100, ResNet-20/CIFAR-10 and ensemble-size implementations.
- `hardware/`: single and ensemble simulators for each memory technology.
- `results/cohorts.csv`: result locations and cohort identifiers. Benchmark folders contain individual and ensemble summaries and records. `results/accuracy/raw/` preserves original CSVs; aggregate tables are retained for provenance.
- `results/hardware/paper_candidates.csv`: nine selected hardware configurations; original component summaries in `raw/`.
- `scripts/`: launchers, result recovery and plotting.
- `docs/`: dependency, checkpoint and release audit information.

## Data provenance
The recovered VGG-8/CIFAR-10 historical cohort matches published anchor values, but not every figure point has been independently validated. The updated CIFAR-10 cohort differs from the published experiment. Original CIFAR-100 records are unavailable; the provided updated evaluation is not a numerical reproduction of Fig. 7. ResNet-20 and ensemble-size scaling records are partial. The stored counts and grouping scheme are preserved; some runs differ from the paper's stated replica or ensemble counts. Missing values are not synthesized. Raw accuracy is a fraction; summary accuracy is percent. Mean is the unweighted mean of recorded instances, with weighted accuracy also retained. Diversity figures are not reconstructed here.

## Usage
Use a compatible Python/PyTorch environment. See `docs/checkpoints.csv` for expected local checkpoint paths. Checkpoints are not bundled; obtain authorized copies before inference. Launchers run the surviving implementation, not necessarily the historical paper code. Original launch scripts are archival and may contain machine-specific settings. No inference is required to plot stored results.

```bash
python scripts/plot_results.py
bash scripts/run_accuracy.sh vgg8_cifar100 4 0.25
bash scripts/run_hardware.sh single SRAM 4
```

Hardware requires a compiled NeuroSim executable; build with the supplied Makefile if available. Review source paths and dependencies before use. Full independent reproduction has not been tested. Preserve upstream notices and confirm redistribution rights before public release.

## Pretrained Checkpoints

Three unique pretrained checkpoints are required:

| Model | Checkpoint |
|---|---|
| VGG-8 / CIFAR-10 | VGG8.pth |
| VGG-8 / CIFAR-100 | VGG_CIFAR100.pth |
| ResNet-20 / CIFAR-10 | decay0_W8.pth |

The checkpoints are distributed separately through GitHub
Releases, subject to redistribution permissions.

After the release assets become available, install the
checkpoints using GitHub CLI:

    gh auth login
    python scripts/setup_checkpoints.py

Alternatively, install checkpoints from a local directory:

    python scripts/setup_checkpoints.py --local-dir /path/to/checkpoints

The installer validates SHA-256 checksums and copies each
checkpoint to the required inference directories.

The updated CIFAR-100 results are provided as a subsequent
evaluation and do not exactly reproduce Fig. 7 of the
published article.
