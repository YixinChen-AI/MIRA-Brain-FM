# OmniMIRA

## Current release status

**Source-code preview only. Model weights, a verified checkpoint and a hosted demo are not yet publicly available.**

The current manuscript describes an anatomically indexed foundation model pretrained on **91,880 scans**, evaluated across **24 tasks in 11 cohorts**. The source and configuration metadata below originate from an earlier development release and have not yet been reconciled with the final manuscript experiments. Historical counts in configuration files are not the current manuscript counts. This preview is not a complete reproduction package.

Checkpoint files and third-party atlas and MNI reference images are deliberately omitted. Installation provides source code only. The feature-extraction commands below describe the development interface and require resources that are not yet included in this release. See [TODO.md](TODO.md) for progress.

OmniMIRA is an anatomically indexed foundation model for mixed-modality
neuroimaging. It extracts frozen ROI embeddings from T1 MRI, amyloid-PET,
FDG-PET and CT.

## Install

```bash
pip install -e .
```

## Extract ROI features

```bash
python examples/extract_roi_features.py \
  --input scan_mni.nii.gz \
  --modality t1 \
  --checkpoint omnimira_public_v1.pt \
  --output features.npz
```

Before extraction, verify the fixed atlas contract. This is also the supported
path for a release that does not redistribute third-party atlas binaries:

```bash
export OMNIMIRA_ATLAS_DIR=/absolute/path/to/omnimira-atlases
python scripts/verify_atlases.py --atlas-dir "$OMNIMIRA_ATLAS_DIR"
```

The loader resamples an MNI-standardized scan to the canonical RAS+ model grid,
applies the modality-specific normalization and exports float32 ROI embeddings:

- AAL3: `(166, 128)`
- Harvard-Oxford: `(69, 128)`
- Yeo 7-network: `(7, 128)`

The final checkpoint provenance and configuration will be verified before publication. No public checkpoint is available yet.

## Test

The model and loss unit tests use synthetic inputs:

```bash
python -m pytest -q tests/test_model_forward.py tests/test_losses.py
```

The complete suite additionally requires verified atlas and template resources:

```bash
OMNIMIRA_ATLAS_DIR=/absolute/path/to/omnimira-atlases python -m pytest -q
```

See `docs/atlas_setup.md`, `docs/input_preparation.md`,
`docs/output_format.md` and `docs/training.md` for the exact contracts.
Research use only; this software is not a clinical or diagnostic device.

## Export the public repository

This directory is maintained inside a larger research workspace. Export a
standalone GitHub-ready tree rather than pushing that workspace directly:

```bash
python scripts/export_public_tree.py --output ../OmniMIRA-public
cd ../OmniMIRA-public
python scripts/audit_public_release.py
git init
```

The export excludes internal audit records, pre-release result artifacts and
third-party atlas resources. It retains the public release contract and the
external atlas verification command.
