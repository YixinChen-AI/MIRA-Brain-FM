# Pretraining OmniMIRA

This guide describes the included development configuration. The configuration
for the final manuscript experiments is not yet released.

The training code includes seven objectives: patch reconstruction, ROI masked
autoencoding, cross-modal ROI identity, augmentation consistency,
chronological age, native-space ROI volume and spatial geometry. Active terms
are combined with homoscedastic uncertainty weighting.

## Manifest

The manifest is a JSON list. Every record contains an MNI-standardized NIfTI
path, modality and participant identifier. Age and per-atlas targets are
optional; missing targets are masked from their corresponding losses.

```json
[
  {
    "path": "sub-01_T1w_mni.nii.gz",
    "modality": "t1",
    "subject_id": "sub-01",
    "age": 71.3,
    "roi_volumes": {"aal3": "sub-01_aal3_volume.npy"},
    "roi_distances": {"aal3": "sub-01_aal3_distance.npy"}
  }
]
```

For existing manifests with `npy_path` and `subject` fields, convert the field
names with:

```bash
python scripts/build_public_pretraining_manifest.py \
  --input source_manifest.json \
  --output manifest.json
```

Manifests contain local data paths and should not be committed to a public
repository. Checkpoints record manifest and recipe hashes rather than those paths.

The sampler constructs batches of 64 scans, with 16 scans from each of T1,
amyloid-PET, FDG-PET and CT, and at least 32 distinct participants.

## Input checks

Before training, validate the manifest and its referenced paths:

```bash
python scripts/validate_pretraining_manifest.py --manifest manifest.json
```

Also verify the atlas binaries used for this run. Set `OMNIMIRA_ATLAS_DIR` when
the atlases are supplied outside the package:

```bash
export OMNIMIRA_ATLAS_DIR=/absolute/path/to/omnimira-atlases
python scripts/verify_atlases.py --atlas-dir "$OMNIMIRA_ATLAS_DIR"
```

Check the setup before training. This checks the
manifest, atlas identities, ROI mappings and model construction without reading
individual training images:

```bash
python scripts/pretrain.py --manifest manifest.json --preflight
```

For cached NumPy inputs, check the preprocessing metadata:

```bash
python scripts/audit_pretraining_inputs.py \
  --manifest manifest.json --strict
```

If cached NumPy inputs have missing or invalid sidecars, rebuild the cache
from an MNI-space NIfTI manifest:

```bash
python scripts/prepare_pretraining_cache.py \
  --input raw_nifti_manifest.json \
  --cache-dir /path/to/cache \
  --output manifest.json
```

This command applies RAS world-coordinate resampling and
modality-specific normalization, then writes the matching `normalized_v2`
sidecar beside every cache file. It rejects legacy `.npy` files
as source inputs because they do not retain enough spatial provenance.

The validator rejects missing modalities, fewer than 16 scans in a modality,
fewer than 32 distinct `subject_id` values, missing paths and unsupported
modalities. It does not read image contents.

## Training

```bash
python scripts/pretrain.py \
  --config configs/omnimira_3atlas.yaml \
  --manifest manifest.json \
  --out-dir runs/omnimira_public
```

The included recipe uses AdamW (`betas=(0.9, 0.95)`, weight decay `0.05`), a base
learning rate of `1.5e-4`, 30 warm-up epochs from `1.5e-6`, cosine decay to
`1e-6`, 300 epochs and BF16 where supported. Checkpoints include the model,
loss state, optimizer state, RNG state and `model_version`.

All scans are resampled in world coordinates to the canonical RAS+ model grid.
Atlas labels use nearest-neighbour interpolation; continuous images use
trilinear interpolation.
