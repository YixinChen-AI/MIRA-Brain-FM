# Pretraining OmniMIRA

The public checkpoint is trained from the paper-defined implementation in this
repository. The seven objectives are patch reconstruction, ROI masked
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

If the authorised training environment already maintains a private manifest
with `npy_path` and `subject` fields, convert it once without altering the
underlying records:

```bash
python scripts/build_public_pretraining_manifest.py \
  --input private_source_manifest.json \
  --output omnimira_public_pretrain.json
```

The converted manifest remains private because it contains local data paths.
The training checkpoint records its SHA256 digest and the recipe digest, not
those paths.

The public sampler constructs a batch size 64 with 16 scans from each of T1,
amyloid-PET, FDG-PET and CT, and at least 32 distinct participants.

Before launching public pretraining, validate the manifest and its referenced
paths:

```bash
python scripts/validate_pretraining_manifest.py --manifest manifest.json
```

Also verify the atlas binaries used for this run. Set `OMNIMIRA_ATLAS_DIR` when
the atlases are supplied outside the package:

```bash
python scripts/verify_atlases.py --atlas-dir "$OMNIMIRA_ATLAS_DIR"
```

Run the complete launch preflight before scheduling a GPU job. This checks the
manifest, atlas identities, ROI mappings and model construction without reading
individual training images:

```bash
OMNIMIRA_ATLAS_DIR=/absolute/path/to/omnimira-atlases \
python scripts/pretrain.py --manifest omnimira_public_pretrain.json --preflight
```

For a private NumPy cache, audit sidecar compliance on an allocated CPU node
before training. The command reports only aggregate counts and never writes
input paths to its output:

```bash
python scripts/audit_pretraining_inputs.py \
  --manifest omnimira_public_pretrain.json --strict
```

Do not relabel a legacy NumPy cache as V4. When the audit finds missing or
invalid sidecars, rebuild the private cache from an MNI-space NIfTI manifest:

```bash
python scripts/prepare_pretraining_cache.py \
  --input raw_nifti_manifest.json \
  --cache-dir /private/output/omnimira_v4_cache \
  --output omnimira_public_pretrain.json
```

This command uses the committed RAS world-coordinate resampling and
modality-specific normalization, then writes the matching `normalized_v2`
sidecar beside every cache file. It intentionally rejects legacy `.npy` files
as source inputs because they do not retain enough spatial provenance.

The validator rejects missing modalities, fewer than 16 scans in a modality,
fewer than 32 distinct `subject_id` values, missing paths and unsupported
modalities. It does not read image contents.

## Run

```bash
python scripts/pretrain.py \
  --config configs/omnimira_3atlas.yaml \
  --manifest manifest.json \
  --out-dir runs/omnimira_public
```

The V4 recipe uses AdamW (`betas=(0.9, 0.95)`, weight decay `0.05`), a base
learning rate of `1.5e-4`, 30 warm-up epochs from `1.5e-6`, cosine decay to
`1e-6`, 300 epochs and BF16 where supported. Checkpoints include the model,
loss state, optimizer state, RNG state and `model_version`.

All scans are resampled in world coordinates to the canonical RAS+ model grid.
Atlas labels use nearest-neighbour interpolation; continuous images use
trilinear interpolation.
