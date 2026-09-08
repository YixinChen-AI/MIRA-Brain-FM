# OmniMIRA ROI feature output

`save_roi_features` writes schema version `1.0` as a NumPy NPZ archive that is
loadable with `numpy.load(path, allow_pickle=False)`. Model version is fixed as
`omnimira-public-v1`; the archive records the SHA256 of the checkpoint bytes.

## Archive contents

The archive contains exactly: `schema_version`, `model_version`,
`checkpoint_sha256`, `modality`, `feature_dim`, `feature_dtype`,
`atlas_hashes_json`, `source_metadata_json`, and the three groups
`{aal3,ho69,yeo7}_{features,roi_ids,roi_names}`.

Feature arrays are float32 with shapes `(166, 128)`, `(69, 128)`, and
`(7, 128)`. ROI IDs are int32. Names and scalar text metadata are NumPy Unicode
strings. `feature_dim` is 128 and `feature_dtype` is `float32`. Both JSON
strings use sorted keys and compact deterministic separators.

## ROI order

AAL3 output follows the 166 occupied AAL3v1 labels in ascending raw-label
order. Placeholder IDs 35, 36, 81 and 82 are excluded; valid IDs 167--170 are
included. The model uses contiguous indices `0..165`, while `aal3_roi_ids`
preserves the corresponding raw atlas IDs.

HO69 uses IDs `1..69`: 48 Harvard-Oxford cortical labels followed by 21
subcortical labels. Yeo7 uses LUT IDs `1..7`. Label-source metadata is
included in `assets/atlases`; atlas images and label tables must be supplied
separately as described in [Atlas setup](atlas_setup.md). No atlas or label
is downloaded at runtime. `atlas_hashes_json` records the atlas SHA256 values.

## Source metadata

`source_metadata_json` has exactly these typed fields: `schema_version`,
`input_contract_version`, `input_contract_sha256`, `input_sha256`,
`template_sha256`, `brain_mask_sha256`, `space`, `shape`, `affine`,
`qform_code`, `sform_code`, `orientation`, `modality`, `normalization`, and
`preprocessing_state`. SHA256 values are lowercase 64-character hexadecimal
strings, shape is three integers, affine is a finite numeric 4x4 matrix, form
codes are integers, and preprocessing state is `normalized_v2`.

The writer validates this metadata against `configs/output_schema.json` and
the packaged input contract. All four modalities use the same canonical RAS+
model grid.
