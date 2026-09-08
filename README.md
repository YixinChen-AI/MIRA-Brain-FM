# OmniMIRA

OmniMIRA learns brain-region representations from T1-weighted MRI, amyloid PET, FDG PET and CT. It combines local image features within atlas-defined regions, allowing the same embeddings to be used for downstream prediction and regional analysis.

## Release status

This is an initial source-code release. The included implementation and configurations are from an earlier development version and are being aligned with the final manuscript. Pretrained weights are not yet available. See the [release roadmap](TODO.md) for planned updates.

## Installation

```bash
pip install -e .
```

## Usage

Feature extraction requires a pretrained checkpoint and the atlas and MNI reference files, which are not included in this release. The example interface is:

```bash
python examples/extract_roi_features.py \
  --input scan_mni.nii.gz \
  --modality t1 \
  --checkpoint /path/to/checkpoint.pt \
  --output features.npz
```

The output contains 128-dimensional embeddings for each region: 166 AAL3 regions, 69 Harvard–Oxford regions and 7 Yeo networks.

- [Input preparation](docs/input_preparation.md)
- [Atlas setup](docs/atlas_setup.md)
- [Output format](docs/output_format.md)
- [Model architecture](docs/model_architecture.md)
- [Pretraining](docs/training.md)

## Tests

Model and loss tests use synthetic inputs:

```bash
python -m pytest -q tests/test_model_forward.py tests/test_losses.py
```

The full test suite also requires the atlas and template files.

## License

Source code is available under the [Apache 2.0 license](LICENSE). Third-party resources have [separate terms](THIRD_PARTY_NOTICES.md).

For research use only; not for clinical diagnosis.
