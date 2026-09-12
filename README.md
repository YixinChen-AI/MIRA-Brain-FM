# OmniMIRA

### An anatomically indexed foundation model across structural, molecular and metabolic neuroimaging

OmniMIRA learns brain-region representations from T1-weighted MRI, amyloid PET, FDG PET and CT. It combines local image features within atlas-defined regions, allowing the same embeddings to be used for downstream prediction and regional analysis.

## Release status

Version 0.2 provides the OmniMIRA v9 inference architecture, its multi-atlas resources and an epoch-1000 checkpoint. It produces anatomically indexed features and is intended for research use, not clinical diagnosis.

## Installation

Requires Python 3.9 or later and PyTorch 2.0 or later.

```bash
pip install omnimira
```

## Usage

The wheel contains the checkpoint, model-space templates and atlas definitions. Input NIfTI images must already be spatially normalized to MNI space. The package resamples them to the committed model grid and applies modality-specific intensity normalization.

```python
from omnimira import from_pretrained

model = from_pretrained(device="cpu")
features = model.extract("scan_mni.nii.gz", modality="t1")
print(features["aal3"].shape)  # (166, 128)
```

The command-line interface writes an NPZ with ROI identifiers, names and features:

```bash
omnimira scan_mni.nii.gz features.npz --modality t1
```

The output contains 128-dimensional embeddings for each region: 166 AAL3 regions, 69 Harvard–Oxford regions and 7 Yeo networks.

The modality options are `t1` (T1-weighted MRI), `av45` (amyloid PET), `fdg` (FDG PET) and `ct` (CT).

## Documentation

- [Input preparation](docs/input_preparation.md)
- [Atlas setup](docs/atlas_setup.md)
- [Output format](docs/output_format.md)
- [Model architecture](docs/model_architecture.md)
- [Pretraining](docs/training.md)

## Tests

Model and loss tests use synthetic inputs:

```bash
pip install pytest
python -m pytest -q tests/test_model_forward.py tests/test_losses.py
```

The full test suite also requires the atlas and template files.

## Questions

For questions about the code, please [open an issue](https://github.com/YixinChen-AI/MIRA-Brain-FM/issues).

## License

Source code is available under the [Apache 2.0 license](LICENSE). The bundled pretrained model weights are licensed separately under [CC BY-NC 4.0](MODEL_LICENSE.md), which does not permit commercial use. Third-party resources have [separate terms](THIRD_PARTY_NOTICES.md).

For research use only; not for clinical diagnosis.
