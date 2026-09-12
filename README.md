# OmniMIRA

### An anatomically indexed foundation model across structural, molecular and metabolic neuroimaging

OmniMIRA learns brain-region representations from T1-weighted MRI, amyloid PET, FDG PET and CT. It combines local image features within atlas-defined regions, allowing the same embeddings to be used for downstream prediction and regional analysis.

## Install

Requires Python 3.9 or later and PyTorch 2.0 or later.

```bash
python -m pip install omnimira==0.2.1
```

## Extract ROI features

The wheel contains the checkpoint, model-space templates and atlas definitions. Input NIfTI images must already be spatially normalized to MNI space. The package resamples them to the committed model grid and applies modality-specific intensity normalization.

```python
from omnimira import from_pretrained

model = from_pretrained(device="cpu")  # use "cuda" when a CUDA GPU is available
features = model.extract("scan_mni.nii.gz", modality="t1")
for atlas, values in features.items():
    print(atlas, values.shape)
```

The command-line interface writes an NPZ with ROI identifiers, names and features:

```bash
omnimira scan_mni.nii.gz features.npz --modality t1 --device cpu
```

The output contains 128-dimensional embeddings for each region: 166 AAL3 regions, 69 Harvard–Oxford regions and 7 Yeo networks. The installed wheel already contains the v9 epoch-1000 checkpoint, atlas definitions and model-space template; no separate model download or source checkout is required.

The modality options are `t1` (T1-weighted MRI), `av45` (amyloid PET), `fdg` (FDG PET) and `ct` (CT).

## Input requirement

Input must be a skull-stripped image spatially normalized to MNI152 space. The package accepts NIfTI (`.nii` or `.nii.gz`) and normalized NumPy (`.npy`) volumes. It resamples NIfTI input to the model grid and performs modality-specific intensity normalization, but it does not perform raw DICOM conversion, skull stripping or nonlinear registration.

## Reference

- [Input preparation](docs/input_preparation.md)
- [Output format](docs/output_format.md)
- [Complete Python example](examples/extract_roi_features.py)

## Questions

For questions, please use the repository issue tracker.

## License

Source code is available under the [Apache 2.0 license](LICENSE). The bundled pretrained model weights are licensed separately under [CC BY-NC 4.0](MODEL_LICENSE.md), which does not permit commercial use. Third-party resources have [separate terms](THIRD_PARTY_NOTICES.md).

For research use only; not for clinical diagnosis.
