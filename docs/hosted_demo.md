# OmniMIRA sample demo

A Gradio app that serves **precomputed** sample cases instantly — input slice
preview, the per-ROI feature heatmap, the top-ROI table, and the downloadable
3-atlas frozen features. The app loads **no model weights**; every case is read
from `demo/sample_cases/<name>/`, generated offline.

The release bundles no sample cases. Add only a redistributable case generated
from an exact contract-valid input.

> Research use only. Not a clinical or diagnostic tool. Single-subject ROI
> features are exploratory representations.

## Run the app

```bash
pip install -e .[demo]
python demo/app.py            # http://127.0.0.1:7860
```

## Generate a sample case

Sample cases are produced offline by running the pretrained checkpoint once on
an input that already satisfies the exact modality-specific contract:

```bash
python demo/generate_sample_cases.py \
    --ckpt /path/to/omnimira_3atlas.pt \
    --config configs/omnimira_3atlas.yaml \
    --input prepared_t1.npy --modality t1 --name prepared_t1
```

To add your own already-MNI volume as a case:

```bash
python demo/generate_sample_cases.py \
    --ckpt /path/to/omnimira_3atlas.pt --input scan_mni.nii.gz \
    --modality t1 --name my_case
```

Each case writes: `input_slices.png`, `roi_heatmap.png`, `top_roi.csv`,
`frozen_features.npz`, `meta.json`.

## Privacy

Only redistributable, contract-valid inputs should be turned into committed
sample cases. Do not commit cases derived from restricted cohorts (e.g. ADNI);
their data-use agreements prohibit redistribution of identifiable scans. A
public template is not accepted merely because it is called MNI152: it must
first be prepared to the exact modality-specific grid and contract.

## Checkpoint

The checkpoint that generates real cases must match `configs/omnimira_3atlas.yaml`
exactly (`strict=True` load). It is hosted separately and is not committed to the
repo; the precomputed sample cases let the app run without it.
# Availability

No hosted demo is currently published. The instructions below are development notes. A working checkpoint and resource setup must be verified before deployment. Follow the root TODO.md for release progress.
