# Atlas setup

OmniMIRA extracts a fixed set of AAL3v1, Harvard--Oxford-69 and Yeo-7 ROI
features. The implementation requires the exact three atlas files
listed below. An atlas with the same name but different voxel grid, label map
or hash is not interchangeable.

## External atlas directory

When `OMNIMIRA_ATLAS_DIR` is set, feature extraction and public pretraining
use only files from that directory. The directory must contain these exact
filenames:

| File | SHA256 | ROIs |
|---|---|---:|
| `AAL3v1_1mm.nii` | `aa44bb1767594f560e811cd917fee5833c2f843b81a8febd614db20284b11a34` | 166 |
| `harvard_oxford_69_mni_1mm.nii.gz` | `9ad6acc55d44c988bf30e997737d20aa6b94453624f4669bdad39f3a7a104e2e` | 69 |
| `yeo7_mni_1mm.nii.gz` | `5b13f9eebcfdb103455a4a424f762f936a2df1c4c185ae3726f4eff68e3cf1a7` | 7 |

It must also contain the matching ROI label tables, which preserve the names
written into the feature archive:

| File | SHA256 |
|---|---|
| `AAL3v1_labels.txt` | `5c643d5bef449948af5d92f116cc2eeb670777a500882ce6e523a033837ffd08` |
| `HarvardOxford69_labels.txt` | `6ce228f7e86ac10838562b9c39301b0bb0ffa54678edf1bf7c91ad2fefb9c4d7` |
| `Yeo7_labels.txt` | `ca23ed960895a5b847603672ef2110b4e8ff13b0552359920b18fa58abdf48b8` |

The atlas files are not bundled; acquisition and preparation instructions are
still being completed. If you already have the matching files, verify them with:

```bash
export OMNIMIRA_ATLAS_DIR=/absolute/path/to/omnimira-atlases
python scripts/verify_atlases.py --atlas-dir "$OMNIMIRA_ATLAS_DIR"
```

The command prints the resolved paths, SHA256 values and ROI counts. It fails
before feature extraction or pretraining if a file is missing or differs by
even one byte.

## Redistribution status

Atlas files are subject to their upstream licenses. See
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) for attribution and
redistribution status.
