# Third-party resources

The repository code is offered under Apache License 2.0 as stated in
`LICENSE`. The OmniMIRA pretrained checkpoint is offered separately under
CC BY-NC 4.0 as stated in `MODEL_LICENSE.md`. Neither license automatically
grants rights to atlas data, upstream datasets, or other third-party assets.

| Asset | Source/version | SHA256 | Redistribution status |
|---|---|---|---|
| AAL3v1 label volume | AAL3v1, Rolls et al. 2020, NITRC AAL project | `aa44bb1767594f560e811cd917fee5833c2f843b81a8febd614db20284b11a34` | Unverified |
| AAL3v1 labels | internal label export corresponding to AAL3v1 | `5c643d5bef449948af5d92f116cc2eeb670777a500882ce6e523a033837ffd08` | Unverified |
| Harvard-Oxford-69 | derived from FSL Harvard-Oxford cortical and subcortical max-probability atlases through nilearn | `9ad6acc55d44c988bf30e997737d20aa6b94453624f4669bdad39f3a7a104e2e` | FSL documents Harvard--Oxford under CC BY-SA 4.0; attribution and derivative distribution review remains open |
| Yeo-7-Network | Yeo et al. 2011 7-network parcellation obtained through nilearn | `5b13f9eebcfdb103455a4a424f762f936a2df1c4c185ae3726f4eff68e3cf1a7` | Upstream redistribution terms unverified |

Raw third-party atlas images are not distributed in the Python package. The
package contains a compact model-space ROI membership contract and reference
grids needed to reproduce model inference; these do not include source cohort
images or upstream training data.

The file versions required by the implementation are listed in
[Atlas setup](docs/atlas_setup.md). The hashes identify those versions;
they do not grant permission to redistribute the files.
