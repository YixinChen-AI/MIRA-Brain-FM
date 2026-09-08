# Model architecture

OmniMIRA has two main components.

**Modality-Conditioned Local Patch Encoder (MCLP).** Four modality-specific 3D
patch embedding layers map scans to a shared 128-dimensional patch space.
Fixed 3D positional encoding and modality-conditioned residual MLP blocks act
independently at each patch position, without cross-patch mixing.

**Anatomy-Anchored ROI Tokenizer (AART).** Each atlas is resampled to the same
RAS+ model grid. A patch is associated with every ROI that overlaps its voxel
cube, allowing a patch to contribute to more than one region. A
learned scalar attention score aggregates the associated patches within each
ROI. The included configuration produces 166 AAL3, 69 Harvard-Oxford and 7 Yeo
ROI embeddings.

The pretraining-only ROI-MAE decoder receives atlas-specific ROI identity
embeddings. Frozen feature extraction bypasses all reconstruction and
supervision heads.
