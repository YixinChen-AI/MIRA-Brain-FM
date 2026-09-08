# Input preparation

Inputs must first be spatially standardized to MNI152 space. The loader
then resamples every T1 MRI, amyloid-PET, FDG-PET and CT scan in world
coordinates to one versioned `96 x 112 x 96` RAS+ reference grid. It does not
resize arrays by shape alone.

Continuous images use trilinear interpolation. Atlas labels use nearest
neighbour interpolation. Modality-specific normalization is applied after
resampling: T1 1st-99th percentile scaling, cerebellar-reference PET SUVR
scaling and CT 0-80 HU scaling.

Normalized NumPy inputs are accepted only with the exact `normalized_v2` JSON
sidecar described by `configs/input_contract.yaml`. This records the model
space, affine, resource hashes, modality and normalization rule.
