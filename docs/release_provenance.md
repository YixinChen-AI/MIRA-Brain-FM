# Release provenance

**Historical development record.** This document and its V4 contract predate the current manuscript (91,880 scans, 24 tasks, 11 cohorts). The code-only preview does not certify alignment with final experiments. Checkpoint selection, possible retraining and configuration reconciliation are pending. The historical plan below is retained for provenance, not as a statement that a final checkpoint is available.

The formal public checkpoint will be newly trained from
`configs/omnimira_3atlas.yaml` and the paper-defined model, loss and
preprocessing code in this repository. Its SHA256, byte count, license and
release URL remain publication gates.

```yaml release-provenance
status: public_retraining_pending
public_model_version: omnimira-public-v1
conflicts: []
unresolved:
  - formal public checkpoint has not yet been trained
publication_gates:
  - public_checkpoint_training
  - checkpoint_redistribution
  - atlas_redistribution
  - self_contained_result_evidence
  - committed_manuscript_revision
```

## Manuscript contract

- SHA256: `7553dc39bacaac630fe5c48c21c0b2744e58bd8dc77a113bc6fe6703d4a2de98`
- Git blob: `1025a59a891262c54d071ecc96cb1f0b741b0d5a`

The public repository records the V4 architectural contract but does not ship
an unpublished manuscript snapshot. A committed manuscript revision is
required before publication.

The public implementation follows the V4 Methods: MCLP, AART, seven
pretraining objectives, HUW, batch size 64, a `1.5e-4` base learning rate and
ROI-MAE. All modalities use one canonical RAS+ model grid. AAL3 uses dense
indices over its 166 occupied raw labels. AART assigns a patch to each ROI
overlapping its voxel cube, preserving small regions at the 8-voxel patch
resolution.

The paper-aligned public retraining recipe requires a private manifest with
33,309 scans from 9,094 participants across 13 cohorts: 26,307 T1 MRI,
3,726 amyloid-PET, 1,143 FDG-PET and 2,133 CT scans. The new checkpoint must
record matching aggregate provenance before it can clear the publication gate.

## Public result evidence

The checkpoint manifest and four-modality golden references must be generated
after public retraining. No pre-release artifact may be relabeled as an output
of `omnimira-public-v1`; published evaluation evidence must identify the exact
new checkpoint and immutable configuration that produced it.

## Release gate

`scripts/audit_public_release.py --publication` remains intentionally failing
while the checkpoint state is `training_pending`. After training, publish an
immutable HTTPS asset, populate `checkpoints/manifest.json`, regenerate golden
references from the new model with
`scripts/generate_public_golden_reference.py`, and run the complete public
release audit.
