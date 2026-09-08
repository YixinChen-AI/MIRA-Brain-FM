# Checkpoint

The formal public checkpoint will be newly pretrained from the paper-defined
implementation and recipe in this repository.

```json release-metadata
{
  "version": "omnimira-public-v1",
  "filename": "omnimira_public_v1.pt",
  "sha256": null,
  "bytes": null,
  "model_config": "configs/omnimira_3atlas.yaml",
  "manuscript_source": "pending committed V4 manuscript revision",
  "state": "training_pending"
}
```

After training, the final byte count, SHA256, license and HTTPS release URL are
written to `manifest.json`. `fetch_checkpoint.py --release` rejects this
placeholder state.
