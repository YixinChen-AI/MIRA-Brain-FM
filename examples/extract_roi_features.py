"""Extract the exact versioned OmniMIRA ROI feature archive."""
from __future__ import annotations

import argparse

import torch

from omnimira import (
    extract_roi_features,
    load_omnimira,
    load_volume,
    save_roi_features,
    source_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="strict model-space .nii/.nii.gz or .npy")
    parser.add_argument("--modality", required=True, choices=("t1", "av45", "fdg", "ct"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    volume = load_volume(args.input, modality=args.modality)
    metadata = source_metadata(args.input, args.modality)
    model = load_omnimira(args.checkpoint, args.config, device=args.device)
    features = extract_roi_features(model, volume, modality=args.modality)
    save_roi_features(features, args.output, metadata)

    print(f"saved {args.output}")
    for name, value in features.items():
        print(f"{name}: {value.shape}")


if __name__ == "__main__":
    main()
