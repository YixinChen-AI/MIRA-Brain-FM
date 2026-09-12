"""Extract ROI features with the checkpoint bundled in the PyPI package."""
from __future__ import annotations

import argparse

import torch
from omnimira import from_pretrained


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="MNI-normalized .nii/.nii.gz or normalized .npy")
    parser.add_argument("--modality", required=True, choices=("t1", "av45", "fdg", "ct"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = from_pretrained(device=args.device)
    model.save(args.input, args.output, modality=args.modality)

    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
