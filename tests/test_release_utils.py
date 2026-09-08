from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.mark.parametrize("wrapper", ["model", "state_dict", "model_state_dict"])
def test_extract_checkpoint_state_supports_wrappers(wrapper: str) -> None:
    from release_utils import extract_checkpoint_state

    state = {"weight": torch.ones(2)}
    assert extract_checkpoint_state({wrapper: state}) is state


def test_extract_checkpoint_state_supports_bare_state() -> None:
    from release_utils import extract_checkpoint_state

    state = {"weight": torch.ones(2), "bias": torch.zeros(2)}
    assert extract_checkpoint_state(state) is state


@pytest.mark.parametrize(
    "checkpoint",
    [[], {}, {"epoch": 1}, {"model": 3}, {"state_dict": {"weight": "bad"}}],
)
def test_extract_checkpoint_state_rejects_missing_or_wrong_types(checkpoint: object) -> None:
    from release_utils import extract_checkpoint_state

    with pytest.raises(ValueError):
        extract_checkpoint_state(checkpoint)


def test_resolve_path_uses_explicit_base(tmp_path: Path) -> None:
    from release_utils import resolve_path

    assert resolve_path("evidence/result.json", base=tmp_path) == (
        tmp_path / "evidence" / "result.json"
    ).resolve()


@pytest.mark.parametrize("filename", ["../model.pt", "subdir/model.pt", "subdir\\model.pt", ".", ".."])
def test_safe_filename_rejects_path_escape(filename: str) -> None:
    from release_utils import safe_filename

    with pytest.raises(ValueError, match="filename"):
        safe_filename(filename)


def test_fetch_rejects_https_redirect_to_http(tmp_path: Path) -> None:
    import fetch_checkpoint

    payload = b"checkpoint"
    record = {
        "filename": "model.pt",
        "url": "https://example.org/model.pt",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def geturl(self) -> str:
            return "http://cdn.example.org/model.pt"

    with pytest.raises(ValueError, match="final URL must use HTTPS"):
        fetch_checkpoint.download_checkpoint(
            record,
            tmp_path,
            opener=lambda _url: Response(payload),
        )


def test_redirect_handler_rejects_https_http_https_multihop_downgrade() -> None:
    import urllib.request

    import fetch_checkpoint

    handler = fetch_checkpoint.HTTPSOnlyRedirectHandler()
    request = urllib.request.Request("https://origin.example/model.pt")
    redirect_chain = [
        "http://middle.example/model.pt",
        "https://cdn.example/model.pt",
    ]
    with pytest.raises(ValueError, match="redirect target must use HTTPS"):
        for target in redirect_chain:
            request = handler.redirect_request(
                request,
                None,
                302,
                "Found",
                {},
                target,
            )


@pytest.mark.parametrize(
    "payload",
    [None, [], {"checkpoints": []}, {"checkpoints": [None]}],
)
def test_manifest_loader_rejects_malformed_shapes(tmp_path: Path, payload: object) -> None:
    from release_utils import load_checkpoint_manifest

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        load_checkpoint_manifest(path)


@pytest.mark.parametrize("payload", ["null\n", "- item\n"])
def test_yaml_loader_rejects_non_mapping_root(tmp_path: Path, payload: str) -> None:
    from release_utils import load_yaml_mapping

    path = tmp_path / "contract.yaml"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml_mapping(path)
