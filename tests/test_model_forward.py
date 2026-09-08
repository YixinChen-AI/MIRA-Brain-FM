import torch

from omnimira.models.omnimira import (
    ACTIVE_HEAD_NAMES,
    AART,
    HeadSpec,
    MLPBlockCond,
    OmniMIRA,
)


OBJECTIVES = (
    "patch_ae",
    "roi_mae",
    "cross_modal",
    "augmentation_consistency",
    "age",
    "roi_volume",
    "spatial_metric",
)


def _heads():
    return [
        HeadSpec("patch_ae", "patch_recon", 0),
        HeadSpec("roi_mae", "roi_mae", 0),
        HeadSpec("cross_modal", "contrast", 8),
        HeadSpec("augmentation_consistency", "augmentation", 0),
        HeadSpec("age", "regression", 1),
        HeadSpec("roi_volume", "roi_volume", 1),
        HeadSpec("spatial_metric", "spatial", 3),
    ]


def _tiny_model():
    return OmniMIRA(
        img_size=(16, 16, 16), patch_size=8, embed_dim=12,
        mlp_depth=2, mlp_ratio=2.0, n_modalities=4, n_rois=3,
        contrastive_dim=8, mask_ratio=0.9, decoder_depth=1,
        decoder_heads=3, head_specs=_heads(),
        atlas_n_rois={"toy": 3}, pool_kind="attention", pool_k=1,
    )


def test_public_contract_has_seven_paper_objectives():
    assert ACTIVE_HEAD_NAMES == OBJECTIVES


def test_mclp_block_cannot_mix_patch_positions():
    block = MLPBlockCond(dim=12, mlp_ratio=2.0, n_modalities=4).eval()
    modality = torch.tensor([0])
    baseline = torch.zeros(1, 4, 12)
    changed = baseline.clone()
    changed[:, 2] = 1
    with torch.no_grad():
        delta = block(changed, modality) - block(baseline, modality)
    assert torch.count_nonzero(delta[:, [0, 1, 3]]) == 0
    assert torch.count_nonzero(delta[:, 2]) > 0


def test_aart_uses_one_scalar_attention_score_per_patch():
    pool = AART(embed_dim=2)
    assert pool.score.out_features == 1
    with torch.no_grad():
        pool.score.weight.copy_(torch.tensor([[1.0, 0.0]]))
    patches = torch.tensor([[[0.0, 2.0], [1.0, 4.0], [9.0, 9.0]]])
    tokens, valid = pool(patches, torch.tensor([0, 0, -1]), n_rois=2)
    weights = torch.softmax(torch.tensor([0.0, 1.0]), dim=0)
    expected = weights[0] * patches[0, 0] + weights[1] * patches[0, 1]
    torch.testing.assert_close(tokens[0, 0], expected)
    assert valid.tolist() == [[True, False]]


def test_aart_allows_boundary_patch_to_contribute_to_each_overlapping_roi():
    pool = AART(embed_dim=2)
    with torch.no_grad():
        pool.score.weight.zero_()
    patches = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    membership = torch.tensor([[True, False], [True, True]])
    tokens, valid = pool(patches, membership, n_rois=2)
    torch.testing.assert_close(tokens[0, 0], patches[0, 0])
    torch.testing.assert_close(tokens[0, 1], patches[0].mean(dim=0))
    assert valid.all()


def test_aart_densely_remaps_all_166_occupied_aal3_labels():
    raw_zero_based = torch.tensor([0, 33, 36, 166, 167, 168, 169, -1])
    dense = AART.dense_roi_indices(raw_zero_based, n_rois=166)
    assert dense.tolist() == [0, 33, 34, 162, 163, 164, 165, -1]


def test_roi_mae_masks_ninety_percent_retains_identity_and_detaches_target():
    torch.manual_seed(4)
    model = _tiny_model().train()
    x = torch.randn(1, 1, 16, 16, 16, requires_grad=True)
    mapping = torch.tensor([0, 0, 1, 1, 2, 2, 2, 2])
    out = model(x, torch.tensor([0]), {"toy": mapping})["atlases"]["toy"]
    assert out["roi_mask"].sum().item() == 3
    assert out["roi_mae_pred"].shape == out["roi_tokens"].shape
    assert not out["roi_mae_target"].requires_grad
    assert model.roi_mae_decoder.roi_identity["toy"].shape == (1, 3, 12)


def test_release_dimensions_and_three_atlas_forward_shapes():
    model = OmniMIRA().eval()
    assert model.public_component_names == ("MCLP", "AART")
    assert (model.img_size, model.patch_size, model.embed_dim) == (
        (96, 112, 96), 8, 128
    )
    assert len(model.patch_embed.convs) == 4
    assert not hasattr(model, "decoder")
    assert not hasattr(model, "patch_mae_decoder")
