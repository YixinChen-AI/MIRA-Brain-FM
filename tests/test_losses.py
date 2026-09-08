import torch

from omnimira.losses import (
    LOSS_REGISTRY,
    OmniMIRALoss,
    loss_age,
    loss_augmentation_consistency,
    loss_cross_modal,
    loss_patch_reconstruction,
    loss_roi_mae,
    loss_roi_volume,
    loss_spatial_metric,
)


OBJECTIVES = (
    "patch_ae", "roi_mae", "cross_modal", "augmentation_consistency",
    "age", "roi_volume", "spatial_metric",
)


def test_registry_contains_exact_public_objectives():
    assert tuple(LOSS_REGISTRY) == OBJECTIVES


def test_patch_reconstruction_scores_foreground_only():
    pred = torch.tensor([[[[[1.0, 10.0]]]]])
    target = torch.zeros_like(pred)
    loss = loss_patch_reconstruction(
        pred, target,
        foreground_patch_mask=torch.tensor([True, False]),
        patch_size=1,
    )
    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_supervised_losses_apply_explicit_per_sample_validity():
    roi_valid = torch.ones(2, 2, dtype=torch.bool)
    pred_age = torch.tensor([[[10.0], [12.0]], [[100.0], [100.0]]])
    age = loss_age(pred_age, torch.tensor([11.0, 0.0]),
                   torch.tensor([True, False]), roi_valid)
    torch.testing.assert_close(age, torch.tensor(1.0))

    pred_volume = torch.zeros(2, 2, 1)
    target_volume = torch.tensor([[1.0, 3.0], [100.0, 100.0]])
    volume = loss_roi_volume(pred_volume, target_volume,
                             torch.tensor([True, False]), roi_valid)
    torch.testing.assert_close(volume, torch.tensor(2.0))

    coords = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                           [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]]])
    distances = torch.zeros(2, 2, 2)
    spatial = loss_spatial_metric(coords, distances,
                                  torch.tensor([True, False]), roi_valid)
    torch.testing.assert_close(spatial, torch.tensor(1.0))


def test_huw_skips_missing_and_zero_valid_terms_entirely():
    loss_fn = OmniMIRALoss(
        head_names=("age", "roi_volume"), atlas_names=("toy",)
    )
    with torch.no_grad():
        loss_fn.log_sigma2.copy_(torch.tensor([2.0, 4.0]))
    atlas_outputs = {"toy": {
        "roi_valid": torch.ones(1, 2, dtype=torch.bool),
        "head_proj": {
            "age": torch.tensor([[[3.0], [3.0]]]),
            "roi_volume": torch.zeros(1, 2, 1),
        },
    }}
    out = loss_fn(
        atlas_outputs=atlas_outputs,
        meta={"age": torch.tensor([2.0]), "age_valid": torch.tensor([True])},
    )
    expected = 0.5 * torch.exp(torch.tensor(-2.0)) * 1.0 + 1.0
    torch.testing.assert_close(out["total"], expected)
    assert set(out["per_head"]) == {"toy.age"}


def test_roi_mae_target_is_not_a_gradient_path():
    pred = torch.zeros(1, 2, 2, requires_grad=True)
    target = torch.ones(1, 2, 2, requires_grad=True)
    loss = loss_roi_mae(
        pred, target, torch.tensor([[True, False]]),
        torch.ones(1, 2, dtype=torch.bool),
    )
    loss.backward()
    assert pred.grad is not None
    assert target.grad is None


def test_cross_modal_positives_use_same_roi_across_modalities_and_participants():
    tokens = torch.nn.functional.normalize(torch.randn(3, 1, 4), dim=-1)
    valid = torch.ones(3, 1, dtype=torch.bool)
    paired = loss_cross_modal(
        tokens, valid, torch.tensor([0, 1, 1]), tau=0.2
    )
    assert paired is not None and torch.isfinite(paired)
    assert loss_cross_modal(
        tokens, valid, torch.tensor([0, 0, 0]), tau=0.2
    ) is None


def test_augmentation_contrast_is_invariant_to_embedding_scale():
    first = torch.randn(3, 2, 4)
    second = torch.randn(3, 2, 4)
    valid = torch.ones(3, 2, dtype=torch.bool)
    baseline = loss_augmentation_consistency(first, second, valid, tau=0.2)
    scaled = loss_augmentation_consistency(first * 7, second * 0.3, valid, tau=0.2)
    torch.testing.assert_close(baseline, scaled)


def test_huw_allocates_only_supported_atlas_objective_combinations():
    supported = {
        "aal3": ("roi_mae", "cross_modal", "augmentation_consistency", "age",
                 "roi_volume", "spatial_metric"),
        "ho69": ("roi_mae", "cross_modal", "augmentation_consistency", "age",
                 "roi_volume", "spatial_metric"),
        "yeo7": ("roi_mae", "cross_modal", "augmentation_consistency", "age",
                 "roi_volume"),
    }
    loss_fn = OmniMIRALoss(
        head_names=OBJECTIVES, atlas_names=("aal3", "ho69", "yeo7"),
        supported_objectives=supported,
    )
    assert "yeo7.spatial_metric" not in loss_fn.task_keys
    assert len(loss_fn.task_keys) == 18

    outputs = {name: {
        "roi_valid": torch.ones(1, 1, dtype=torch.bool),
        "head_proj": {"age": torch.zeros(1, 1, 1, requires_grad=True)},
    } for name in supported}
    result = loss_fn(
        outputs,
        meta={"age": torch.tensor([2.0]), "age_valid": torch.tensor([True])},
    )
    result["total"].backward()
    active_indices = {loss_fn.task_keys.index(f"{name}.age") for name in supported}
    for index, gradient in enumerate(loss_fn.log_sigma2.grad):
        assert (gradient != 0) == (index in active_indices)
