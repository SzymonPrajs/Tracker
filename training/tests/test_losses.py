import pytest
import torch

from tracker_training.losses import HCDS31Loss, LossWeights, centernet_focal_loss
from tracker_training.targets import HeadTarget, encode_targets


def test_well_aligned_prediction_has_lower_loss_and_finite_gradients():
    targets = encode_targets([HeadTarget(41.0, 51.0, 30.0, 36.0)])
    good = torch.zeros((1, 16, 40, 72))
    good[:, 0] = -6.0
    anchor = targets["reg_mask"][0].nonzero()[0]
    y, x = anchor.tolist()
    good[0, 0, y, x] = 6.0
    good[0, 1:3, y, x] = targets["offset"][:, y, x]
    good[0, 3:5, y, x] = targets["size"][:, y, x]
    good.requires_grad_()
    bad = torch.zeros_like(good, requires_grad=True)
    criterion = HCDS31Loss()
    good_loss = criterion(good, targets)["loss"]
    bad_loss = criterion(bad, targets)["loss"]
    assert good_loss < bad_loss
    good_loss.backward()
    assert torch.isfinite(good.grad).all()


def test_empty_frame_focal_loss_is_finite():
    logits = torch.zeros(2, 1, 40, 72, requires_grad=True)
    target = torch.zeros_like(logits)
    loss = centernet_focal_loss(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_background_and_encoded_saturation_penalties_are_explicit():
    targets = encode_targets([])
    prediction = torch.zeros((1, 16, 40, 72), requires_grad=True)
    prediction.data[0, 1, 0, 0] = 2.0  # Encodes to 32, beyond S8 exponent -3.
    losses = HCDS31Loss()(prediction, targets)
    assert losses["background"] > 0
    assert losses["saturation"] > 0
    assert LossWeights().background == 0.01
    assert LossWeights().saturation == 0.01
    losses["loss"].backward()
    assert torch.isfinite(prediction.grad).all()


def test_owned_regression_is_not_background_penalized():
    targets = encode_targets([HeadTarget(41.0, 51.0, 30.0, 36.0)])
    prediction = torch.zeros((1, 16, 40, 72))
    y, x = targets["reg_mask"][0].nonzero()[0].tolist()
    prediction[0, 1:5, y, x] = 0.25
    losses = HCDS31Loss()(prediction, targets)
    assert losses["background"].item() == 0.0


def test_config_wires_every_implemented_weight_and_shape_parameter():
    config = {
        "heatmap": 0.9,
        "offset": 1.1,
        "size": 0.2,
        "padding": 0.3,
        "background_regression": 0.04,
        "encoded_saturation": 0.05,
        "decode_consistency": 0.0,
        "focal_alpha": 1.5,
        "focal_beta": 3.5,
        "smooth_l1_beta": 1.0 / 9.0,
    }
    criterion = HCDS31Loss.from_config(config)
    assert criterion.weights == LossWeights(
        heatmap=0.9,
        offset=1.1,
        size=0.2,
        padding=0.3,
        background=0.04,
        saturation=0.05,
    )
    assert criterion.focal_alpha == 1.5
    assert criterion.focal_beta == 3.5
    assert criterion.smooth_l1_beta == pytest.approx(1.0 / 9.0)


def test_smooth_l1_beta_changes_owned_regression_loss():
    targets = encode_targets([HeadTarget(41.0, 51.0, 30.0, 36.0)])
    prediction = torch.zeros((1, 16, 40, 72))
    narrow = HCDS31Loss(smooth_l1_beta=1.0 / 9.0)(prediction, targets)
    wide = HCDS31Loss(smooth_l1_beta=1.0)(prediction, targets)
    assert narrow["offset"] > wide["offset"]
    assert narrow["size"] > wide["size"]


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: LossWeights(background=-0.1), "background"),
        (lambda: HCDS31Loss(focal_alpha=-1.0), "focal_alpha"),
        (lambda: HCDS31Loss(focal_beta=float("inf")), "focal_beta"),
        (lambda: HCDS31Loss(smooth_l1_beta=0.0), "smooth_l1_beta"),
        (
            lambda: HCDS31Loss.from_config(
                {
                    "heatmap": 1.0, "offset": 1.0, "size": 0.15,
                    "padding": 0.1, "background_regression": 0.01,
                    "encoded_saturation": 0.01, "decode_consistency": 0.1,
                    "focal_alpha": 2.0, "focal_beta": 4.0,
                    "smooth_l1_beta": 1.0 / 9.0,
                }
            ),
            "decode_consistency",
        ),
    ],
)
def test_invalid_loss_configuration_fails_closed(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()
