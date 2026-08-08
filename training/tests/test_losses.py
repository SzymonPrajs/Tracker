import torch

from tracker_training.losses import HCDS31Loss, centernet_focal_loss
from tracker_training.targets import HeadTarget, encode_targets


def test_well_aligned_prediction_has_lower_loss_and_finite_gradients():
    targets = encode_targets([HeadTarget(41.0, 51.0, 30.0, 36.0)])
    good = torch.full((1, 16, 40, 72), -6.0)
    good[:, 5:] = 0.0
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
