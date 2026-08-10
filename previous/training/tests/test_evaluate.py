import numpy as np

from evaluate import average_precision, box_iou


def test_box_iou_and_average_precision_are_exact_for_perfect_predictions():
    ground_truth = {
        0: np.asarray([[0, 0, 10, 10]], dtype=np.float32),
        1: np.asarray([[5, 5, 15, 15]], dtype=np.float32),
    }
    predictions = [
        (0, 0.9, np.asarray([0, 0, 10, 10], dtype=np.float32)),
        (1, 0.8, np.asarray([5, 5, 15, 15], dtype=np.float32)),
    ]
    assert np.allclose(box_iou(predictions[0][2], ground_truth[0]), [1.0])
    assert average_precision(predictions, ground_truth, 0.5) == 1.0


def test_duplicate_detection_is_a_false_positive():
    ground_truth = {0: np.asarray([[0, 0, 10, 10]], dtype=np.float32)}
    predictions = [
        (0, 0.9, np.asarray([0, 0, 10, 10], dtype=np.float32)),
        (0, 0.8, np.asarray([0, 0, 10, 10], dtype=np.float32)),
    ]
    # Recall is already complete before the duplicate, so interpolated AP stays one.
    assert average_precision(predictions, ground_truth, 0.5) == 1.0
