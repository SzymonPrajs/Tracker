import unittest

import numpy as np

from tracker_training.decode import decode_heatmap, decode_hwc16_c_parity, local_maxima


class DecodeTest(unittest.TestCase):
    def test_top_k_nms_and_row_major_tie(self) -> None:
        heatmap = np.zeros((5, 6), dtype=np.float32)
        heatmap[1, 1] = 0.8
        heatmap[1, 2] = 0.8
        heatmap[4, 5] = 0.9
        peaks = decode_heatmap(heatmap, top_k=2, score_threshold=0.5)
        self.assertEqual([(peak.cell_y, peak.cell_x) for peak in peaks], [(4, 5), (1, 1)])

    def test_offsets_sizes_and_stride(self) -> None:
        heatmap = np.zeros((3, 4), dtype=np.float32)
        heatmap[1, 2] = 1.0
        offsets = np.zeros((3, 4, 2), dtype=np.float32)
        offsets[1, 2] = [0.25, -0.125]
        sizes = np.zeros((2, 3, 4), dtype=np.float32)
        sizes[:, 1, 2] = [40.0, 50.0]
        peak = decode_heatmap(heatmap, offsets=offsets, sizes=sizes, output_stride=4, top_k=1)[0]
        self.assertAlmostEqual(peak.x, 11.0)
        self.assertAlmostEqual(peak.y, 5.5)
        self.assertEqual((peak.width, peak.height), (40.0, 50.0))

    def test_flat_plateau_is_deterministic(self) -> None:
        self.assertEqual(local_maxima(np.ones((3, 3)), score_threshold=1.0), [(0, 0)])
        peak = decode_heatmap(np.ones((3, 3)), output_stride=4, top_k=1)[0]
        self.assertEqual((peak.x, peak.y), (2.0, 2.0))

    def test_c_parity_known_peak(self) -> None:
        head = np.full((3, 3, 16), -128, dtype=np.int8)
        head[1, 1, 0] = 100
        head[1, 1, 1] = 32
        head[1, 1, 2] = -16
        result = decode_hwc16_c_parity(head, output_stride=4, top_k=1)[0]
        self.assertEqual(result.flat_index, 4)
        self.assertEqual(result.x_q16, 7 * 65536)
        self.assertEqual(result.y_q16, (5 * 65536) + 32768)

    def test_c_parity_flat_falls_back_to_first_peak(self) -> None:
        head = np.full((3, 3, 16), -128, dtype=np.int8)
        head[0, 0, 1:3] = 0
        result = decode_hwc16_c_parity(head, output_stride=4, top_k=1)[0]
        self.assertEqual(result.flat_index, 0)
        self.assertEqual(result.x_q16, 2 * 65536)
        self.assertEqual(result.y_q16, 2 * 65536)

    def test_rejects_non_finite_heatmap(self) -> None:
        with self.assertRaises(ValueError):
            decode_heatmap(np.asarray([[np.nan]]))


if __name__ == "__main__":
    unittest.main()
