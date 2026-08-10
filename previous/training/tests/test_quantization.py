from tracker_training.quantization import representative_indices, scale_exponent, stratum


def row(image_id, source, sizes):
    return {
        "image_id": image_id,
        "source": source,
        "heads": [
            {"ignore": False, "bbox_cache_xywh": [0, 0, width, height]}
            for width, height in sizes
        ],
    }


def test_representative_indices_are_repeatable_and_span_strata():
    rows = [
        row("a", "near", []),
        row("b", "near", [(60, 60)]),
        row("c", "far", [(8, 8)] * 12),
        row("d", "far", [(18, 18)] * 3),
    ]
    first = representative_indices(rows, 4, seed=7)
    assert first == representative_indices(rows, 4, seed=7)
    assert len({stratum(rows[index]) for index in first}) == 4


def test_scale_exponent_rejects_non_power_of_two():
    assert scale_exponent(1 / 128) == -7
    try:
        scale_exponent(0.01)
    except RuntimeError:
        pass
    else:
        raise AssertionError("non-power-of-two scale accepted")
