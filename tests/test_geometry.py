"""Unit tests for geometry utilities.

These tests do not require YOLO or OpenCV.
"""

import pytest

from multicamera_ai.vision.geometry import (
    bottom_center,
    cross_product,
    crossing_direction,
    normalized_to_pixel,
    point_in_polygon,
    side_of_line,
)


def test_bottom_center():
    point = bottom_center(100, 200, 300, 500)

    assert point == (200, 500)


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((0.0, 0.0), (0, 0)),
        ((1.0, 1.0), (1919, 1079)),
        ((0.5, 0.5), (960, 540)),
    ],
)
def test_normalized_to_pixel(point, expected):
    result = normalized_to_pixel(point, 1920, 1080)

    assert result == expected


@pytest.mark.parametrize(
    "width,height",
    [
        (0, 1080),
        (1920, 0),
        (-1, 200),
    ],
)
def test_reject_invalid_image_dimensions(width, height):
    with pytest.raises(ValueError, match="dimensions"):
        normalized_to_pixel(
            (0.5, 0.5),
            width,
            height,
        )


@pytest.mark.parametrize(
    "point",
    [
        (-0.1, 0.5),
        (0.5, 1.2),
        (float("nan"), 0.5),
    ],
)
def test_reject_invalid_normalized_coordinates(point):
    with pytest.raises(ValueError, match="Normalized"):
        normalized_to_pixel(point, 640, 360)


def test_cross_product_and_side_of_line():
    a = (0, 5)
    b = (10, 5)

    assert cross_product(a, b, (5, 3)) == -20
    assert cross_product(a, b, (5, 7)) == 20

    assert side_of_line(a, b, (5, 3)) == -1
    assert side_of_line(a, b, (5, 7)) == 1
    assert side_of_line(a, b, (5, 5)) == 0


def test_line_crossing_negative_to_positive():
    direction = crossing_direction(
        previous=(5, 3),
        current=(5, 7),
        start=(0, 5),
        end=(10, 5),
    )

    assert direction == "negative_to_positive"


def test_line_crossing_positive_to_negative():
    direction = crossing_direction(
        previous=(5, 7),
        current=(5, 3),
        start=(0, 5),
        end=(10, 5),
    )

    assert direction == "positive_to_negative"


def test_movement_same_side_does_not_cross():
    direction = crossing_direction(
        previous=(5, 2),
        current=(5, 3),
        start=(0, 5),
        end=(10, 5),
    )

    assert direction is None


def test_crossing_outside_finite_segment_not_counted():
    direction = crossing_direction(
        previous=(15, 3),
        current=(15, 7),
        start=(0, 5),
        end=(10, 5),
    )

    assert direction is None


@pytest.mark.parametrize(
    ("previous", "current"),
    [
        ((5, 3), (5, 5)),
        ((5, 5), (5, 7)),
        ((0, 3), (0, 7)),
        ((5, 5), (7, 5)),
    ],
)
def test_touching_line_or_endpoint_not_counted(
    previous,
    current,
):
    direction = crossing_direction(
        previous,
        current,
        (0, 5),
        (10, 5),
    )

    assert direction is None


SQUARE = [
    (0, 0),
    (10, 0),
    (10, 10),
    (0, 10),
]


@pytest.mark.parametrize(
    "point",
    [
        (5, 5),
        (0, 5),
        (0, 0),
        (10, 10),
    ],
)
def test_point_in_or_on_polygon(point):
    assert point_in_polygon(point, SQUARE)


@pytest.mark.parametrize(
    "point",
    [
        (-1, 5),
        (15, 5),
        (5, 11),
    ],
)
def test_point_outside_polygon(point):
    assert not point_in_polygon(point, SQUARE)


def test_concave_polygon():
    polygon = [
        (0, 0),
        (4, 0),
        (4, 4),
        (2, 2),
        (0, 4),
    ]

    assert point_in_polygon((2, 1), polygon)
    assert not point_in_polygon((2, 3), polygon)


def test_polygon_with_too_few_points():
    polygon = [
        (0, 0),
        (2, 0),
    ]

    assert point_in_polygon((1, 1), polygon) is False
