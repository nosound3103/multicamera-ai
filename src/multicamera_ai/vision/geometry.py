from collections.abc import Sequence
from math import isfinite
from typing import Literal

type Point2D = tuple[float, float]

type CrossingDirection = Literal[
    "negative_to_positive",
    "positive_to_negative",
]

EPSILON = 1e-9


def bottom_center(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> Point2D:
    """Return the bottom-center point of a bounding box."""
    return ((x1 + x2) / 2.0, y2)


def normalized_to_pixel(
    point: Point2D,
    width: int,
    height: int,
) -> tuple[int, int]:
    """Convert normalized coordinates [0, 1] to pixel coordinates."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")

    x, y = point

    if not (isfinite(x) and isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1):
        raise ValueError("Normalized coordinates must be finite and between 0 and 1")

    return (
        round(x * (width - 1)),
        round(y * (height - 1)),
    )


def cross_product(
    a: Point2D,
    b: Point2D,
    p: Point2D,
) -> float:
    """Calculate the signed cross product AB x AP."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def side_of_line(
    a: Point2D,
    b: Point2D,
    p: Point2D,
) -> int:
    """Return -1, 0, or 1 according to the side of line AB."""
    value = cross_product(a, b, p)

    if value > EPSILON:
        return 1

    if value < -EPSILON:
        return -1

    return 0


def crossing_direction(
    previous: Point2D,
    current: Point2D,
    start: Point2D,
    end: Point2D,
) -> CrossingDirection | None:
    """Detect crossing of a finite counting line.

    Args:
        previous: Previous tracked object position.
        current: Current tracked object position.
        start: Counting line starting point.
        end: Counting line ending point.

    Returns:
        Crossing direction or None if no crossing occurred.

    Touching the line or an endpoint is not counted.
    """
    before = side_of_line(
        start,
        end,
        previous,
    )

    after = side_of_line(
        start,
        end,
        current,
    )

    # The object must move between opposite sides.
    if before == 0 or after == 0 or before == after:
        return None

    # Check intersection with the finite line segment.
    start_side = side_of_line(
        previous,
        current,
        start,
    )

    end_side = side_of_line(
        previous,
        current,
        end,
    )

    # Exclude endpoint touching and non-intersections.
    if start_side == 0 or end_side == 0 or start_side == end_side:
        return None

    if before < 0:
        return "negative_to_positive"

    return "positive_to_negative"


def point_in_polygon(
    point: Point2D,
    polygon: Sequence[Point2D],
) -> bool:
    """Determine whether a point is inside a polygon.

    Uses the ray-casting algorithm.
    Points on the polygon boundary count as inside.
    The polygon is expected to be simple (non-self-intersecting).
    """
    if len(polygon) < 3:
        return False

    x, y = point
    inside = False

    for i, a in enumerate(polygon):
        b = polygon[(i + 1) % len(polygon)]

        # Check whether the point lies on an edge.
        cross = cross_product(a, b, point)

        if abs(cross) <= EPSILON:
            if (
                min(a[0], b[0]) - EPSILON <= x <= max(a[0], b[0]) + EPSILON
                and min(a[1], b[1]) - EPSILON <= y <= max(a[1], b[1]) + EPSILON
            ):
                return True

        # Ray-casting intersection test.
        if (a[1] > y) != (b[1] > y):
            intersection_x = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])

            if x < intersection_x:
                inside = not inside

    return inside
