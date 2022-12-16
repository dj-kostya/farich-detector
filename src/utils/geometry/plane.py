import numpy as np
from skspatial.objects import Plane
from collections.abc import Callable
from typing import Tuple


def get_plain_and_transform_by_3_points(point1, point2, point3) -> \
        Tuple[Plane, Callable[np.ndarray, np.ndarray], Callable[np.ndarray, np.ndarray]]:
    """
    Fit plane from three points
    :param point1: Point1
    :param point2: Point2
    :param point3: Point3
    :return:
    Plane - plane object
    Transform - transform to plane coordinate system
    InvertedTransform - transform from plane coordinate system to base coordinate system
    """
    p = Plane.from_points(point1, point2, point3)
    T = np.column_stack((point2 - point1, point3 - point1, p.normal))
    return (p,
            lambda x: np.linalg.inv().dot(x - point1),
            lambda x: T.dot(x) + point1
            )
