import numpy as np
from skspatial.objects import Plane
# from collections.abc import Callable
from typing import Tuple


def get_plain_and_transform_by_3_points(point1, point2, point3) -> \
        Tuple[Plane, any, any]:
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
    new_O = (point2 + point1) / 2
    v1 = point2 - new_O
    v2 = point3 - new_O
    v3 = p.normal + new_O

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = v3 / np.linalg.norm(v3)

    T = np.column_stack((v1, v2, v3))
    T_inv = np.linalg.inv(T)
    return (p,
            lambda x: T_inv.dot((x - new_O).T),
            lambda x: T.dot(x) + np.array(new_O)
            )
