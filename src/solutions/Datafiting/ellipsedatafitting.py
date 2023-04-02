import statistics
from typing import Set, List

import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass
from matplotlib import pyplot as plt
from skspatial.objects import Point
from src.solutions import ISolution
from src.utils.geometry import get_plain_and_transform_by_3_points, get_ellipse_from_points


@dataclass(frozen=True)
class EllipsePoint:
    x: float
    y: float
    t: float

    def distance(self, other):
        return np.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.t - other.t) ** 2
        )

    @staticmethod
    def from_numpy(arr: np.ndarray):
        return EllipsePoint(x=arr[0], y=arr[1], t=arr[2])

    def to_numpy(self):
        return np.array([self.x, self.y, self.t])


@dataclass
class Ellipse:
    points: Set[EllipsePoint]
    center: EllipsePoint


class EllipseDataFitting(ISolution):

    def __init__(self, use_tqdm=True, eps_proj=1e-4, eps_ellipse=1e-4, center_parameter=5, save_graphics=False,
                 minimal_points_in_plain=10):
        self.save_graphics = save_graphics
        self.eps_proj = eps_proj
        self.eps_ellipse = eps_ellipse
        self.use_tqdm = use_tqdm
        self.center_parameter = center_parameter
        self.minimal_points_in_plain = minimal_points_in_plain
        if self.use_tqdm:
            from tqdm import tqdm
            self.tqdm = tqdm
        else:
            self.tqdm = (lambda el: el)

    @staticmethod
    def _ellipse_function(A, B, C, D, E):
        return np.vectorize(lambda x, y: A * x * x + B * x * y + C * y * y + D * x + E * y - 1)

    def _get_points(self, df_input: pd.DataFrame):
        points = df_input.apply(lambda point: Point([point.x_c, point.y_c, point.t_c]), axis=1).to_numpy()

        for i1 in self.tqdm(range(len(points) - 2)):
            for i2 in range(i1 + 1, len(points) - 1):
                for i3 in range(i2 + 1, len(points)):
                    p1 = points[i1]
                    p2 = points[i2]
                    p3 = points[i3]
                    try:
                        plane, transf, inv_transf = get_plain_and_transform_by_3_points(p1, p2, p3)
                    except ValueError as e:
                        continue
                    cur_points = []
                    not_projected = []
                    for i4, p4 in enumerate(points):
                        # if i4 in [i1, i2, i3]:
                        #     continue

                        projected_p4 = plane.project_point(p4)
                        if np.abs(np.linalg.norm(projected_p4 - p4)) > self.eps_proj:
                            continue
                        not_projected.append(p4)
                        cur_points.append(np.array(projected_p4))

                    if len(cur_points) < self.minimal_points_in_plain:
                        continue
                    cur_points = np.array(cur_points)
                    yield not_projected, cur_points, transf, inv_transf

    def run(self, df_input: pd.DataFrame):
        ellipses: List[Ellipse] = []
        max_cur_points_in_plane = 0
        for base_points, points, transf, inv_transf in self._get_points(df_input):
            points_in_plane = transf(points)

            x, y = points_in_plane[:2, :]
            max_cur_points_in_plane = max(len(x), max_cur_points_in_plane)
            try:
                nzs = cv2.findNonZero(np.array([x, y]))
                elips = cv2.fitEllipse(nzs)
                # center_in_plane, (A, B, C, D, E), vectors = get_ellipse_from_points(x, y)
            except np.linalg.LinAlgError as e:
                print(e, max_cur_points_in_plane)
                continue
            except ValueError as e:
                print(e, max_cur_points_in_plane)
                continue
            # ellipse_func = self._ellipse_function(A, B, C, D, E)
            # r = ellipse_func(x, y)
            # if np.max(np.abs(r)) > self.eps_ellipse:
            #     continue
            ellipse = Ellipse(
                set(EllipsePoint.from_numpy(p) for p in base_points),
                EllipsePoint.from_numpy(inv_transf([*elips[0], 0])))
            ellipses.append(ellipse)
        print(max_cur_points_in_plane)
        if len(ellipses) == 0:
            return []
        # res_points = set()
        # center_0 = statistics.median([el.center.x for el in ellipses])
        # center_1 = statistics.median([el.center.y for el in ellipses])
        # center_2 = statistics.median([el.center.t for el in ellipses])
        # center = EllipsePoint(center_0, center_1, center_2)
        #
        # for el in self.tqdm(ellipses):
        #     if center.distance(el.center) < self.center_parameter:
        #         if not res_points:
        #             res_points = el.points
        #         res_points = res_points.union(el.points)
        return ellipses
