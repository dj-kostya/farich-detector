import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skspatial.objects import Point
from src.solutions import ISolution
from src.utils.geometry import get_plain_and_transform_by_3_points, get_ellipse_from_points


def ellipse(A, B, C, D, E):
    return np.vectorize(lambda x, y: A * x * x + B * x * y + C * y * y + D * x + E * y - 1)


class EllipseDataFitting(ISolution):
    def __init__(self, use_tqdm=True, eps=1e-4, save_graphics=False):
        super(EllipseDataFitting, self).__init__()
        self.save_graphics = save_graphics
        self.eps = eps
        self.use_tqdm = use_tqdm

    def run(self, df_input: pd.DataFrame):
        df = df_input.copy(deep=True)
        points = df.apply(lambda x: Point([x.x_c, x.y_c, x.t_c]), axis=1).to_numpy()
        if self.use_tqdm:
            from tqdm import tqdm
        else:
            tqdm = (lambda x: x)

        vectors_all = []
        centers = []
        normals = []
        point1 = []
        point2 = []
        point3 = []
        for i1 in tqdm(range(len(points) - 2)):
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
                    for i4, p4 in enumerate(points):
                        # if i4 in [i1, i2, i3]:
                        #     continue
                        projected_p4 = plane.project_point(p4)
                        if np.abs(np.linalg.norm(projected_p4 - p4)) > self.eps:
                            continue
                        cur_points.append(np.array(projected_p4))

                    if len(cur_points) < 5:
                        continue
                    cur_points = np.array(cur_points)
                    # a = cur_points - p1
                    # b = cur_points.T - p1
                    points_in_plane = transf(cur_points)
                    x, y = points_in_plane[:2, :]
                    try:
                        center_in_plane, (A, B, C, D, E), vectors = get_ellipse_from_points(x, y)
                    except np.linalg.LinAlgError as e:
                        continue
                    except ValueError as e:
                        continue
                    ellipse_func = ellipse(A, B, C, D, E)
                    r = ellipse_func(x, y)
                    if np.max(np.abs(r)) > self.eps:
                        continue
                    v1 = vectors[0]
                    v2 = vectors[1]
                    if self.save_graphics:
                        fig, ax = plt.subplots(nrows=1, ncols=1)
                        ax.scatter(x, y)
                        ax.set_xlabel('x')
                        ax.set_ylabel('y')
                        ax.plot([center_in_plane[0], vectors[0][0]], [center_in_plane[1], vectors[0][1]])
                        ax.plot([center_in_plane[0], vectors[1][0]], [center_in_plane[1], vectors[1][1]])
                        ax.scatter(*center_in_plane)

                        neg_v1 = 2 * center_in_plane[0] - (vectors[0])
                        neg_v2 = 2 * center_in_plane[1] - (vectors[1])

                        left_x = min(v1[0], v2[0], neg_v1[0], neg_v2[0]) - 10
                        right_x = max(v1[0], v2[0], neg_v1[0], neg_v2[0]) + 10
                        left_y = min(v1[1], v2[1], neg_v1[1], neg_v2[1]) - 10
                        right_y = max(v1[1], v2[1], neg_v1[1], neg_v2[1]) + 10
                        x_el = np.arange(left_x, right_x, 0.1)
                        y_el = np.arange(left_y, right_y, 0.1)
                        X, Y = np.meshgrid(x_el, y_el)
                        ax.contour(x_el, y_el, ellipse(A, B, C, D, E)(X, Y))

                        fig.savefig(f'tmp/{i1}_{i2}_{i3}.png')
                        plt.close(fig)
                    napr1 = np.array([*v1, 0])
                    napr2 = np.array([*v2, 0])

                    vectors_all.append((inv_transf(napr1), inv_transf(napr2)))
                    centers.append(inv_transf(center_in_plane))
                    normals.append(plane.normal)
                    point1.append(p1)
                    point2.append(p2)
                    point3.append(p3)
                    return centers, vectors_all, normals, point1, point2, point3

        return centers, vectors_all, normals, point1, point2, point3
