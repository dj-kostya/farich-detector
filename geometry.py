import numpy as np
from skspatial.objects import Plane, Point
from skspatial.plotting import plot_3d


def get_plain_and_transform(point1, point2, point3):
    p = Plane.from_points(point1, point2, point3)
    return p, lambda x: np.linalg.inv(np.column_stack((point2 - point1, point3 - point1, p.normal))).dot(
        x - point1), lambda x: np.column_stack((point2 - point1, point3 - point1, p.normal)).dot(x) + point1


def fit_ellipse_v2(x, y):
    D =  np.column_stack((x**2, x*y, y**2, x, y))
    C = np.ones_like(x)
    return np.linalg.inv(D.T.dot(D)).dot(D.T).dot(C) # МНК для СЛАУ http://aco.ifmo.ru/el_books/numerical_methods/lectures/glava4.html


def get_elipse_from_points(x,y):
    A, B, C, D, E = fit_ellipse_v2(x, y)
    if B**2 - 4*A*C >= 0:
        raise ValueError("Not elipse")
    x0 = (2*C*D - E*B) / (B*B-4*A*C)
    y0 = (2*A*E - B*D) / (B*B-4*A*C)
    centr = [x0, y0]
    Form = np.array([[A, B/2],[B/2, C]])
    I3 = np.linalg.det(np.array([[A, B/2, D/2], [B/2, C, E/2], [D/2, E/2, -1]]))
    I2 = np.linalg.det(Form)
    (alpha1, alpha2), (v1, v2) = np.linalg.eigh(Form, UPLO='U')
    if alpha1 == 0 or alpha2 == 0 or I2 == 0:
        raise ValueError("Bad invariants")
    a=np.sqrt(-I3/(alpha1*I2))
    b=np.sqrt(-I3/(alpha2*I2))

    if np.isnan(a) or np.isnan(b):
        raise ValueError("Bad invariants")
    v1_norm = (v1 * a) / np.linalg.norm(v1)
    v2_norm = (v2 * b) / np.linalg.norm(v2)
    return [x0, y0, 0], (A, B, C, D, E), (v1_norm+centr, v2_norm+centr)


if __name__ == '__main__':
    plane, transform = get_plain_and_transform(Point([-3, 2, -1]), Point([-1, 2, 4]), Point([3, 3, -1]))
    print(plane)

    # p = Plain(Point(-3, 2, -1), Point(-1, 2, 4), Point(3, 3, -1))
    # print(p)
    # print(p.vectors)
    # print(p.T)
    # print(p.get_coord_point_in_plane(Point(-3, 2, -1)))
# TODO: по 5 точкам получить плоскость через МНК, посмотреть точность, потом по этим 5 точкам построить элипс