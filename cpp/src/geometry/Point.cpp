//
// Created by Константин Носорев on 06.05.2023.
//

#include "Point.h"

#include <utility>

Point::Point(Eigen::Vector3d point, size_t index) : point(std::move(point)), index(index) {}

void Point::project_to_plane(const Plane &plane) {
    point_projected = plane.projectPoint(point);
}

double Point::get_distance_between_projection() {
    return (point - point_projected).norm();
}

void Point::transform_to_plane(const Plane &plane) {
    point_in_plane = plane.transformToPlane(point);
}

Point::Point() = default;
