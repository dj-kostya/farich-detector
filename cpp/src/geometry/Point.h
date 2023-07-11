//
// Created by Константин Носорев on 06.05.2023.
//

#ifndef DETECTOR_CPP_POINT_H
#define DETECTOR_CPP_POINT_H


#include <Eigen/Core>
#include "Plane.h"

class Point {
private:

public:
    Eigen::Vector3d point;
    Eigen::Vector3d point_in_plane;
    Eigen::Vector3d point_projected;
    size_t index;

    Point();

    Point(double x, double y, size_t index);

    explicit Point(Eigen::Vector3d point, size_t index);

    void project_to_plane(const Plane &plane);

    void transform_to_plane(const Plane &plane);

    double get_distance_between_projection();


};


#endif //DETECTOR_CPP_POINT_H
