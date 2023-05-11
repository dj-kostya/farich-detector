//
// Created by Константин Носорев on 30.04.2023.
//

#ifndef DETECTOR_CPP_PLANE_H
#define DETECTOR_CPP_PLANE_H

#include <vector>
#include <locale>
#include <Eigen/Core>

class Plane {
private:
    Eigen::Matrix<Eigen::Vector3d::Scalar, 3, 3> T, T_inv;

public:
    Eigen::Vector3d centroid, normal;
    explicit Plane(const std::vector<Eigen::Vector3d> &points);

    [[nodiscard]] Eigen::Vector3d projectPoint(const Eigen::Vector3d &point) const;

    [[nodiscard]] Eigen::Vector3d transformToPlane(const Eigen::Vector3d &point) const;

    [[nodiscard]] Eigen::Vector3d transformFromPlane(const Eigen::Vector3d &point) const;
};


#endif //DETECTOR_CPP_PLANE_H
