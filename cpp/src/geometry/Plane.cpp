//
// Created by Константин Носорев on 30.04.2023.
//

#include "Plane.h"
#include <eigen3/Eigen/Geometry>

Plane::Plane(const std::vector<Eigen::Vector3d> &c) {
    size_t num_atoms = c.size();
    if (num_atoms < 3) {
        throw std::runtime_error("Not enough points to create plane");
    }
    Eigen::Matrix<typename Eigen::Vector3d::Scalar, Eigen::Dynamic, Eigen::Dynamic> coord(3, num_atoms);
    for (Eigen::Index i = 0; i < num_atoms; ++i) coord.col(i) = c[i];

    // calculate _centroid
    Eigen::Vector3d _centroid = coord.rowwise().mean();
    coord.colwise() -= _centroid;

    // we only need the left-singular matrix here
    //  http://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
    auto svd = coord.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto Umatrix = svd.matrixU();
    Eigen::Vector3d plane_normal = Umatrix.rightCols(1);

    normal = plane_normal;
    centroid = _centroid;

    auto v1 = c[0] - centroid;
    auto v2 = c[1] - centroid;
    auto v3 = normal + centroid;
    auto v1_normalized = v1 / v1.norm();
    auto v2_normalized = v2 / v2.norm();
    auto v3_normalized = v3 / v3.norm();

    T.col(0) = v1_normalized;
    T.col(1) = v2_normalized;
    T.col(2) = v3_normalized;

    T_inv = Eigen::Inverse(T);
}

Eigen::Vector3d Plane::projectPoint(const Eigen::Vector3d &point) const{
    auto vector = (point - centroid);
    auto vector_point = normal * Eigen::Transpose(vector);
    return point - vector_point * normal;
}

Eigen::Vector3d Plane::transformToPlane(const Eigen::Vector3d &point) const{
    return T_inv * (point - centroid);
}

Eigen::Vector3d Plane::transformFromPlane(const Eigen::Vector3d &point) const{
    return T * point + centroid;
}
