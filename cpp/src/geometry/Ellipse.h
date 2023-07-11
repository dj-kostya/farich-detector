//
// Created by Константин Носорев on 30.04.2023.
//

#ifndef DETECTOR_CPP_ELLIPSE_H
#define DETECTOR_CPP_ELLIPSE_H


#include <Eigen/Core>
#include <vector>
#include "Point.h"

struct EllipseParams {
    double a, b, c, d, e, f;
};


class Ellipse {
private:
    Eigen::Vector3d center;
    std::vector<Point> inliner;

    static Eigen::Vector3d get_center(const EllipseParams &params);

    static bool check_invariants(const EllipseParams &params);

public:
    double mean_distance;

    Ellipse(const std::vector<Point> &inliners, Eigen::Vector3d center, const double &mean_distance);

    static Ellipse use_RANSAC(const std::vector<Point> &points, double threshold);

    static Ellipse use_OPENCV(const std::vector<Point> &points, double threshold);

    static Ellipse use_RANSAC_V2(const std::vector<Point> &points, double threshold);

    Ellipse() = default;

    [[nodiscard]] const std::vector<Point> &getInliner() const {
        return inliner;
    }
};


#endif //DETECTOR_CPP_ELLIPSE_H
