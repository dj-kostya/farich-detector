//
// Created by Константин Носорев on 30.04.2023.
//

#ifndef DETECTOR_CPP_ELLIPSE_H
#define DETECTOR_CPP_ELLIPSE_H


#include <Eigen/Core>
#include <vector>
#include "Point.h"

class Ellipse {
private:
    Eigen::Vector3d center;
    std::vector<Point> inliner;
public:
    double mean_distance;

    Ellipse(const std::vector<Point> &inliners, Eigen::Vector3d center, const double &mean_distance);

    static Ellipse use_RANSAC(const std::vector<Point> &points, double threshold);
    static Ellipse use_OPENCV(const std::vector<Point> &points, double threshold);

    Ellipse() = default;

    const std::vector<Point> &getInliner() {
        return inliner;
    }
};


#endif //DETECTOR_CPP_ELLIPSE_H
