//
// Created by Константин Носорев on 30.04.2023.
//

#include "Ellipse.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <utility>

using cv::Point2f;

Ellipse::Ellipse(const std::vector<Point> &inliners, Eigen::Vector3d center, const double &mean_distance)
        : inliner(inliners), center(std::move(center)), mean_distance(mean_distance) {}

Ellipse Ellipse::use_RANSAC(const std::vector<Point> &points, double threshold) {
    Eigen::MatrixXd A(points.size(), 5);
    Eigen::VectorXd B(points.size());
    //    // Compute coefficients a, b, c, d, e for all points
    for (Eigen::Index i = 0; i < points.size(); ++i) {
        double x = points[i].point_in_plane[0];
        double y = points[i].point_in_plane[1];
        A(i, 0) = x * x;
        A(i, 1) = x * y;
        A(i, 2) = y * y;
        A(i, 3) = x;
        A(i, 4) = y;
        B(i) = 1;
    }

    // Solve for coefficients using all points
    Eigen::VectorXd X = A.colPivHouseholderQr().solve(B);
    double a = X(0);
    double b = X(1);
    double c = X(2);
    double d = X(3);
    double e = X(4);
    std::vector<Point> inliner;
    inliner.reserve(points.size());
    // Identify points within distance threshold of ellipse
    for (auto &p: points) {
        auto point = p.point_in_plane;
//        double distance =
//                pow(p.point_in_plane.x() - d, 2) * a +
//                (p.point_in_plane.x() - d) * (p.point_in_plane.y() - e) * b +
//                pow(p.point_in_plane.y() - e, 2) * c;

        double distance = pow(point.x(), 2) * a +
                          point.x() * point.y() * b + pow(point.y(), 2) * c +
                          point.x() * d + point.y() * e - 1;
        if (fabs(distance) <= threshold) {
            inliner.emplace_back(p);
        }
    }

    // Re-compute coefficients using inliers only
    A.resize(inliner.size(), 5);
    B.resize(inliner.size());
    for (Eigen::Index i = 0; i < inliner.size(); ++i) {
        double x = inliner[i].point_in_plane[0];
        double y = inliner[i].point_in_plane[1];
        A(i, 0) = x * x;
        A(i, 1) = x * y;
        A(i, 2) = y * y;
        A(i, 3) = x;
        A(i, 4) = y;
        B(i) = 1;
    }

    // Solve for coefficients using inliers only
    X = A.colPivHouseholderQr().solve(B);
    a = X(0);
    b = X(1);
    c = X(2);
    d = X(3);
    e = X(4);

    auto is_ellipse = check_invariants(EllipseParams{a, b, c, d, e, 1});
    if(!is_ellipse){
        throw std::runtime_error("Not an ellipse =<");
    }
    double delta = b * b - 4 * a * c;
    double xc = (2 * c * d - b * e) / delta;
    double yc = (2 * a * e - b * d) / delta;

    Eigen::Vector3d center = {xc, yc, 0};
    inliner.clear();
    double distance_sum = 0;
    for (auto &p: points) {
        auto point = p.point_in_plane;
//        double distance =
//                pow(p.point_in_plane.x() - d, 2) * a +
//                (p.point_in_plane.x() - d) * (p.point_in_plane.y() - e) * b +
//                pow(p.point_in_plane.y() - e, 2) * c;
        double distance = pow(point.x(), 2) * a +
                          point.x() * point.y() * b + pow(point.y(), 2) * c +
                          point.x() * d + point.y() * e - 1;
        if (fabs(distance) <= threshold) {
            inliner.emplace_back(p);
            distance_sum = fmax(fabs(distance), distance_sum);
        }
    }
    double mean_distance = distance_sum;
    return {
            inliner,
            center,
            mean_distance
    };
}

Ellipse Ellipse::use_OPENCV(const std::vector<Point> &points, double threshold) {
    std::vector<Point2f> points_;
    points_.reserve(points.size());
    for (auto &p: points) {
        points_.emplace_back(p.point_in_plane.x(), p.point_in_plane.y());
    }
    auto ellipse = cv::fitEllipse(points_);
    double x0 = ellipse.center.x;
    double y0 = ellipse.center.y;
    double A = ellipse.size.width / 2;
    double B = ellipse.size.height / 2;
    double theta = ellipse.angle * CV_PI / 180.0;
    auto a = pow(cos(theta), 2) / pow(A, 2);
    auto b = 2 * cos(theta) * sin(theta) / (pow(A, 2) * pow(B, 2));
    auto c = pow(sin(theta), 2) / pow(B, 2);
    auto d = -2 * x0 * pow(cos(theta), 2) / pow(A, 2) - 2 * y0 * cos(theta) * sin(theta) / (pow(A, 2) * pow(B, 2));
    auto e = -2 * y0 * pow(sin(theta), 2) / pow(B, 2);
    auto f = (pow(x0, 2) * pow(cos(theta), 2) + 2 * x0 * y0 * cos(theta) * sin(theta) +
              pow(y0 * sin(theta), 2)) / pow(A, 2) +
             (pow(x0 * sin(theta), 2) - 2 * x0 * y0 * cos(theta) * sin(theta) +
              pow(y0 * cos(theta), 2)) / pow(B, 2) - 1;

    auto is_ellipse = check_invariants(EllipseParams{a, b, c, d, e, f});
    if(!is_ellipse){
        throw std::runtime_error("Not an ellipse =<");
    }
    Eigen::Vector3d center = {x0, y0, 0};
    std::vector<Point> inliner;
    double distance_sum = 0;
    for (auto &p: points) {
        auto point = p.point_in_plane;
//        double distance =
//                pow(p.point_in_plane.x() - d, 2) * a +
//                (p.point_in_plane.x() - d) * (p.point_in_plane.y() - e) * b +
//                pow(p.point_in_plane.y() - e, 2) * c;
        double distance = pow(point.x(), 2) * a +
                          point.x() * point.y() * b + pow(point.y(), 2) * c +
                          point.x() * d + point.y() * e + f;
        double sq_error = pow(distance, 2);
        if (sq_error <= threshold) {
            inliner.emplace_back(p);
            distance_sum += sq_error;
        }
    }
    distance_sum /= inliner.size();
    return {
            inliner,
            center,
            distance_sum
    };
}

Ellipse Ellipse::use_RANSAC_V2(const std::vector<Point> &points, double threshold) {
    Eigen::MatrixXd A(points.size(), 6);
    for (Eigen::Index i = 0; i < points.size(); ++i) {
        double x = points[i].point_in_plane[0];
        double y = points[i].point_in_plane[1];
        A(i, 0) = x * x;
        A(i, 1) = x * y;
        A(i, 2) = y * y;
        A(i, 3) = x;
        A(i, 4) = y;
        A(i, 5) = 1;
    }
    Eigen::MatrixXd S = A.transpose() * A;
    Eigen::MatrixXd C(6, 6);
    C(0, 2) = 2;
    C(1, 1) = -1;
    C(2, 0) = 2;
//    Eigen::Matrix3Xd Z = ;
    Eigen::EigenSolver<Eigen::MatrixXd> solver(S.inverse() * C);
    Eigen::Matrix<double, 6, 1> eigen_value = solver.eigenvalues().real().eval();
    Eigen::Matrix<double, 6, 6> eigen_vec = solver.eigenvectors().real().eval();
    Eigen::Array<bool, 6, 1> mask = (eigen_value.array() > 0.0).eval() && !(eigen_value.array().isInf());

    Eigen::MatrixXd a = eigen_vec.block(0, mask.count() - 1, eigen_vec.rows(), mask.count());
    auto params = EllipseParams{a(0, 0), a(1, 0), a(2, 0), a(3, 0), a(4, 0), a(5, 0)};

    auto center = get_center(params);
    return Ellipse();
}

Eigen::Vector3d Ellipse::get_center(const EllipseParams &params) {
    auto num = params.b * params.b - params.a * params.c;
    auto x0 = (params.c * params.d - params.b * params.f) / num;
    auto y0 = (params.a * params.f - params.b * params.d) / num;
    Eigen::Vector3d center = {x0, y0, 0};
    return center;
}

bool Ellipse::check_invariants(const EllipseParams &params) {

    double I1 = params.a + params.c;

    Eigen::Matrix<double, 2, 2> I2;
    I2(0, 0) = params.a;
    I2(0, 1) = params.b / 2;
    I2(1, 0) = params.b / 2;
    I2(1, 1) = params.c;

    if (I2.determinant() <= 0) {
        return false;
    }

    Eigen::Matrix<double, 3, 3> I3;
    I3(0, 0) = params.a;
    I3(0, 1) = params.b / 2;
    I3(0, 2) = params.d;
    I3(1, 0) = params.b / 2;
    I3(1, 1) = params.c;
    I3(1, 2) = params.e;
    I3(2, 0) = params.d;
    I3(2, 1) = params.e;
    I3(2, 2) = params.f;

    if (I1 * I3.determinant() < 0) {
        return true;
    }
    return false;
}
