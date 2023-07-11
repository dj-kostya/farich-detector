//
// Created by Константин Носорев on 16.05.2023.
//

#ifndef DETECTOR_CPP_PIPELINE_H
#define DETECTOR_CPP_PIPELINE_H

#include <DataFrame/DataFrame.h>
#include <Eigen/Core>
#include <Eigen/geometry>
#include <set>
#include "spdlog/spdlog.h"
#include "../geometry/Point.h"
#include "../geometry/Ellipse.h"
#include "../filters/FilterByTime.h"

using namespace hmdf;
using UIntDataFrame = StdDataFrame<uint>;


Eigen::Vector3d get_vector_from_row(const HeteroVector &row) {
    auto t = row.at<double>(3);
    auto x = row.at<uint>(2);
    auto y = row.at<uint>(3);

    return {x, y, t};
}

std::tuple<std::set<uint>, std::vector<Ellipse>, double> pipeline(
        std::vector<Point> points,
        double eps,
        double delta,
        double ksi = 0.002,
        size_t min_ellipse_size = 8,
        size_t min_points_size = 6,
        double threshold = 0.8

) {
    clock_t start = clock();
//    UIntDataFrame interestingWindow = onlyFirstEntry;
//    for (const auto &filter: filters) {
//        spdlog::debug("FILTERS: Run {:}", filter->get_name());
//        clock_t start_filt = clock();
//        interestingWindow = filter->process(interestingWindow);
//        auto end_filt = (double) (clock() - start_filt) / CLOCKS_PER_SEC;
//        spdlog::debug("FILTERS: Shape after {:} is {:d}", filter->get_name(), interestingWindow.shape().first);
//        spdlog::debug("FILTERS: Finish {:} by {:f}", filter->get_name(), end_filt);
//    }
//    double min_mean = 9999999;
//    std::vector<Point> result;
//    std::set < std::set < size_t >> processed_points;
//    auto points_size = interestingWindow.shape().first;
//    std::vector<Point> points(points_size);
//    for (size_t i = 0; i < points_size; i++) {
//        auto row_i = interestingWindow.get_row<uint, double, double>(i);
//        auto p_i = get_vector_from_row(row_i);
//        auto subentry = row_i.at<uint>(1);
//        points[i] = Point(p_i, subentry);
//    }
//    points_size = 17;
    double min_mean = 9999999;
    std::set < std::set < size_t >> processed_points;
    std::vector<Ellipse> ellipses;
    std::vector<Point> result;
    auto points_size = points.size();
//    auto points_size = 17;
    for (size_t i = 0; i < points_size - 2; i++) {
        auto &point_i = points[i];
        if (i % 10 == 0) {
            spdlog::debug("PIPELINE: process {:d}/{:d} ({:f} / {:d})", i, points_size, min_mean,
                          result.size());
        }
        for (size_t j = i + 1; j < points_size - 1; j++) {
            auto &point_j = points[j];
            for (size_t k = j + 1; k < points_size; k++) {
                auto &point_k = points[k];
                Plane curPlane;
                {
                    std::vector<Eigen::Vector3d> cur_points = {point_i.point, point_j.point, point_k.point};
                    curPlane = Plane(cur_points);
                    if (fabs(curPlane.normal.x()) > ksi) {
                        continue;
                    }
                }


                double angel_between = curPlane.normal.z() / curPlane.normal.norm();
                double angel = std::acos(angel_between) * 180 / M_PI;
                if ((curPlane.normal.z() <= 0 && angel <= 90) || (curPlane.normal.z() > 0 && angel >= 90)) {
                    continue;
                }

                std::set < size_t > cur_indexes = {i, j, k};
                std::vector<Point> projected_points = {point_i, point_j, point_k};

                for (size_t c = k + 1; c < points_size; c++) {
                    auto &point_c = points[c];
                    point_c.project_to_plane(curPlane);
                    auto dist = point_c.get_distance_between_projection();
                    if (dist > eps) {
                        continue;
                    }
                    projected_points.push_back(point_c);
                    cur_indexes.insert(c);
                }
                for (auto &p: projected_points) {
                    p.transform_to_plane(curPlane);
                }
                if (projected_points.size() < min_points_size) {
                    continue;
                }
                if (processed_points.contains(cur_indexes)) {
                    continue;
                }
                processed_points.insert(cur_indexes);
                Ellipse el;
                try {
                    el = Ellipse::use_OPENCV(projected_points, delta);
//                    el = Ellipse::use_RANSAC(projected_points, delta);
//                    Ellipse el2 = Ellipse::use_RANSAC_V2(projected_points, 2);
                } catch (std::exception &) {
                    continue;
                }

                if (el.getInliner().size() <= min_ellipse_size) continue;
//                if (el.getInliner().size() > result.size()) {
//                    result = el.getInliner();
//                    min_mean = el.mean_distance;
//                }
                if (el.mean_distance < delta / 2) {
                    ellipses.push_back(el);
                }
//
//                if (el.mean_distance < min_mean ||
//                    (el.mean_distance == min_mean & el.getInliner().size() > result.size())) {
//                    result = el.getInliner();
//                    min_mean = el.mean_distance;
//                }
            }
        }
    }
    auto exec_time = (double) (clock() - start) / CLOCKS_PER_SEC;
    std::set < uint > result_idxes;
    for (const auto &el: ellipses) {
        for (const auto &p: el.getInliner()) {
            result_idxes.insert(p.index);
        }
    }
    return std::make_tuple(result_idxes, ellipses, exec_time);
}

#endif //DETECTOR_CPP_PIPELINE_H
