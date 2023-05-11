#include <iostream>
#include <DataFrame/DataFrame.h>
#include <Eigen/Core>
#include <Eigen/geometry>
#include <set>
#include "src/geometry/Plane.h"
#include "src/geometry/Ellipse.h"
#include "src/filters/FilterByTime.h"
#include "src/filters/FilterInPlane.h"
#include "src/geometry/Point.h"

using namespace hmdf;
using UIntDataFrame = StdDataFrame<uint>;

const double windowSize = 2.5;
const double tStep = 0.05;
const double eps = 0.05;


Eigen::Vector3d get_vector_from_row(const HeteroVector &row) {
    auto t = row.at<double>(3);
    auto x = row.at<uint>(2);
    auto y = row.at<uint>(3);

    return {x, y, t};
}


int main() {
    UIntDataFrame dataFrame;
    // Успростили задачу
    dataFrame.read("dataset_100_with_noise_2e5.csv", io_format::csv2);
    const UIntDataFrame onlyFirstEntry = dataFrame.get_data_by_idx<uint, double>(
            Index2D<UIntDataFrame::IndexType>{0, 0});

    IFilter *filters[] = {new FilterByTime(windowSize, tStep)}; // Константы нужно из экспиремента
    UIntDataFrame interestingWindow = onlyFirstEntry;
    for (auto filter: filters) {
        interestingWindow = filter->process(onlyFirstEntry);
    }
    interestingWindow.write<uint, double>("dataset_after_filters.csv", io_format::csv2);
    double min_mean = 9999999;
    std::vector<Point> result;
    std::set < std::set < size_t >> processed_points;
    auto max_count_in_window = interestingWindow.shape().first;
    std::vector<Point> points(max_count_in_window);
    for (size_t i = 0; i < max_count_in_window; i++) {
        auto row_i = interestingWindow.get_row<uint, double, double>(i);
        auto p_i = get_vector_from_row(row_i);
        auto subentry = row_i.at<uint>(1);
        points[i] = Point(p_i, subentry);
    }
//    max_count_in_window = 18;
    for (size_t i = 0; i < max_count_in_window; i++) {
        auto point_i = points[i];
        if (i % 10 == 0) {
            std::cout << i << "/" << max_count_in_window << "(" << min_mean << '/' << result.size() << ")" << std::endl;
        }
        for (size_t j = max_count_in_window - 1; j >= i + 1; j--) {
            auto point_j = points[j];
            for (size_t k = j + 1; k < max_count_in_window; k++) {
                auto point_k = points[k];
                std::vector<Eigen::Vector3d> cur_points = {point_i.point, point_j.point, point_k.point};
                Plane curPlane(cur_points);
                if (fabs(curPlane.normal.x()) > 0.002) {
                    continue;
                }

                double angel_between = curPlane.normal.z() / curPlane.normal.norm();
                double angel = std::acos(angel_between) * 180 / M_PI;
                if ((curPlane.normal.z() <= 0 && angel <= 90) || (curPlane.normal.z() > 0 && angel >= 90)) {
                    continue;
                }
                cur_points.clear();
                std::set < size_t > cur_indexes = {i, j, k};
                std::vector<Point> projected_points = {point_i, point_j, point_k};

                for (size_t c = k + 1; c < max_count_in_window; c++) {
                    auto point_c = points[c];
                    point_c.project_to_plane(curPlane);
                    if (point_c.get_distance_between_projection() > eps) {
                        continue;
                    }
                    projected_points.push_back(point_c);
                    cur_indexes.insert(c);
                }
                for (auto &p: projected_points) {
                    p.transform_to_plane(curPlane);
                }
                if (projected_points.size() < 6) {
                    continue;
                }
                if (processed_points.contains(cur_indexes)) {
                    std::cout << "Find collision" << std::endl;
                    continue;
                }
                processed_points.insert(cur_indexes);
                Ellipse el = Ellipse::use_OPENCV(projected_points, 2);
//                Ellipse el = Ellipse::use_RANSAC(projected_points, 2);
                if (el.getInliner().size() <= 8) continue;
//                if (el.getInliner().size() > result.size()) {
//                    result = el.getInliner();
//                    min_mean = el.mean_distance;
//                }
                if (el.mean_distance < min_mean ||
                    (el.mean_distance == min_mean & el.getInliner().size() > result.size())) {
                    result = el.getInliner();
                    min_mean = el.mean_distance;
                }
            }
        }
    }
    std::set < uint > result_idxes;
    for (const auto &v: result) {
        result_idxes.insert(v.index);
        auto p = v.point;
        std::cout << "[" << p.x() << "," << p.y() << "," << p.z() << "]," << std::endl;
    }
    auto t_c_window_fun = [&result_idxes](const uint &, const uint &val) -> bool {
        return (result_idxes.contains(val));
    };
    auto result_view = interestingWindow.get_data_by_sel<uint, decltype(t_c_window_fun), uint, double>(
            "subentry", t_c_window_fun);
    result_view.write<uint, double>("dataset_result.csv", io_format::csv2);
    return 0;
}
