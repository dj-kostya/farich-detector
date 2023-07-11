#include <iostream>

#include "rapidcsv.h"
#include "src/metrics.h"
#include "src/benchmarks.h"
#include "src/farich/pipeline.h"
#include "src/geometry/Ellipse.h"
#include "src/filters/FilterInPlane.h"


const double windowSize = 2.5;
const double tStep = 0.05;
const double eps = 0.1;


int main() {
    spdlog::set_level(spdlog::level::info);
    int test_size = 100;

    std::vector<double> metrics_recall(test_size);
    std::vector<double> metrics_size(test_size);
    std::vector<double> metrics_time(test_size);
    UIntDataFrame dataFrame;
    dataFrame.read("dataset_100_with_noise_2000.csv", io_format::csv2);
    for (uint test_idx = 0; test_idx < test_size; test_idx++) {
        spdlog::info("TEST: start pipeline test {:d}/{:d}", test_idx, test_size);
        UIntDataFrame onlyFirstEntry = dataFrame.get_data_by_idx<uint, double>(
                Index2D<UIntDataFrame::IndexType>{test_idx, test_idx});
//        std::vector<IFilter *> filters = {new FilterByTime(windowSize, tStep), new FilterInPlane(200, 1)};
//        std::vector<IFilter *> filters = {new FilterByTime(windowSize, tStep)};
//        std::vector<IFilter *> filters = {new FilterInPlane(200, 1)};
//        std::vector<IFilter *> filters = {new FilterInPlane(200, 1), new FilterByTime(windowSize, tStep)};
        std::vector<IFilter *> filters = {};
        UIntDataFrame interestingWindow = onlyFirstEntry;
        clock_t start = clock();
        for (const auto filter: filters) {
            spdlog::debug("FILTERS: run {:}", filter->get_name());
            clock_t start_filt = clock();
            interestingWindow = filter->process(interestingWindow);
            auto end_filt = (double) (clock() - start_filt) / CLOCKS_PER_SEC;
            spdlog::debug("FILTERS: shape after {:} is {:d}", filter->get_name(), interestingWindow.shape().first);
            spdlog::debug("FILTERS: finish {:} by {:f}", filter->get_name(), end_filt);
        }
        auto time_filter = (double) (clock() - start) / CLOCKS_PER_SEC;
//        std::map<uint, Point> points;
        auto points_size = interestingWindow.shape().first;
        std::vector<Point> points;
        points.reserve(points_size);
        for (size_t i = 0; i < points_size; i++) {
            auto row_i = interestingWindow.get_row<uint, double, double>(i);
            auto p_i = get_vector_from_row(row_i);
            auto subentry = row_i.at<uint>(1);
            points.emplace_back(p_i, subentry);
        }
        spdlog::info("TEST: finished applying filters by {:f}", time_filter);

        auto [indexes, elipses, time] = pipeline(
                points,
                eps,
                3
        );
        spdlog::info("TEST: finished applying pipeline by {:f}", time);
//    std::cout << "Bench ransac" << std::endl;
//    benchmark::bench_plane_algs(Ellipse::use_RANSAC);
//    std::cout << "Bench OPENCV" << std::endl;
//    benchmark::bench_plane_algs(Ellipse::use_OPENCV);
        auto df_size = onlyFirstEntry.shape().first;
        std::vector<uint> signal_true(df_size);
        std::vector<uint> signal_pred(df_size);
        for (int i = 0; i < df_size; i++) {
            auto row_i = onlyFirstEntry.get_row<uint, double, double>(i);
            auto subentry_idx = row_i.at<uint>(1);
            signal_true[subentry_idx] = row_i.at<uint>(4);
            if (indexes.contains(subentry_idx)) {
                signal_pred[subentry_idx] = 1;
            } else {
                signal_pred[subentry_idx] = 0;
            }
        }
        metrics_recall[test_idx] = (recall(signal_true, signal_pred));
        metrics_size[test_idx] = ((double) indexes.size() / (double) df_size);
        metrics_time[test_idx] = time;
        spdlog::info("TEST: finished {:d} by {:}", test_idx, time);
    }

    auto recall_mean = vector_mean(metrics_recall);
    auto recall_max = vector_max(metrics_recall);
    auto recall_min = vector_min(metrics_recall);

    std::cout << "Recall: max " << recall_max << " mean " << recall_mean << " min " << recall_min << std::endl;

    auto size_mean = vector_mean(metrics_size);
    auto size_max = vector_max(metrics_size);
    auto size_min = vector_min(metrics_size);
    std::cout << "Size: max " << size_max << " mean " << size_mean << " min " << size_min << std::endl;

    auto time_mean = vector_mean(metrics_time);
    auto time_max = vector_max(metrics_time);
    auto time_min = vector_min(metrics_time);
    std::cout << "Time: max " << time_max << " mean " << time_mean << " min " << time_min << std::endl;
    return 0;
}
