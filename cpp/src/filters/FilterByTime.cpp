//
// Created by Константин Носорев on 06.05.2023.
//

#include "FilterByTime.h"

FilterByTime::FilterByTime(double windowSize, double tStep) : windowSize(windowSize), tStep(tStep) {}

UIntDataFrame FilterByTime::process(const UIntDataFrame &df) {
    auto max_t = get_max_t(df);
    auto window = _find_max_window(df, max_t);
    auto from_t = window.first;
    auto to_t = window.second;
    auto t_c_window_fun = [&from_t, &to_t](const uint &, const double &val) -> bool {
        return (val >= from_t && val <= to_t);
    };
    return df.get_data_by_sel<double, decltype(t_c_window_fun), uint, double>(
            "t_c", t_c_window_fun);
}

double FilterByTime::get_max_t(const UIntDataFrame &df) {
    MaxVisitor<double, UIntDataFrame::IndexType> maxTCVisitor;
    df.visit<double, MaxVisitor<double, UIntDataFrame::IndexType>>("t_c", maxTCVisitor);
    return ceil(maxTCVisitor.get_result());;
}

size_t FilterByTime::_process_window(const UIntDataFrame &df, double from_t, double to_t) {
    auto t_c_window_fun = [&from_t, &to_t](const uint &, const double &val) -> bool {
        return (val >= from_t && val <= to_t);
    };
    auto currWindow = df.get_view_by_sel<double, decltype(t_c_window_fun), uint, double>(
            "t_c", t_c_window_fun);
    return currWindow.get_column<double>("t_c").size();
}

std::pair<double, double> FilterByTime::_find_max_window(const UIntDataFrame &df, double max_t) {
    double max_window = 0;
    size_t max_count_in_window = 0;
    for (int delta_t = 0; delta_t <= ceil((max_t - windowSize) / tStep); delta_t += 1) {
        double from_t = delta_t * tStep;
        double to_t = windowSize + from_t;
        auto countValuesInWindow = _process_window(df, from_t, to_t);
        if (countValuesInWindow > max_count_in_window) {
            max_count_in_window = countValuesInWindow;
            max_window = from_t;
        }
    }
    double from_t = max_window;
    double to_t = windowSize + max_window;
    return std::make_pair(from_t, to_t);
}
