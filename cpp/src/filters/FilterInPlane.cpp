//
// Created by Константин Носорев on 06.05.2023.
//

#include "FilterInPlane.h"

FilterInPlane::FilterInPlane(uint windowSize, uint dStep) : windowSize(windowSize), dStep(dStep) {}

UIntDataFrame FilterInPlane::process(const UIntDataFrame &df) {
    auto min_x = get_min<uint>(df, "x_c");
    auto min_y = get_min<uint>(df, "y_c");
    auto max_x = get_max<uint>(df, "x_c");
    auto max_y = get_max<uint>(df, "y_c");

    uint max_dx = 0;
    uint max_dy = 0;
    uint max_cur_point = 0;

    for (auto from_x = min_x; from_x <= (max_x - windowSize); from_x += dStep) {
        auto to_x = from_x + windowSize;
        auto x_c_window_fun = [&from_x, &to_x](const uint &, const uint &val) -> bool {
            return (val >= from_x && val <= to_x);
        };
        auto curr_x_window = df.get_view_by_sel<uint, decltype(x_c_window_fun), uint, double>(
                "x_c", x_c_window_fun);
        for (auto from_y = min_y; from_y < (max_y - windowSize); from_y += dStep) {
            auto to_y = from_y + windowSize;
            auto y_c_window_fun = [&from_y, &to_y](const uint &, const uint &val) -> bool {
                return (val >= from_y && val <= to_y);
            };
            auto curr_y_window = df.get_view_by_sel<uint, decltype(y_c_window_fun), uint, double>(
                    "y_c", y_c_window_fun);

            auto window_size = curr_y_window.shape().first;

            if (window_size > max_cur_point) {
                max_dx = from_x;
                max_dy = from_y;
                max_cur_point = window_size;
            }

        }
    }
    auto window_fun = [&max_dx, &max_dy, this](const uint &, const uint &val_x, const uint &val_y) -> bool {
        return (val_x >= max_dx && val_x <= max_dx + windowSize) & (val_y >= max_dy && val_y <= max_dy + windowSize);
    };

    return df.get_data_by_sel<uint, uint, decltype(window_fun), uint, double>("x_c", "y_c", window_fun);
}

template<typename T>
T FilterInPlane::get_max(const UIntDataFrame &df, const char *column) {
    MaxVisitor<T, UIntDataFrame::IndexType> maxTCVisitor;
    df.visit<T, MaxVisitor<T, UIntDataFrame::IndexType>>(column, maxTCVisitor);
    return ceil(maxTCVisitor.get_result());;
}

template<typename T>
T FilterInPlane::get_min(const UIntDataFrame &df, const char *column) {
    MinVisitor<T, UIntDataFrame::IndexType> minVisitor;
    df.visit<T, MinVisitor<T, UIntDataFrame::IndexType>>(column, minVisitor);
    return ceil(minVisitor.get_result());;
}
