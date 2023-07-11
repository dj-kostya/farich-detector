//
// Created by Константин Носорев on 15.05.2023.
//

#ifndef DETECTOR_CPP_METRICS_H
#define DETECTOR_CPP_METRICS_H

#include <vector>
#include <locale>

double recall(std::vector<uint> &signal, std::vector<uint> &pred_signal);

template<typename T>
T vector_mean(const std::vector<T> &vec) {
    auto n = vec.size();
    T sum = 0;
    T result_cnt = n;
    for (auto i = 0; i < n; i++) {
        if (isnan(vec[i]) || isinf(vec[i])){
            result_cnt--;
            continue;
        }
        sum += vec[i];
    }

    return sum / (T) result_cnt;
}

template<typename T>
T vector_max(const std::vector<T> &vec) {
    T max = 0;

    for (auto i = 0; i < vec.size(); i++) {
        max = fmax(max, vec[i]);
    }

    return max;
}
template<typename T>
T vector_min(const std::vector<T> &vec) {
    T min = std::numeric_limits<T>::infinity();

    for (auto i = 0; i < vec.size(); i++) {
        min = fmin(min, vec[i]);
    }

    return min;
}
#endif //DETECTOR_CPP_METRICS_H
