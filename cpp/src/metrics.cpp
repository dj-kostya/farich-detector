//
// Created by Константин Носорев on 15.05.2023.
//

#include "metrics.h"

double recall(std::vector<uint> &signal, std::vector<uint> &pred_signal) {
    double true_positives = 0;
    double false_negatives = 0;

    for (size_t i = 0; i < signal.size(); i++) {
        if (signal[i] == 1 && pred_signal[i] == 1) {
            true_positives++;
        } else if (signal[i] == 1 && pred_signal[i] == 0) {
            false_negatives++;
        }
    }

    return true_positives / (true_positives + false_negatives);
}

//template<typename T>
//T vector_mean(const std::vector<T> &vec) {
//    auto n = vec.size();
//    T sum = 0;
//
//    for (auto i = 0; i < n; i++) {
//        sum += vec[i];
//    }
//
//    return sum / (T) n;
//}
