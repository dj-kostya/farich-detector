//
// Created by Константин Носорев on 06.05.2023.
//

#ifndef DETECTOR_CPP_FILTERBYTIME_H
#define DETECTOR_CPP_FILTERBYTIME_H

#include <DataFrame/DataFrame.h>
#include "IFilter.h"


class FilterByTime: public IFilter {
private:
    double windowSize;
    double tStep;

    static size_t _process_window(const UIntDataFrame &df, double from_t, double to_t);
    std::pair<double, double> _find_max_window(const UIntDataFrame &df, double max_t);
public:
    FilterByTime(double windowSize, double tStep);

    UIntDataFrame process(const UIntDataFrame &df) override;

    static double get_max_t(const UIntDataFrame &df);
};


#endif //DETECTOR_CPP_FILTERBYTIME_H
