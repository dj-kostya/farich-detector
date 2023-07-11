//
// Created by Константин Носорев on 06.05.2023.
//

#ifndef DETECTOR_CPP_FILTERINPLANE_H
#define DETECTOR_CPP_FILTERINPLANE_H

#include <DataFrame/DataFrame.h>
#include "IFilter.h"


class FilterInPlane : public IFilter {
private:
    std::string filename = "FilterInPlane";

    uint windowSize;
    uint dStep;

    template<typename T>
    static T get_max(const UIntDataFrame &df, const char *column);

    template<typename T>
    static T get_min(const UIntDataFrame &df, const char *column);

public:
    FilterInPlane(uint windowSize, uint dStep);

    UIntDataFrame process(const UIntDataFrame &df) override;

    std::string &get_name() override{
        return filename;
    }
};


#endif //DETECTOR_CPP_FILTERINPLANE_H
