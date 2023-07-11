//
// Created by Константин Носорев on 07.05.2023.
//

#ifndef DETECTOR_CPP_IFILTER_H
#define DETECTOR_CPP_IFILTER_H

#include <DataFrame/DataFrame.h>

using namespace hmdf;
using UIntDataFrame = StdDataFrame<uint>;

class IFilter {
public:
    virtual UIntDataFrame process(const UIntDataFrame &df) = 0;

    virtual std::string &get_name() = 0;
};

#endif //DETECTOR_CPP_IFILTER_H
