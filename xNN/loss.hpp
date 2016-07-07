/*
* loss.hpp
*
*  Created on: 2016Äê6ÔÂ5ÈÕ
*      Author: Administrator
*/

#ifndef LOSS_HPP_
#define LOSS_HPP_

#include <cmath>

#include "macros.h"

template<typename DType>
class Softmax {
public:
	inline void operator()(DType * output, const DType * input, int len) {
		DType sum = static_cast<DType>(GLOBAL_EPSILON);
		for (int i = 0; i < len; ++i) {
			sum += exp(input[i]);
		}
		for (int i = 0; i < len; ++i) {
			// f(x) = exp(x) / sum
			output[i] =  -log(exp(input[i]) / sum);
		}
	}
};

template<typename DType>
class PartialSoftmax {
public:
	inline void operator()(DType * output, const DType * input, int correctLabel, int len) {
		DType sum = static_cast<DType>(GLOBAL_EPSILON);
		for (int i = 0; i < len; ++i) {
			sum += exp(input[i]);
		}
		for (int i = 0; i < len; ++i) {
			// f(x) = exp(x) / sum
			output[i] = exp(input[i]) / sum;
		}
		output[correctLabel] -= 1;
	}
};

#endif
