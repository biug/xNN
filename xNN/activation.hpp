/*
* activation.hpp
*
*  Created on: 2016Äê6ÔÂ5ÈÕ
*      Author: Administrator
*/

#ifndef ACTIVATION_HPP_
#define ACTIVATION_HPP_

#include <cmath>

template<typename DType>
class Sigmod {
public:
	inline void operator()(DType * output, const DType * input, int len) {
		for (int i = 0; i < len; ++i) {
			// f(x) = 1 / (1 + exp(-x))
			output[i] = 1 / (1 + exp(-input[i]));
		}
	}
};

template<typename DType>
class PartialSigmod {
public:
	inline void operator()(DType * output, const DType * input, int len) {
		for (int i = 0; i < len; ++i) {
			// f'(x) = exp(-x) / (1 + exp(-x))^2 = f(x) * (1 - f(x))
			output[i] = 1 / (1 + exp(-input[i]));
			output[i] = output[i] * (1 - output[i]);
		}
	}
};

template<typename DType>
class Cubic {
public:
	inline void operator()(DType * output, const DType * input, int len) {
		for (int i = 0; i < len; ++i) {
			// f(x) = x^3
			output[i] = input[i] * input[i] * input[i];
		}
	}
};

template<typename DType>
class PartialCubic {
public:
	inline void operator()(DType * output, const DType * input, int len) {
		for (int i = 0; i < len; ++i) {
			// f(x) = 3 * x^2
			output[i] = static_cast<DType>(3.0) * input[i] * input[i];
		}
	}
};

#endif
