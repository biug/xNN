/*
* random_generator.hpp
*
*  Created on: 2016Äê6ÔÂ5ÈÕ
*      Author: Administrator
*/

#ifndef NORMAL_GENERATOR_HPP_
#define NORMAL_GENERATOR_HPP_

#include <random>

#include "random_generator.hpp"

template<typename DType, typename RandomEngine = std::default_random_engine>
class NormalGenerator : public RandomGenerator<DType, RandomEngine> {
	std::normal_distribution<DType> m_Distribution;
public:
	NormalGenerator(DType miu, DType sigma) : m_Distribution(miu, sigma) {}
	~NormalGenerator() {}

	DType generate() override {
		return m_Distribution(m_Engine);
	}
};

#endif